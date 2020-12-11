import os
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import warnings
import pandas as pd
import numpy as np

import utils
from select_gpu import select_gpu
from EA.read_data import Reader, Sampler
from EA.model_search import Interstellar

parser = argparse.ArgumentParser(description="Parser for KG entity alignment evaluation")

parser.add_argument('--data_path', type=str, default='data/dbp_wd_15k_V1/mapping/0_3/', 
        help='the directory to dataset')
parser.add_argument('--hidden_size', type=int, default=256, 
        help="hidden dimension")
parser.add_argument('--learning_rate', type=float, default=0.0003, 
        help="learning rate")
parser.add_argument('--L2', type=float, default=0.0015, 
        help="weight decay")
parser.add_argument('--batch_size', type=int, default=1024, 
        help="batch size")
parser.add_argument('--decay_rate', type=float, default=0.99663, 
        help="decay of learning rate")
parser.add_argument('--drop', type=float, default=0.3, 
        help="dropout rate")
parser.add_argument('--test_batch_size', type=int, default=128, 
        help="batch size in testing procedure")
parser.add_argument('--epoch_per_test', type=int, default=1, 
        help="intervals for evaluation")
parser.add_argument('--max_length', type=int, default=15, 
        help="maximum length for the relational path")
parser.add_argument('--struct', type=str, default=[0,0,0,0,0,0,0,0,0,0,0], 
        help="structure indicateor")
parser.add_argument('--n_epoch', type=int, default=40, 
        help="number of epochs for training")
parser.add_argument('--alpha', type=float, default=0.9, 
        help="param alpha for the biased random walk")
parser.add_argument('--beta', type=float, default=0.9, 
        help="param beta for the biased random walk")
parser.add_argument('--out_file_info', type=str, default='eval', 
        help='extra string for the output file name')


def test_align(data, reader, model, struct, args):
    batch_size = args.test_batch_size
    label = data.kb_2 if args.kb_1to2==True else data.kb_1

    data, padding_num = utils.padding_EAdata(data, batch_size)
    s_em = model.sub_embed

    num_batch = len(data) // batch_size

    probs = []
    for i in range(num_batch):
        one_batch_data = data.iloc[i*batch_size: (i+1)*batch_size]
        if args.kb_1to2:
            e = torch.LongTensor(one_batch_data.kb_1.values).cuda()
        else:
            e = torch.LongTensor(one_batch_data.kb_2.values).cuda()

        h_embed = s_em(e)
        norm_embed = s_em.weight
        h_embed = h_embed / torch.norm(h_embed, dim=-1, keepdim=True)
        norm_embed = norm_embed / torch.norm(norm_embed, dim=-1, keepdim=True)
        values = torch.mm(h_embed, norm_embed.t())
        probs.append(values.data.cpu().numpy())

    probs = np.concatenate(probs)[:len(data) - padding_num]
    candi = reader._ent_testing.kb_2 if args.kb_1to2==True else reader._ent_testing.kb_1
    mask = np.in1d(np.arange(probs.shape[1]), candi)
    probs[:,~mask] = probs.min() - 1
    ranks = utils.cal_ranks(probs, method='sort', label=label)
    f_mrr, f_h1, f_h10= utils.cal_performance(ranks)
    return f_mrr, f_h1, f_h10

def run_model(train_data, valid_data, test_data, reader, args):
    config = 'lr:%f, L2:%f, drop:%.2f, batch_size:%d, decay:%f\n' %(args.learning_rate, args.L2, args.drop, args.batch_size, args.decay_rate)
    model = Interstellar(args).cuda()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.L2)
    scheduler = ExponentialLR(optimizer, args.decay_rate)
    MAX_TIME = 5

    n_train = len(train_data)
    batch_size = args.batch_size
    n_batch = n_train // batch_size + int(n_train%batch_size>0)
    early_stop = 0
    best_hit1 = 0
    best_str = ''
    for epoch in range(args.n_epoch):
        if early_stop > MAX_TIME:
            break

        # shuffle training data
        choices = np.random.choice(n_train, size=n_train, replace=False)

        for i in range(n_batch):
            start = i*batch_size
            end = min(n_train, (i+1) * batch_size)
            one_batch_choices = choices[start:end]
            one_batch_data = train_data.iloc[one_batch_choices]

            seqs = torch.LongTensor(one_batch_data.values[:, :args.max_length]).cuda()
            model.zero_grad()
            loss = model._loss(seqs, args.struct)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
        scheduler.step()

        # evaluation
        if (epoch+1) % args.epoch_per_test == 0:
            v_mrr, v_h1, v_h10 = test_align(valid_data, reader, model, args.struct, args)
            t_mrr, t_h1, t_h10 = test_align(test_data, reader, model, args.struct, args)
            if v_h1 > best_hit1:
                best_hit1 = v_h1
                early_stop = 0
                best_str = str(args.struct) + '\tepoch:%d  VALID mrr:%.3f  h1:%.3f  h10:%.3f\tTEST mrr:%.3f  h1:%.3f  h10:%.3f\n' % (epoch+1, v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10)
            else:
                early_stop += 1
    print(best_str)

    with open(args.out_filename, 'a') as f:
        f.write(config)
        f.write(best_str)
    return best_hit1

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "6"
    os.environ["MKL_NUM_THREADS"] = "6"
    warnings.filterwarnings("ignore", category=FutureWarning)
    torch.cuda.set_device(select_gpu())

    args = parser.parse_args()
    args.struct = utils.parse_struct(args.struct)

    directory = 'results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # load data
    data = args.data_path.split('/')[1]
    args.out_filename = '%s/EA_%s_%s.txt'%(directory, data, args.out_file_info)

    reader = Reader()
    reader.read(data_path=args.data_path)
    pather = Sampler(reader,args)
    args.kb_1to2 = True

    args._ent_num = reader._ent_num
    args._rel_num = reader._rel_num
    print('data:%s,  #entities:%d,  #relations:%d' % (data, args._ent_num, args._rel_num))

    # load/sample the relational path
    sequence_datapath = os.path.join(args.data_path, 'paths_%.1f_%.1f' % (args.alpha, args.beta))
    if not os.path.exists(sequence_datapath):
        print('start to sample paths')
        pather.sample_paths()
        train_data = reader._train_data
    else:
        print('load existing training sequences')
        train_data = pd.read_csv(sequence_datapath, index_col=0)
    data_size = len(reader._ent_testing)
    valid_data = reader._ent_testing.iloc[:data_size//10]
    test_data = reader._ent_testing.iloc[data_size//10:]

    run_model(train_data, valid_data, test_data, reader, args)

