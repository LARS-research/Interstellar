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
from LP.read_data import LPReader, Sampler
from LP.model_search import Interstellar

parser = argparse.ArgumentParser(description="Parser for KG link prediction evaluation")

parser.add_argument('--data_path', type=str, default='data/WN18RR/', 
        help='the directory to dataset')
parser.add_argument('--hidden_size', type=int, default=256, 
        help="hidden dimension")
parser.add_argument('--test_batch_size', type=int, default=128, 
        help="batch size in testing procedure")
parser.add_argument('--learning_rate', type=float, default=0.0003, 
        help="learning rate")
parser.add_argument('--L2', type=float, default=0.00001, 
        help="weight decay")
parser.add_argument('--batch_size', type=int, default=1024, 
        help="batch size")
parser.add_argument('--decay_rate', type=float, default=0.99, 
        help="decay of learning rate")
parser.add_argument('--drop', type=float, default=0.3, 
        help="dropout rate")
parser.add_argument('--epoch_per_test', type=int, default=2, 
        help="intervals for evaluation")
parser.add_argument('--max_length', type=int, default=7, 
        help="maximum length for the relational path")
parser.add_argument('--struct', type=str, default=[0,0,0,0,0,0,0,0,0,0,0], 
        help="structure indicateor")
parser.add_argument('--n_epoch', type=int, default=50, 
        help="number of epochs for training")
parser.add_argument('--alpha', type=int, default=0.7, 
        help="param alpha for the biased random walk")
parser.add_argument('--beta', type=int, default=0.5, 
        help="param beta for the biased random walk")
parser.add_argument('--out_file_info', type=str, default='eval', 
        help='extra string for the output file name')

def test_link(data, filter_mat, model, args):
    batch_size = args.test_batch_size
    label = data[:,2]
    data, padding_num = utils.padding_LPdata(data, batch_size)
    num_batch = len(data) // batch_size

    probs = []
    for i in range(num_batch):
        seqs = torch.LongTensor(data[i*batch_size:(i+1)*batch_size]).cuda()
        probs.append(model.evaluate(seqs, args.struct).data.cpu().numpy())

    probs = np.concatenate(probs)[:len(data) - padding_num]
    filter_probs = probs * filter_mat
    filter_probs[range(len(label)), label] = probs[range(len(label)), label]
    filter_ranks = utils.cal_ranks(filter_probs, method='sort', label=label)
    f_mrr, f_h1, f_h10= utils.cal_performance(filter_ranks)
    return f_mrr, f_h1, f_h10


def run_model(train_data, valid_data, vfilter_mat, test_data, tfilter_mat, args):

    config = 'lr:%f, L2:%f, drop:%.2f, batch_size:%d, decay:%f\n' %(args.learning_rate, args.L2, args.drop, args.batch_size, args.decay_rate)
    model = Interstellar(args).cuda()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.L2)
    scheduler = ExponentialLR(optimizer, args.decay_rate)
    best_hit1 = 0
    early_stop = 0
    MAX_TIME = 5

    n_train = len(train_data)
    batch_size = args.batch_size
    n_batch = n_train // batch_size + int(n_train%batch_size>0)
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
            model.train()
            loss = model._loss(seqs, args.struct)
            loss.backward()
            optimizer.step()
            
            for n,p in model.named_parameters():
                if 'sub' in n or 'obj' in n:
                    X = p.data.clone()
                    Z = torch.norm(X, p=2, dim=1, keepdim=True)
                    Z[Z<1] = 1
                    X = X/Z
                    p.data.copy_(X.view(-1, args.hidden_size))
        scheduler.step()

        # evaluation
        if (epoch+1) % args.epoch_per_test == 0:
            model.eval()
            v_mrr, v_h1, v_h10 = test_link(valid_data, vfilter_mat, model, args)
            t_mrr, t_h1, t_h10 = test_link(test_data,  tfilter_mat, model, args)
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
    args.out_filename = '%s/LP_%s_%s.txt'%(directory, data, args.out_file_info)

    reader = LPReader()
    reader.read(data_path=args.data_path, opts=args)
    pather = Sampler(reader,args)

    args._ent_num = reader._ent_num
    args._rel_num = reader._rel_num
    print('data:%s,  #entities:%d,  #relations:%d' %(data, args._ent_num, args._rel_num))

    # load/sample the relational path
    sequence_datapath = os.path.join(args.data_path, 'paths_%.1f_%.1f' % (args.alpha, args.beta))
    if not os.path.exists(sequence_datapath):
        print('start to sample paths')
        pather.sample_paths()
        train_data = reader._train_data
    else:
        print('load existing training sequences')
        train_data = pd.read_csv(sequence_datapath, index_col=0)
    valid_data = reader._valid_data[['h_id', 'r_id', 't_id']].values
    test_data  = reader._test_data[['h_id', 'r_id', 't_id']].values
    vfilter_mat = reader._tail_valid_filter_mat
    tfilter_mat = reader._tail_test_filter_mat

    run_model(train_data, valid_data, vfilter_mat, test_data, tfilter_mat, args)
