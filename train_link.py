import os
import argparse
import torch
import warnings
import pandas as pd
import numpy as np

from select_gpu import select_gpu
from asng import CategoricalASNG
from LP.read_data import LPReader, Sampler
from LP.base_model import BaseModule

parser = argparse.ArgumentParser(description="Parser for KG link prediction searching")

parser.add_argument('--data_path', type=str, default='data/WN18RR/', 
        help='the directory to dataset')
parser.add_argument('--seed', type=int, default=1234, 
        help="random seed number")
parser.add_argument('--hidden_size', type=int, default=64, 
        help="hidden dimension")
parser.add_argument('--test_batch_size', type=int, default=128, 
        help="batch size in testing procedure")
parser.add_argument('--epoch_per_test', type=int, default=2, 
        help="intervals for evaluation")
parser.add_argument('--max_length', type=int, default=7, 
        help="maximum length for the relational path")
parser.add_argument('--n_epoch', type=int, default=20, 
        help="number of epochs for training")
parser.add_argument('--alpha', type=int, default=0.7, 
        help="param alpha for the biased random walk")
parser.add_argument('--beta', type=int, default=0.5, 
        help="param beta for the biased random walk")
parser.add_argument('--drop', type=int, default=0.0, 
        help="dropout rate")
parser.add_argument('--lam', type=int, default=2, 
        help="number of models sampled for NG")
parser.add_argument('--mode', type=str, default='asng', 
        help="mode of sampling")
parser.add_argument('--out_file_info', type=str, default='hybrid', 
        help='extra string for the output file name')


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "6"
    os.environ["MKL_NUM_THREADS"] = "6"
    warnings.filterwarnings("ignore", category=FutureWarning)
    torch.cuda.set_device(select_gpu())

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    directory = 'results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # load data
    data = args.data_path.split('/')
    if len(data[-1]) > 0:
        data = data[-1]
    else:
        data = data[-2]
    args.out_filename = '%s/%s_%d_%s_%s.txt'%(directory, args.mode, args.seed, data, args.out_file_info)
    print('dataset:%s, seed:%d, search mode:%s'% (data, args.seed, args.mode))

    if data == 'WN18RR':
        args.learning_rate = 0.0036225411209161286
        args.decay_rate = 0.99184314910166
        args.batch_size = 512
        args.L2 = 7.096653356534566e-05
        #args.learning_rate = 0.00437
        #args.decay_rate = 0.987
        #args.batch_size = 512
        #args.L2 = 0.000047
        #args.drop = 0.24

    elif data == 'FB15K237':
        args.learning_rate = 0.002858371279116538
        args.decay_rate = 0.9911310581870849
        args.L2 = 0.0009525756060385333
        args.batch_size = 1024

    reader = LPReader()
    reader.read(data_path=args.data_path, opts=args)
    pather = Sampler(reader,args)

    args._ent_num = reader._ent_num
    args._rel_num = reader._rel_num
    print('#entities:%d, #relations:%d' %(args._ent_num, args._rel_num))

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

    # macro-level update
    C_out = [4, 13, 13]
    asng_out = CategoricalASNG(np.array(C_out), alpha=1.5, delta_init=1)
    run_round = 0
    best_hit1 = 0
    num_train = 0

    print('\n starting searching!')
    while(True):
        Ms = []
        ma_structs = []
        for i in range(args.lam):
            M = asng_out.sampling()
            struct = np.argmax(M, axis=1)
            Ms.append(M)
            ma_structs.append(list(struct))

        model = BaseModule(reader, args, ma_structs)
        valid_hit1, structs = model.train(train_data, valid_data, vfilter_mat, MAX_EPOCH=args.n_epoch)
        scores = - np.array(valid_hit1)

        if np.min(scores) < best_hit1:
            best_hit1 = np.min(scores)
            print('best_hit@1 changed:', best_hit1)
        if args.mode == 'asng':
            asng_out.update(np.array(Ms), scores, True)
        num_train += args.lam
        run_round += 1
        out_str = ''
        for i, struct in enumerate(structs):
            out_str += '\t' + str(struct) + ' H@1:' + str(valid_hit1[i])
        print("ROUND: %d. %d models trained. Best Hit@1: %.4f."%(run_round, num_train, -best_hit1), out_str)


