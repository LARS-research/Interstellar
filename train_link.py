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

parser = argparse.ArgumentParser(description="Paser for KG link prediction")
parser.add_argument('--data_path', type=str, default='data/WN18RR/', help='the directory to dataset')
parser.add_argument('--seed', type=int, default=1234, help="random seed number")
parser.add_argument('--mode', type=str, default='asng', help="mode of sampling")
parser.add_argument('--out_file_info', type=str, default='hybrid', help='extra string for the output file name')

args = parser.parse_args()

class Options(object):
    pass

# init hyper-parameters
opts = Options()
opts.hidden_size = 64
opts.num_layers = 1
opts.test_batch_size = 128
opts.lam = 2
opts.epoch_per_test = 2
opts.max_length = 7
opts.alpha = 0.7
opts.beta = 0.5
opts.data_path = args.data_path

data = args.data_path.split('/')
if len(data[-1]) > 0:
    data = data[-1]
else:
    data = data[-2]

if data == 'WN18RR':
    opts.learning_rate = 0.0036225411209161286
    opts.decay_rate = 0.99184314910166
    opts.batch_size = 512
    opts.L2 = 7.096653356534566e-05
elif data == 'FB15K237':
    opts.data_path = '../data/FB15K237/'
    opts.learning_rate = 0.002858371279116538
    opts.decay_rate = 0.9911310581870849
    opts.L2 = 0.0009525756060385333
    opts.batch_size = 1024

SEED = args.seed
opts.out_filename = 'results/%s_%d_%s_%s.txt'%(args.mode, SEED, data, args.out_file_info)

print(opts.data_path, SEED, data, args.mode)

reader = LPReader()
reader.read(data_path=opts.data_path, opts=opts)
pather = Sampler(reader, opts)

opts._ent_num = reader._ent_num
opts._rel_num = reader._rel_num
print(opts._ent_num, opts._rel_num)

sequence_datapath = '%spaths_%.1f_%.1f' % (opts.data_path, opts.alpha, opts.beta)

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

gpu = select_gpu()
torch.cuda.set_device(gpu)

def run_kge(structs, run_round):
    model = BaseModule(reader, opts, structs)
    valid_hit1 = model.train(train_data, valid_data, vfilter_mat, MAX_EPOCH=16)
    return - np.array(valid_hit1)

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "6"
    os.environ["MKL_NUM_THREADS"] = "6"
    warnings.filterwarnings("ignore", category=UserWarning)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # macro-level update
    C_out = [4, 13, 13]
    asng_out = CategoricalASNG(np.array(C_out), alpha=1.5, delta_init=1)
    run_round = 0
    best_hit1 = 0
    num_train = 0
    while(True):
        Ms = []
        structs = []
        for i in range(opts.lam):
            M = asng_out.sampling()
            struct = np.argmax(M, axis=1)
            Ms.append(M)
            structs.append(list(struct))

        scores = run_kge(structs, run_round)

        if np.min(scores) < best_hit1:
            best_hit1 = np.min(scores)
            print('best_hit@1 changed:', best_hit1)
        if args.mode == 'asng':
            asng_out.update(np.array(Ms), scores, True)
        num_train += opts.lam
        run_round += 1
        print("ROUND: %d. %d models trained. Best Hit@1: %.4f"%(run_round, num_train, -best_hit1))


