import os
import argparse
import torch
import warnings
import pandas as pd
import numpy as np

from select_gpu import select_gpu
from asng import CategoricalASNG
from EA.read_data import Reader, Sampler
from EA.base_model import BaseModule

parser = argparse.ArgumentParser(description="Paser for KG entity alignment")
parser.add_argument('--data_path', type=str, default='data/dbp_wd_15k_V1/mapping/0_3/', help='the directory to dataset')
parser.add_argument('--seed', type=int, default=1234, help="random seed number")
parser.add_argument('--mode', type=str, default='asng', help="mode of sampling")
parser.add_argument('--out_file_info', type=str, default='hybrid', help='extra string for the output file name')

args = parser.parse_args()

class Options(object):
    pass

# init hyper-parameters
opts = Options()
opts.hidden_size = 256
opts.num_layers = 1
opts.test_batch_size = 1000
opts.lam = 2
opts.epoch_per_test = 1
opts.max_length = 15
opts.alpha = 0.9
opts.beta = 0.9
opts.data_path = args.data_path
opts.learning_rate = 0.0003
opts.decay_rate = 0.99663
opts.L2 = 0.0015
opts.batch_size = 1024
opts.drop = 0.3

SEED = args.seed
opts.out_filename = 'results/%s_%d_%s.txt'%(args.mode, SEED, args.out_file_info)

print(opts.data_path, SEED, args.mode)

reader =  Reader()
reader.read(data_path=opts.data_path)
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

data_size = len(reader._ent_testing)
valid_data = reader._ent_testing.iloc[:data_size//10]
test_data = reader._ent_testing.iloc[data_size//10:]

gpu = select_gpu()
torch.cuda.set_device(gpu)

def run_kge(structs, run_round):
    model = BaseModule(reader, opts, structs)
    tester_val = lambda x: model.test_align(reader, valid_data, x, kb_1to2=True)
    tester_tst = lambda x: model.test_align(reader, test_data, x, kb_1to2=True)
    valid_hit1 = model.train(train_data, tester_val, tester_tst, MAX_EPOCH=20)
    return - np.array(valid_hit1)

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "6"
    os.environ["MKL_NUM_THREADS"] = "6"
    warnings.filterwarnings("ignore")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # macro-level update
    C_out = [4,13,13]
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


