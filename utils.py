import logging
import os
import datetime
import random
import pandas as pd
import numpy as np

def logger_init(args):
    logging.basicConfig(level=logging.DEBUG, format='%(module)15s %(asctime)s %(message)s', datefmt='%H:%M:%S')
    if args.log_to_file:
        log_filename = os.path.join(args.log_dir, args.log_prefix+datetime.datetime.now().strftime("%m%d%H%M%S"))
        logging.getLogget().addHandler(logging.FileHandler(log_filename))


def plot_config(args):
    out_str = "\noptim:{} lr:{} lamb:{}, d:{}, n_sample:{}\n".format(
            args.optim, args.lr, args.lamb, args.n_dim, args.n_sample)
    with open(args.perf_file, 'a') as f:
        f.write(out_str) 


def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(random.randint(0, i))
    for ls in lists:
        j = idx[i]
        ls[i], ls[j] = ls[j], ls[i]

def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    for i in range(n_batch):
        start = int(n_sample * i / n_batch)
        end = int(n_sample * (i+1) / n_batch)
        ret = [ls[start:end] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    start = 0
    while(start < n_sample):
        end = min(n_sample, start + batch_size)
        ret = [ls[start:end] for ls in lists]
        start += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def cal_ranks(probs, method, label):
    if method == 'min':
        probs = probs.astype('float64')
        min_prob = np.expand_dims(probs[range(len(label)), label], 1)
        ranks = (probs > min_prob).sum(axis=1) + 1
        if ranks.mean() == 1:
            print(probs)
        probs_zero = ((probs==min_prob).sum(axis=1) > 1)
        if probs_zero.sum() >0:
            print('{}   {}    {}'.format((probs==min_prob).sum(axis=1)[probs_zero], probs_zero.sum(), label[probs_zero]))
    elif method == 'sort':
        sorted_idx = np.argsort(probs, axis=1)[:,::-1]
        find_target = sorted_idx == np.expand_dims(label, 1)
        ranks = np.nonzero(find_target)[1] + 1
    else:
        ranks = pd.DataFrame(probs).rank(axis=1, ascending=False, method=method)
        ranks = ranks.values[range(len(label)), label]
    return ranks

def cal_performance(ranks, top=10):
    mrr = (1. / ranks).sum() / len(ranks)
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_3 = sum(ranks<=3) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, m_r, h_1, h_3, h_10

def padding_EAdata(data, batch_size):
    padding_num = batch_size - len(data) % batch_size
    data = pd.concat([data, pd.DataFrame(np.zeros((padding_num, data.shape[1])), dtype=np.int32, columns=data.columns)], ignore_index=True, axis=0)
    return data, padding_num

def padding_LPdata(data, batch_size):
    padding_num = batch_size - len(data) % batch_size
    data = np.concatenate([data, np.zeros((padding_num, data.shape[1]), dtype=np.int32)])
    return data, padding_num

