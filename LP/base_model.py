import os
import numpy as np
import torch
from torch.optim import Adam

import utils
from asng import CategoricalASNG
from torch.optim.lr_scheduler import ExponentialLR

from LP.model_search import Interstellar

class BaseModule(object):
    def __init__(self, reader, options, structs):
        self.opts = options

        self.models = []
        self.structs = structs

        C_in = [3,3, 2,2,2,2,2,2]
        self.asng_in = CategoricalASNG(np.array(C_in), alpha=1.5, delta_init=1)

    def train(self, train_data, valid_data, vfilter_mat, MAX_EPOCH=10):
        hit1s = []
        structs = []
        MAX_TIME = 3
        for l1 in range(self.opts.lam):
            # micro-level update
            model = Interstellar(self.opts).cuda()
            optimizer = Adam(model.parameters(), lr=self.opts.learning_rate, weight_decay=self.opts.L2)
            scheduler = ExponentialLR(optimizer, self.opts.decay_rate)
            best_hit1 = 0
            early_stop = 0

            n_train = len(train_data)
            n_valid = len(valid_data)
            batch_size = self.opts.batch_size
            n_batch = n_train // batch_size + (n_train%batch_size>0)
            for epoch in range(6):
                # shuffle training set
                choices = np.random.choice(n_train, size=n_train, replace=False)
                model.train()

                for i in range(n_batch):
                    start = i*batch_size
                    end = min(n_train, (i+1)*batch_size)
                    one_batch_choices = choices[start:end]
                    one_batch_data = train_data.iloc[one_batch_choices]
                    
                    # update model parameters
                    seqs = torch.LongTensor(one_batch_data.values[:, :self.opts.max_length]).cuda()
                    loss = 0
                    Ms = []
                    for l2 in range(self.opts.lam):
                        model.zero_grad()
                        M = self.asng_in.sampling()
                        struct_in = list(np.argmax(M, axis=1))
                        Ms.append(M)
                        loss += model._loss(seqs, self.structs[l1]+struct_in)
                    loss.backward()
                    optimizer.step()
                    self.projector(model)

                    # update architecture parameters after the warmup procedure
                    if epoch < 3:
                        continue
                    Ms = []
                    scores = []
                    index = np.random.choice(n_valid, size=self.opts.test_batch_size, replace=False)
                    batch_valid = valid_data[index]

                    with torch.no_grad():
                        for l2 in range(self.opts.lam):
                            M = self.asng_in.sampling()
                            struct_in = list(np.argmax(M, axis=1))
                            Ms.append(M)
                            hit1 =  model._loss(torch.LongTensor(batch_valid).cuda(), self.structs[l1]+struct_in).cpu().data.numpy()
                            scores.append(hit1)
                    try:
                        self.asng_in.update(np.array(Ms), np.array(scores), True)
                    except AssertionError:
                        break
                        
                scheduler.step()

            # evaluate to get accurate feedback for macri-level update
            best_struct = list(self.asng_in.theta.argmax(axis=1))
            struct = self.structs[l1] + best_struct

            model = Interstellar(self.opts).cuda()
            optimizer = Adam(model.parameters(), lr=self.opts.learning_rate, weight_decay=self.opts.L2)
            scheduler = ExponentialLR(optimizer, self.opts.decay_rate)


            best_hit1= 0
            early_stop = 0
            for epoch in range(MAX_EPOCH):
                if (epoch+1) % self.opts.epoch_per_test  == 0:
                    mrr, h1, h10 = self.test_link(valid_data, vfilter_mat, model, struct)
                    if h1 > best_hit1:
                        best_hit1= h1
                        best_str = str(struct) + '\tepoch:%d  mrr:%.4f  h1:%.4f  h10:%.4f\n' % (epoch+1, mrr, h1, h10)
                        early_stop = 0
                    else:
                        early_stop += 1

                    if early_stop > MAX_TIME:
                        break

                model.train()
                choices = np.random.choice(n_train, size=n_train, replace=True)
                for i in range(n_batch):
                    start = i*batch_size
                    end = min(n_train, (i+1)*batch_size)
                    one_batch_choices = choices[start:end]
                    one_batch_data = train_data.iloc[one_batch_choices]
                    seqs = torch.LongTensor(one_batch_data.values[:, :self.opts.max_length]).cuda()
                    model.zero_grad()
                    loss = model._loss(seqs, struct) 
                    loss.backward()
                    optimizer.step()
                    self.projector(model)
                scheduler.step()

            with open(self.opts.out_filename, 'a') as f:
                f.write(best_str)
            hit1s.append(best_hit1)
            structs.append(struct)
        return hit1s, structs

    def save(self, filename='SHARED_PARAMS.pt'):
        torch.save(self.model.state_dict(), self.opts.modelname)

    def load(self):
        if os.path.isfile(self.opts.modelname):
            self.model.load_state_dict(torch.load(self.opts.modelname, map_location=lambda storage, location: storage.cuda()))

    def projector(self, model):
        for n, p in model.named_parameters():
            if 'sub' in n or 'obj' in n:
                X = p.data.clone()
                Z = torch.norm(X, p=2, dim=1, keepdim=True)
                Z[Z<1] = 1
                X = X/Z
                p.data.copy_(X.view(-1, self.opts.hidden_size))


    def test_link(self, data, filter_mat, model, struct, method='sort'):
        options = self.opts
        batch_size = options.test_batch_size
        label = data[:,2]
        data, padding_num = utils.padding_LPdata(data, batch_size)
        num_batch = len(data) // batch_size

        probs = []
        model.eval()
        for i in range(num_batch):
            seqs = torch.LongTensor(data[i*batch_size:(i+1)*batch_size]).cuda()
            probs.append(model.evaluate(seqs, struct).data.cpu().numpy())
        if(np.isnan(probs).sum() > 0):
            self.flag = True
            print('Encountered NAN : {}'.format(self.opts.struct))
        probs = np.concatenate(probs)[:len(data) - padding_num]
        filter_probs = probs * filter_mat
        filter_probs[range(len(label)), label] = probs[range(len(label)), label]
        filter_ranks = utils.cal_ranks(filter_probs, method=method, label=label)
        f_mrr, f_h1, f_h10= utils.cal_performance(filter_ranks)
        return f_mrr, f_h1, f_h10

