import os
import numpy as np
import torch
from torch.optim import Adam

import utils
from asng import CategoricalASNG
from torch.optim.lr_scheduler import ExponentialLR

from EA.model_search import SRAPModule

def write_row(filename, data, reset=False):
    with open(filename, 'w' if reset else 'a') as o:
        row = ''
        for i,c in enumerate(data):
            row += ('' if i==0 else ',') + str(c)
        row += '\n'
        o.write(row)

class BaseModule(object):
    def __init__(self, reader, options, structs):
        self.opts = options

        self.models = []
        self.structs = [list(struct) for struct in structs]

        C_in = [3,3, 2,2,2,2,2,2]
        self.asng_in = CategoricalASNG(np.array(C_in), alpha=1.5, delta_init=1)

    def train(self, train_data, tester_val, tester_tst, MAX_EPOCH=10):
        hit1s = []
        MAX_TIME = 2
        for l1 in range(self.opts.lam):
            # micro-level update
            model = SRAPModule(self.opts).cuda()
            optimizer = Adam(model.parameters(), lr=self.opts.learning_rate, weight_decay=self.opts.L2)
            scheduler = ExponentialLR(optimizer, self.opts.decay_rate)
            best_hit1 = 0
            early_stop = 0

            n_train = len(train_data)
            batch_size = self.opts.batch_size
            n_batch = n_train // batch_size + (n_train % batch_size > 0)
            for epoch in range(1):
                # shuffle training set
                choices = np.random.choice(n_train, size=n_train, replace=False)

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
                        loss += model._loss(seqs, self.structs[l1] + struct_in)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                   
                    # update architecture parameters
                    Ms = []
                    losses = []
                    index = np.random.choice(n_train, size=batch_size, replace=False)
                    batch_valid = train_data.iloc[choices[index]]
                    seqs = torch.LongTensor(batch_valid.values[:, :self.opts.max_length]).cuda()

                    with torch.no_grad():
                        for l2 in range(self.opts.lam):
                            M = self.asng_in.sampling()
                            struct_in = list(np.argmax(M, axis=1))
                            Ms.append(M)
                            loss = model._loss(seqs, self.structs[l1]+struct_in).cpu().data.numpy()
                            losses.append(loss)
                    try:
                        self.asng_in.update(np.array(Ms), losses, True)
                    except AssertionError:
                        break
            
                scheduler.step()
           
            # evaluate to get accurate feedback for macri-level update
            best_struct = list(self.asng_in.theta.argmax(axis=1))
            struct = self.structs[l1] + best_struct

            model = SRAPModule(self.opts).cuda()
            optimizer = Adam(model.parameters(), lr=self.opts.learning_rate, weight_decay=self.opts.L2)
            scheduler = ExponentialLR(optimizer, self.opts.decay_rate)

            best_hit1 = 0
            early_stop = 0
            for epoch in range(MAX_EPOCH):
                if (epoch+1) % self.opts.epoch_per_test == 0:
                    mrr, mr, h1, h3, h10 = tester_val(model)
                    if h1 > best_hit1:
                        best_hit1 = h1
                        best_str = str(struct) + '\t %d  %.4f %.1f %.4f %.4f %.4f\n' % (epoch+1, mrr, mr, h1, h3, h10)
                        early_stop = 0
                    else:
                        early_stop += 1
                    
                    if early_stop > MAX_TIME:
                        break

                choices = np.random.choice(n_train, size=n_train, replace=False)
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
                scheduler.step()
                
            with open(self.opts.out_filename, 'a') as f:
                f.write(best_str)
            hit1s.append(best_hit1)
        return hit1s

    def save(self, filename='SHARED_PARAMS.pt'):
        torch.save(self.model.state_dict(), self.opts.modelname)

    def load(self):
        if os.path.isfile(self.opts.modelname):
            self.model.load_state_dict(torch.load(self.opts.modelname, map_location=lambda storage, location: storage.cuda()))

    def test_align(self, reader, data, model, method='sort', kb_1to2=False):
        options = self.opts
        batch_size = options.test_batch_size
        label = data.kb_2 if kb_1to2==True else data.kb_1

        data, padding_num = utils.padding_EAdata(data, batch_size)
        s_em = model.sub_embed
        
        num_batch = len(data)//batch_size
        probs = []
        for i in range(num_batch):
            one_batch_data = data.iloc[i*batch_size: (i+1)*batch_size]
            if kb_1to2:
                e = torch.LongTensor(one_batch_data.kb_1.values).cuda()
            else:
                e = torch.LongTensor(one_batch_data.kb_2.values).cuda()
    
            s_embed = s_em(e)

            h_embed = s_embed
            norm_embed = s_em.weight
            h_embed = h_embed / torch.norm(h_embed, dim=-1, keepdim=True)
            norm_embed = norm_embed/ torch.norm(norm_embed, dim=-1, keepdim=True)
            values = torch.mm(h_embed, norm_embed.t())
            probs.append(values.data.cpu().numpy())


        probs = np.concatenate(probs)[:len(data)-padding_num]
        candi = reader._ent_testing.kb_2 if kb_1to2==True else reader._ent_testing.kb_1
        mask = np.in1d(np.arange(probs.shape[1]), candi)
        probs[:, ~mask] = probs.min() - 1
        ranks = utils.cal_ranks(probs, method=method, label=label)
        f_mrr, f_mr, f_h1, f_h3, f_h10 = utils.cal_performance(ranks)
        return f_mrr, f_mr, f_h1, f_h3, f_h10

       




