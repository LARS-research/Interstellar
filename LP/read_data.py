import os, pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

"""
this file partially refers to https://github.com/nju-websoft/RSN
"""

class Reader(object):
    
    def read(self, data_path, opts=None):
        handled_path = data_path + 'basic_trainer_saved.pkl'
        self._options = opts

        if os.path.exists(handled_path):
            print('load file from local')
            (self._entity_num, self._relation_num, self._relation_num_for_eval, self._train_data, self._test_data,
             self._valid_data) = pickle.load(open(handled_path, 'rb'))
        else:
            self.read_data()
            self.merge_id()
            self.add_reverse()
            #self.reindex_kb()
            self.gen_t_label()

            print('start save dfs')
            saved = (
                self._entity_num, self._relation_num, self._relation_num_for_eval, self._train_data, self._test_data,
                self._valid_data)
            pickle.dump(saved, open(handled_path, 'wb'))

        self.gen_filter_mat()
        
        self._ent_num = self._entity_num
        self._rel_num = self._relation_num
        self._ent_mapping = pd.DataFrame({'kb_1':{}, 'kb_2':{}})
        self._rel_mapping = pd.DataFrame({'kb_1':{}, 'kb_2':{}})
        self._ent_testing = pd.DataFrame({'kb_1':{}, 'kb_2':{}})
        self._rel_testing = pd.DataFrame({'kb_1':{}, 'kb_2':{}})
        
        
        self._kb = self._train_data
        print('finished loading data')
        
        return 

    def merge_id(self):
        self._train_data['h_id'] = self._e_id[self._train_data.h].values
        self._train_data['r_id'] = self._r_id[self._train_data.r].values
        self._train_data['t_id'] = self._e_id[self._train_data.t].values

        self._test_data['h_id'] = self._e_id[self._test_data.h].values
        self._test_data['r_id'] = self._r_id[self._test_data.r].values
        self._test_data['t_id'] = self._e_id[self._test_data.t].values

        self._valid_data['h_id'] = self._e_id[self._valid_data.h].values
        self._valid_data['r_id'] = self._r_id[self._valid_data.r].values
        self._valid_data['t_id'] = self._e_id[self._valid_data.t].values
    
    def gen_t_label(self):
        full = pd.concat([self._train_data, self._test_data, self._valid_data], ignore_index=True)
        f_t_labels = full['t_id'].groupby([full['h_id'], full['r_id']]).apply(lambda x: pd.unique(x.values))
        f_t_labels.name = 't_label'

        self._test_data = self._test_data.join(f_t_labels, on=['h_id', 'r_id'])

        self._valid_data = self._valid_data.join(f_t_labels, on=['h_id', 'r_id'])


    def add_reverse(self):
        def add_reverse_for_data(data):
            reversed_data = data.rename(columns={'h_id': 't_id', 't_id': 'h_id'})
            reversed_data.r_id += self._relation_num
            data = pd.concat(([data, reversed_data]), ignore_index=True)
            return data

        self._train_data = add_reverse_for_data(self._train_data)
        self._test_data = add_reverse_for_data(self._test_data)
        self._valid_data = add_reverse_for_data(self._valid_data)
        self._relation_num_for_eval = self._relation_num
        self._relation_num *= 2

    def reindex_kb(self):
        train_data = self._train_data
        test_data = self._test_data
        valid_data = self._valid_data
        eids = pd.concat([train_data.h_id, train_data.t_id,], ignore_index=True)

        tv_eids = np.unique(pd.concat([test_data.h_id, test_data.t_id, valid_data.t_id, valid_data.h_id]))
        not_train_eids = tv_eids[~np.in1d(tv_eids, eids)]

        rids = pd.concat([train_data.r_id,],ignore_index=True)
        
        def gen_map(eids, rids):
            e_num = eids.groupby(eids.values).size().sort_values()[::-1]
            not_train = pd.Series(np.array(not_train_eids), index=not_train_eids)
            e_num = pd.concat([e_num, not_train])

            r_num = rids.groupby(rids.values).size().sort_values()[::-1]
            e_map = pd.Series(range(e_num.shape[0]), index=e_num.index)
            r_map = pd.Series(range(r_num.shape[0]), index=r_num.index)
            return e_map, r_map
        
        def remap_kb(kb, e_map, r_map):
            kb.loc[:, 'h_id'] = e_map.loc[kb.h_id.values].values
            kb.loc[:, 'r_id'] = r_map.loc[kb.r_id.values].values
            kb.loc[:, 't_id'] = e_map.loc[kb.t_id.values].values
            return kb
        
        def remap_id(s, rm):
            s = rm.loc[s.values].values
            return s
        
        e_map, r_map = gen_map(eids, rids)
        self._e_map, self._r_map = e_map, r_map
        
        self._train_data = remap_kb(train_data, e_map, r_map)
        self._valid_data = remap_kb(self._valid_data, e_map, r_map)
        self._test_data = remap_kb(self._test_data, e_map, r_map)
        
        self._e_id = remap_id(self._e_id, e_map)
        self._r_id = remap_id(self._r_id, r_map)
      
        return not_train_eids
    
    def in2d(self, arr1, arr2):
        """Generalisation of numpy.in1d to 2D arrays"""

        assert arr1.dtype == arr2.dtype

        arr1_view = np.ascontiguousarray(arr1).view(np.dtype((np.void,
                                                              arr1.dtype.itemsize * arr1.shape[1])))
        arr2_view = np.ascontiguousarray(arr2).view(np.dtype((np.void,
                                                              arr2.dtype.itemsize * arr2.shape[1])))
        intersected = np.in1d(arr1_view, arr2_view)
        return intersected.view(np.bool).reshape(-1)


    def gen_filter_mat(self):
        def gen_filter_vector(r):
            v = np.ones(self._entity_num)
            v[r] = -1
            return v

        print('start gen filter mat')



        self._tail_valid_filter_mat = np.stack(self._valid_data.t_label.apply(gen_filter_vector).values)
        self._tail_test_filter_mat = np.stack(self._test_data.t_label.apply(gen_filter_vector).values)


class LPReader(Reader):

    def read_data(self):
        path = self._options.data_path
        tr = pd.read_csv(path + 'train.txt', header=None, sep='\t', names=['h', 't', 'r'])
        te = pd.read_csv(path + 'test.txt', header=None, sep='\t', names=['h', 't', 'r'])
        val = pd.read_csv(path + 'valid.txt', header=None, sep='\t', names=['h', 't', 'r'])

        e_id = pd.read_csv(path + 'entity2id.txt', header=None, sep='\t', names=['e', 'eid'])
        e_id = pd.Series(e_id.eid.values, index=e_id.e.values)
        r_id = pd.read_csv(path + 'relation2id.txt', header=None, sep='\t', names=['r', 'rid'])
        r_id = pd.Series(r_id.rid.values, index=r_id.r.values)
        
        

        self._entity_num = e_id.shape[0]
        self._relation_num = r_id.shape[0]
        self._relation_num_for_eval = r_id.shape[0]


        self._train_data = tr
        self._test_data = te
        self._valid_data = val

        self._e_id, self._r_id = e_id, r_id


class Sampler(object):
    def __init__(self, reader, opts=None):
        self.reader = reader
        self._options = opts

    def sample_paths(self, repeat_times=2):
        opts = self._options

        kb = self.reader._kb.copy()

        kb = kb[['h_id', 'r_id', 't_id']]

        # sampling triples with the h_id-(r_id,t_id) form.

        rtlist = np.unique(kb[['r_id', 't_id']].values, axis=0)

        rtdf = pd.DataFrame(rtlist, columns=['r_id', 't_id'])

        rtdf = rtdf.reset_index().rename({'index': 'tail_id'}, axis='columns')

        rtkb = kb.merge(
            rtdf, left_on=['r_id', 't_id'], right_on=['r_id', 't_id'])

        htail = np.unique(rtkb[['h_id', 'tail_id']].values, axis=0)

        htailmat = csr_matrix((np.ones(len(htail)), (htail[:, 0], htail[:, 1])),
                              shape=(self.reader._ent_num, rtlist.shape[0]))

        # calulate corss-KG bias at first
        em = pd.concat(
            [self.reader._ent_mapping.kb_1, self.reader._ent_mapping.kb_2]).values

        rtkb['across'] = rtkb.t_id.isin(em)
        rtkb.loc[rtkb.across, 'across'] = opts.beta
        rtkb.loc[rtkb.across == 0, 'across'] = 1-opts.beta

        rtailkb = rtkb[['h_id', 't_id', 'tail_id', 'across']]

        def gen_tail_dict(x):
            return x.tail_id.values, x.across.values / x.across.sum()

        rtailkb = rtailkb.groupby('h_id').apply(gen_tail_dict)

        rtailkb = pd.DataFrame({'tails': rtailkb})

        # start sampling

        hrt = np.repeat(kb.values, repeat_times, axis=0)

        # for initial triples
        def perform_random(x):
            return np.random.choice(x.tails[0], 1, p=x.tails[1].astype(np.float))

        # else
        def perform_random2(x):

            # calculate depth bias
            pre_c = htailmat[np.repeat(x.pre, x.tails[0].shape[0]), x.tails[0]]
            pre_c[pre_c == 0] = opts.alpha
            pre_c[pre_c == 1] = 1-opts.alpha
            p = x.tails[1].astype(np.float).reshape(
                [-1, ]) * pre_c.A.reshape([-1, ])
            p = p / p.sum()
            return np.random.choice(x.tails[0], 1, p=p)

        rt_x = rtailkb.loc[hrt[:, 2]].apply(perform_random, axis=1)
        rt_x = rtlist[np.concatenate(rt_x.values)]

        rts = [hrt, rt_x]
        c_length = 5
        while(c_length < opts.max_length):
            curr = rtailkb.loc[rt_x[:, 1]]
            curr.loc[:, 'pre'] = hrt[:, 0]

            rt_x = curr.apply(perform_random2, axis=1)
            rt_x = rtlist[np.concatenate(rt_x.values)]

            rts.append(rt_x)
            c_length += 2

        data = np.concatenate(rts, axis=1)
        data = pd.DataFrame(data)
        
        self._train_data = data
        data.to_csv('%spaths_%.1f_%.1f' % (opts.data_path, opts.alpha, opts.beta))

