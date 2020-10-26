import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

"""
this file partially refers to https://github.com/nju-websoft/RSN
"""

class Reader(object):

    def read(self, data_path='data/dbp_wd_15k_V1/mapping/0_3/'):
        # read KGs
        def read_kb(path, names):
            return pd.read_csv(path, sep='\t', header=None, names=names)

        kb1 = read_kb(data_path+'triples_1', names=['h_id', 'r_id', 't_id'])
        kb2 = read_kb(data_path+'triples_2', names=['h_id', 'r_id', 't_id'])

        ent_mapping = read_kb(data_path+'sup_ent_ids', names=['kb_1', 'kb_2'])
        ent_testing = read_kb(data_path+'ref_ent_ids', names=['kb_1', 'kb_2'])

        if not os.path.exists(data_path+'sup_rel_ids'):
            os.mknod(data_path+'sup_rel_ids')
        if not os.path.exists(data_path+'rel_rel_ids'):
            os.mknod(data_path+'rel_rel_ids')

        rel_mapping = read_kb(data_path+'sup_rel_ids', names=['kb_1', 'kb_2'])
        rel_testing = read_kb(data_path+'rel_rel_ids', names=['kb_1', 'kb_2'])

        ent_id_1 = read_kb(data_path+'ent_ids_1', names=['id', 'e'])
        ent_id_2 = read_kb(data_path+'ent_ids_2', names=['id', 'e'])
        ent_id_2.loc[:, 'e'] += ':KB2'
        i2el_1 = pd.Series(ent_id_1.e.values, index=ent_id_1.id.values)
        i2el_2 = pd.Series(ent_id_2.e.values, index=ent_id_2.id.values)

        rel_id_1 = read_kb(data_path+'rel_ids_1', names=['id', 'r'])
        rel_id_2 = read_kb(data_path+'rel_ids_2', names=['id', 'r'])
        rel_id_2.loc[:, 'r'] += ':KB2'
        i2rl_1 = pd.Series(rel_id_1.r.values, index=rel_id_1.id.values)
        i2rl_2 = pd.Series(rel_id_2.r.values, index=rel_id_2.id.values)

        # convert id
        def id2label(df, i2el, i2rl, is_kb=True):
            if is_kb:
                df['h'] = i2el.loc[df.h_id.values].values
                df['r'] = i2rl.loc[df.r_id.values].values
                df['t'] = i2el.loc[df.t_id.values].values

                return df
            else:
                df['kb_1'] = i2el.loc[df.kb_1.values].values
                df['kb_2'] = i2rl.loc[df.kb_2.values].values

                return df

        id2label(kb1, i2el_1, i2rl_1)
        id2label(kb2, i2el_2, i2rl_2)
        id2label(ent_mapping, i2el_1, i2el_2, is_kb=False)
        id2label(rel_mapping, i2rl_1, i2rl_2, is_kb=False)
        id2label(ent_testing, i2el_1, i2el_2, is_kb=False)
        id2label(rel_testing, i2rl_1, i2rl_2, is_kb=False)

        # add reverse edges
        kb = pd.concat([kb1, kb2], ignore_index=True)
        kb = kb[['h', 'r', 't']]

        rev_r = kb.r + ':reverse'
        rev_kb = kb.rename(columns={'h': 't', 't': 'h'})
        rev_kb['r'] = rev_r.values
        kb = pd.concat([kb, rev_kb], ignore_index=True)


        rev_rmap = rel_mapping + ':reverse'
        rel_mapping = pd.concat([rel_mapping, rev_rmap], ignore_index=True)

        # resort id in descending order of frequency, since we use log-uniform sampler for NCE loss
        def remap_kb(kb):
            es = pd.concat([kb.h, kb.t], ignore_index=True)
            rs = kb.r
            e_num = es.groupby(es.values).size().sort_values()[::-1]
            r_num = rs.groupby(rs.values).size().sort_values()[::-1]

            e_map = pd.Series(range(e_num.shape[0]), index=e_num.index)
            r_map = pd.Series(range(r_num.shape[0]), index=r_num.index)

            return e_map, r_map

        def index(df, e_map, r_map, is_kb=True):
            if is_kb:
                df['h_id'] = e_map.loc[df.h.values].values
                df['r_id'] = r_map.loc[df.r.values].values
                df['t_id'] = e_map.loc[df.t.values].values
            else:
                df['kb_1'] = e_map.loc[df.kb_1.values].values
                df['kb_2'] = e_map.loc[df.kb_2.values].values

        e_map, r_map = remap_kb(kb)

        index(kb, e_map, r_map)
        index(ent_mapping, e_map, None, is_kb=False)
        index(ent_testing, e_map, None, is_kb=False)
        index(rel_mapping, r_map, None, is_kb=False)
        index(rel_testing, r_map, None, is_kb=False)

        index(kb1, e_map, r_map)
        index(kb2, e_map, r_map)
        eid_1 = pd.unique(pd.concat([kb1.h_id, kb1.t_id], ignore_index=True))
        eid_2 = pd.unique(pd.concat([kb2.h_id, kb2.t_id], ignore_index=True))
        
        
        # add shortcuts
        self._eid_1 = pd.Series(eid_1)
        self._eid_2 = pd.Series(eid_2)

        self._ent_num = len(e_map)
        self._rel_num = len(r_map)
        self._ent_id = e_map
        self._rel_id = r_map

        self._ent_mapping = ent_mapping
        self._rel_mapping = rel_mapping
        self._ent_testing = ent_testing
        self._rel_testing = rel_testing
        
        
        

        self._kb = kb
        # we first tag the entities that have algined entities according to entity_mapping
        self.add_align_infor()
        # we then connect two KGs by creating new triples involving aligned entities.
        self.add_weight()

    def add_align_infor(self):
        kb = self._kb
        
        ent_mapping = self._ent_mapping
        rev_e_m = ent_mapping.rename(columns={'kb_1': 'kb_2', 'kb_2': 'kb_1'})
        rel_mapping = self._rel_mapping
        rev_r_m = rel_mapping.rename(columns={'kb_1': 'kb_2', 'kb_2': 'kb_1'})
        
        ent_mapping = pd.concat([ent_mapping, rev_e_m], ignore_index=True)
        rel_mapping = pd.concat([rel_mapping, rev_r_m], ignore_index=True)
        
        ent_mapping = pd.Series(ent_mapping.kb_2.values, index=ent_mapping.kb_1.values)
        rel_mapping = pd.Series(rel_mapping.kb_2.values, index=rel_mapping.kb_1.values)
        
        self._e_m = ent_mapping
        self._r_m = rel_mapping
        
        kb['ah_id'] = kb.h_id
        kb['ar_id'] = kb.r_id
        kb['at_id'] = kb.t_id
        
        h_mask = kb.h_id.isin(ent_mapping)
        r_mask = kb.r_id.isin(rel_mapping)
        t_mask = kb.t_id.isin(ent_mapping)
        
        kb['ah_id'][h_mask] = ent_mapping.loc[kb['ah_id'][h_mask].values]
        kb['ar_id'][r_mask] = rel_mapping.loc[kb['ar_id'][r_mask].values]
        kb['at_id'][t_mask] = ent_mapping.loc[kb['at_id'][t_mask].values]
        
        self._kb = kb
        
    def add_weight(self):
        kb = self._kb[['h_id', 'r_id', 't_id', 'ah_id', 'ar_id', 'at_id']]

        kb['w_h'] = 0
        kb['w_r'] = 0
        kb['w_t'] = 0


        h_mask = ~(kb.h_id == kb.ah_id)
        r_mask = ~(kb.r_id == kb.ar_id)
        t_mask = ~(kb.t_id == kb.at_id)

        kb.loc[h_mask, 'w_h'] = 1
        kb.loc[r_mask, 'w_r'] = 1
        kb.loc[t_mask, 'w_t'] = 1

        akb = kb[['ah_id','ar_id','at_id', 'w_h', 'w_r', 'w_t']]
        akb = akb.rename(columns={'ah_id':'h_id','ar_id':'r_id','at_id':'t_id'})

        ahkb = kb[h_mask][['ah_id','r_id','t_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ah_id':'h_id'})
        arkb = kb[r_mask][['h_id','ar_id','t_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ar_id':'r_id'})
        atkb = kb[t_mask][['h_id','r_id','at_id', 'w_h', 'w_r', 'w_t']].rename(columns={'at_id':'t_id'})
        ahrkb = kb[h_mask&r_mask][['ah_id','ar_id','t_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ah_id':'h_id', 'ar_id':'r_id'})
        ahtkb = kb[h_mask&t_mask][['ah_id','r_id','at_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ah_id':'h_id', 'at_id':'t_id'})
        artkb = kb[r_mask&t_mask][['h_id','ar_id','at_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ar_id':'r_id', 'at_id':'t_id'})
        ahrtkb = kb[h_mask&r_mask&t_mask][['ah_id','ar_id','at_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ah_id':'h_id',
                                                                                                          'ar_id':'r_id',
                                                                                                          'at_id':'t_id'})

        kb['w_h'] = 0
        kb['w_r'] = 0
        kb['w_t'] = 0

        kb = pd.concat([akb, ahkb, arkb, atkb, ahrkb, ahtkb, artkb, ahrtkb, kb[['h_id','r_id','t_id', 'w_h', 'w_r', 'w_t']]],
                       ignore_index=True).drop_duplicates()

        self._kb = kb.reset_index(drop=True)


class Sampler(object):
    def __init__(self, reader, options):
        self._options = options
        self.reader = reader

    def sample_paths(self, repeat_times=2):
        opts = self._options

        kb = self.reader._kb.copy()

        kb = kb[['h_id', 'r_id', 't_id']]

        # sampling paths in the h_id-(r_id,t_id) form.

        rtlist = np.unique(kb[['r_id', 't_id']].values, axis=0)

        rtdf = pd.DataFrame(rtlist, columns=['r_id', 't_id'])
        
        # assign tail=(r_id, t_id), we assign an id for each tail
        rtdf = rtdf.reset_index().rename({'index': 'tail_id'}, axis='columns')
        
        # merge kb with rtdf, to get the (h_id, tail_id) dataframe
        rtkb = kb.merge(
            rtdf, left_on=['r_id', 't_id'], right_on=['r_id', 't_id'])

        htail = np.unique(rtkb[['h_id', 'tail_id']].values, axis=0)
        
        # save to the sparse matrix
        htailmat = csr_matrix((np.ones(len(htail)), (htail[:, 0], htail[:, 1])),
                              shape=(self.reader._ent_num, rtlist.shape[0]))

        # calulate corss-KG bias at first, note that we use an approximate method: 
        # if next entity e_{i+1} is in entity_mapping, e_i and e_{i+2} entity are believed in different KGs
        em = pd.concat(
            [self.reader._ent_mapping.kb_1,self.reader._ent_mapping.kb_2]).values

        rtkb['across'] = rtkb.t_id.isin(em)
        rtkb.loc[rtkb.across, 'across'] = opts.beta
        rtkb.loc[rtkb.across == 0, 'across'] = 1-opts.beta

        rtailkb = rtkb[['h_id', 't_id', 'tail_id', 'across']]
        
        def gen_tail_dict(x):
            return x.tail_id.values, x.across.values / x.across.sum()
        
        # each item in rtailkb is in the form of (tail_ids, cross-KG biases)
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
            
            # combine the biases
            p = x.tails[1].astype(np.float).reshape(
                [-1, ]) * pre_c.A.reshape([-1, ])
            p = p / p.sum()
            return np.random.choice(x.tails[0], 1, p=p)

        rt_x = rtailkb.loc[hrt[:, 2]].apply(perform_random, axis=1)
        rt_x = rtlist[np.concatenate(rt_x.values)]

        rts = [hrt, rt_x]
        c_length = 5
        print('current path length == %i' % c_length)
        while(c_length < opts.max_length):
            curr = rtailkb.loc[rt_x[:, 1]]
            
            # always using hrt[:, 0] as the previous entity is a stronger way to
            # generate deeper and cross-KG paths for the starting point. 
            # use 'curr.loc[:, 'pre'] = pre' for 2nd-order sampling.
            curr.loc[:, 'pre'] = hrt[:, 0]

            rt_x = curr.apply(perform_random2, axis=1)
            rt_x = rtlist[np.concatenate(rt_x.values)]

            rts.append(rt_x)
            c_length += 2
            # pre = curr.index.values
            print('current path length == %i' % c_length)
            
        data = np.concatenate(rts, axis=1)
        data = pd.DataFrame(data)
        
        self._train_data = data
        data.to_csv('%spaths_%.1f_%.1f' % (opts.data_path, opts.alpha, opts.beta))

