import torch
import torch.nn as nn
import torch.nn.functional as F


class Interstellar(nn.Module):
    """
    this class partially refers to https://github.com/nju-websoft/RSN
    """
    def __init__(self, options):
        super(Interstellar, self).__init__()
        self._options = options
        hidden_size = self.hidden_size = options.hidden_size

        self.sub_embed = nn.Embedding(options._ent_num, hidden_size)
        self.rel_embed = nn.Embedding(options._rel_num, hidden_size)
        self.obj_embed = nn.Embedding(options._ent_num, hidden_size)

        self.drop = nn.Dropout(options.drop)
        self.gate = nn.Linear(2*hidden_size, hidden_size)

        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W4 = nn.Linear(hidden_size, hidden_size)
        self.W5 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W6 = nn.Linear(hidden_size, hidden_size)

        self.idd = lambda x:x

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            if param.dim()>1:
                nn.init.xavier_uniform_(param.data)

    def _loss(self, seqs, struct):
        res_outputs, res_outputs_inv, obj_embed, obj_embed_inv = self.get_res_outputs(seqs, struct)
        positive = self.get_pair(res_outputs, obj_embed) + self.get_pair(res_outputs_inv, obj_embed_inv)
        negative = self.get_tail(res_outputs) + self.get_head(res_outputs_inv)
        max_neg = torch.max(negative, 1, keepdim=True)[0]

        loss = - positive + max_neg + torch.log(torch.sum(torch.exp(negative - max_neg), 1))
        return torch.sum(loss)

    def evaluate(self, seqs, struct):
        res_outputs, res_outputs_inv = self.get_res_outputs(seqs, struct)[:2]
        logits = self.get_tail(res_outputs) + self.get_head(res_outputs_inv)
        logits = logits - torch.min(logits, dim=1, keepdim=True)[0]
        return logits

    def get_pair(self, x1, x2):
        return torch.sum(x1*x2, dim=1)

    def get_head(self, x):
        return torch.mm(x, self.sub_embed.weight.transpose(1,0))

    def get_tail(self, x):
        return torch.mm(x, self.obj_embed.weight.transpose(1,0))

    def get_res_outputs(self, seqs, struct):
        length = seqs.size(1)
        sub = seqs[:, :-1:2]
        rel = seqs[:, 1::2]
        obj = seqs[:, 2::2]
        half_num = self._options._rel_num // 2
        rel_inv = torch.remainder(rel+half_num, self._options._rel_num)

        sub_emb_f = self.sub_embed(sub)
        rel_emb_f = self.rel_embed(rel)
        obj_emb_f = self.obj_embed(obj)

        sub_emb_b = self.obj_embed(sub)
        rel_emb_b = self.rel_embed(rel_inv)
        obj_emb_b = self.sub_embed(obj)

        forw_seq = []
        back_seq = []
        for i in range(length//2):
            f_seq = []
            f_seq.append(sub_emb_f[:,i,:])
            f_seq.append(rel_emb_f[:,i,:])
            forw_seq.append(f_seq)

            b_seq = []
            b_seq.append(sub_emb_b[:,i,:])
            b_seq.append(rel_emb_b[:,i,:])
            back_seq.append(b_seq)


        # h0
        h_f = sub_emb_f[:,0,:]
        h_b = sub_emb_b[:,0,:]

        outputs_f = []
        outputs_b = []
        for i in range(length//2):
            out_f, h_f = self.get_connect(forw_seq[i], h_f, struct)
            out_b, h_b = self.get_connect(back_seq[i], h_b, struct)
            outputs_f.append(out_f)
            outputs_b.append(out_b)

        outputs_f = self.drop(torch.cat(outputs_f, dim=0))
        outputs_b = self.drop(torch.cat(outputs_b, dim=0))

        target_f = obj_emb_f.permute(1,0,2).contiguous().view(-1, self.hidden_size)
        target_b = obj_emb_b.permute(1,0,2).contiguous().view(-1, self.hidden_size)
        return outputs_f, outputs_b, target_f, target_b
   

    def get_connect(self, ent_rel, h_tm1, struct):
        st = ent_rel[0]
        rt = ent_rel[1]
        zeros = torch.zeros(st.size()).cuda()
        ops = ['add', 'mult', 'complx', 'gate']

        W1 = [self.idd, self.W1][struct[5]]
        W2 = [self.idd, self.W2][struct[6]]

        if struct[0] == 0:
            op1 = self.ops(st, h_tm1, 'add', W1, W2)
        elif struct[0] == 1:
            op1 = self.ops(st, h_tm1, 'mult', W1, W2)
        elif struct[0] == 2:
            op1 = self.ops(st, h_tm1, 'complx', W1, W2)
        elif struct[0] == 3:
            op1 = self.ops(st, h_tm1, 'gate', W1, W2)

        if struct[3] == 1:
            op1 = torch.tanh(op1)
        elif struct[3] == 2:
            op1 = torch.sigmoid(op1)

        W3 = [self.idd, self.W3][struct[7]]
        W4 = [self.idd, self.W4][struct[8]]

        op2_in = [st, h_tm1, op1, zeros][struct[1]//4]
        op2 = self.ops(op2_in, rt, ops[struct[1]%4], W3, W4)

        if struct[4] == 1:
            op2 = torch.tanh(op2)
        elif struct[4] == 2:
            op2 = torch.sigmoid(op2)

        W5 = [self.idd, self.W5][struct[9]]
        W6 = [self.idd, self.W6][struct[10]]

        op3_in = [st, h_tm1, op1, zeros][struct[2]//4]
        op3 = self.ops(op3_in, op2, ops[struct[2]%4], W5, W6)

        ht = op2
        return op3.view(-1, self.hidden_size), ht.view(-1, self.hidden_size)


    def ops(self, x, h, name, W1, W2):
        x = W1(x)
        h = W2(h)
        if name == 'add':
            out = x+h
        elif name == 'mult':
            out = x*h
        elif name == 'complx':
            x1, x2 = torch.chunk(x, 2, dim=1)
            h1, h2 = torch.chunk(h, 2, dim=1)
            o1 = x1*h1 - x2*h2
            o2 = x1*h2 + x2*h1
            out = torch.cat([o1, o2], dim=1)
        elif name == 'gate':
            gate = torch.sigmoid(self.gate(torch.cat([x, h], dim=1)))
            out = gate * h + (1-gate)*x
        return out.squeeze()


