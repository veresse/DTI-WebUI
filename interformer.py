
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))
    def forward(self, query, key, value, mask=None):
        if len(query.shape)>len(key.shape):
            bsz = query.shape[0]
        else:
            bsz = key.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        del Q,K
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        return self.fc(torch.matmul(self.do(F.softmax(energy, dim=-1)), V).permute(0, 2, 1, 3).contiguous().view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads)))

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc_2(self.do(F.relu(self.fc_1(x.permute(0, 2, 1))))).permute(0, 2, 1)

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg= trg.float()
        trg1 = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg1 = self.ln(trg1 + self.do(self.ea(trg1, src, src, src_mask)))
        trg1 = self.ln(trg1 + self.do(self.pf(trg1)))
        src1 = self.ln(src + self.do(self.sa(src, src, src, src_mask)))
        src1 = self.ln(src1 + self.do(self.ea(src1, trg, trg, trg_mask)))
        src1 = self.ln(src1 + self.do(self.pf(src1)))
        # trg,src= trg.cpu(),src.cpu()
        del trg,src, trg_mask, src_mask
        return trg1,src1

class Decoder(nn.Module):
    def __init__(self, hid_dim, n_layers, n_heads, pf_dim,  dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
             for _ in range(n_layers)])


    def forward(self, trg, src, trg_mask=None,src_mask=None):
        src = src.to(torch.float32)

        for layer in self.layers:
            trg,src = layer(trg, src,trg_mask,src_mask)
        del trg_mask,src_mask
        return trg,src


