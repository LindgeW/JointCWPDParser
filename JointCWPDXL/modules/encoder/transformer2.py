import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# PE(pos, 2i) = sin(pos/10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
def PositionEncoding(max_len, d_model, pad_idx=None):
    pe = np.asarray([[pos / np.power(10000, 2*(i//2) / d_model) for i in range(d_model)]
                     for pos in range(max_len)], dtype=np.float32)
    pe[:, 0::2] = np.sin(pe[:, 0::2])  # start : end : step
    pe[:, 1::2] = np.cos(pe[:, 1::2])

    if pad_idx is not None:
        pe[pad_idx] = 0

    return pe


class SelfAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.scale = 1. / math.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self._dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, att_mask=None):
        '''
        :param q: [bz, len_q, Q]
        :param k: [bz, len_k, K]
        :param v: [bz, len_v, V]
        :param att_mask: [bz, len_q, len_k]  填充部分的mask
        more: Q==K, len_k==len_v
        :return: [bz, len_q, V]
        '''
        # [bz, len_q, Q] * [bz, K, len_k] -> [bz, len_q, len_k]
        att_weights = torch.matmul(q, k.transpose(-1, -2)).mul(self.scale)

        if att_mask is not None:
            att_weights.masked_fill_(att_mask, -1e9)  # float('-inf')

        # [bz, len_q, len_k]
        soft_att_weights = self.softmax(att_weights)

        if self.training:
            soft_att_weights = self._dropout(soft_att_weights)

        # [bz, len_q, len_k] * [bz, len_v, V] -> [bz, len_q, V]
        att_out = torch.matmul(soft_att_weights, v)

        return att_out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, nb_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self._d_model = d_model
        self._d_k = d_k
        self._d_v = d_v
        self._nb_heads = nb_heads

        self._linear_qs = nn.Linear(in_features=d_model, out_features=d_k * nb_heads)

        self._linear_ks = nn.Linear(in_features=d_model, out_features=d_k * nb_heads)

        self._linear_vs = nn.Linear(in_features=d_model, out_features=d_v * nb_heads)

        self._linear_out = nn.Linear(in_features=d_v * nb_heads, out_features=d_model)

        self._self_attention = SelfAttention(d_k, dropout)

        self._dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self._linear_qs.weight, mean=0, std=math.sqrt(1 / self._d_model))
        nn.init.normal_(self._linear_ks.weight, mean=0, std=math.sqrt(1 / self._d_model))
        nn.init.normal_(self._linear_vs.weight, mean=0, std=math.sqrt(1 / self._d_model))
        # nn.init.normal_(self._linear_out.weight, mean=0, std=math.sqrt(1 / self._d_model))
        nn.init.xavier_normal_(self._linear_out.weight)

    def forward(self, q, k, v, att_mask=None):
        '''
        :param q: [bz, len_q, d_model]
        :param k: [bz, len_k, d_model]
        :param v: [bz, len_v, d_model]
        :param att_mask: [bz, len_k]
        more: Q == K, len_k==len_v
        :return: [bz, len_q, d_model]
        '''
        bz, len_q, _ = q.size()
        bz, len_k, _ = k.size()
        bz, len_v, _ = v.size()
        # [bz, len_q, d_k * nb_heads] -> [bz, nb_heads, len_q, d_k]
        q_fc = self._linear_qs(q).reshape(bz, len_q, self._nb_heads, -1).transpose(1, 2)
        # [bz, len_k, d_k * nb_heads] -> [bz, nb_heads, len_k, d_k]
        k_fc = self._linear_ks(k).reshape(bz, len_k, self._nb_heads, -1).transpose(1, 2)
        # [bz, len_v, d_v * nb_heads] -> [bz, nb_heads, len_v, d_v]
        v_fc = self._linear_vs(v).reshape(bz, len_v, self._nb_heads, -1).transpose(1, 2)

        if att_mask is not None:
            # (bz, 1, 1, len_k)
            att_mask = att_mask[:, None, None, :]

        # (bz, nb_heads, len_q, d_v)
        att_out = self._self_attention(q_fc, k_fc, v_fc, att_mask)
        att_out = att_out.transpose(1, 2).reshape(bz, len_q, -1)
        # [bz, len_q, nb_heads*d_v] -> [bz, len_q, d_model]
        multi_head = self._linear_out(att_out)

        if self.training:
            multi_head = self._dropout(multi_head)

        return multi_head


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_in, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_in)
        )

        self._dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        '''
        :param inputs: [bz, len_q, d_model]
        :return: [bz, len_q, d_model]
        '''
        # [bz, len_q, d_model]
        fc_out = self.ffn(inputs)

        if self.training:
            fc_out = self._dropout(fc_out)

        return fc_out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, nb_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # multi_head self-attention
        self._multi_head_att = MultiHeadAttention(d_model=d_model,
                                                  d_k=d_k,
                                                  d_v=d_v,
                                                  nb_heads=nb_heads,
                                                  dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        # feedforward
        self._pwffn = PositionwiseFeedForward(d_in=d_model,
                                              d_ff=d_ff,
                                              dropout=dropout)

        self.norm_2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, enc_in, att_mask=None, seq_mask=None):
        '''
        :param enc_in: [bz, len_k, d_model]
        :param att_mask: [bz, len_k] attention mask, 将attention填充部分mask
        :param seq_mask: [bz, len_q, 1] 将序列填充部分mask
        :return: [bz, len_q, d_model]
        '''
        enc_in = self.norm_1(enc_in)
        att_out = self._multi_head_att(enc_in, enc_in, enc_in, att_mask)
        enc_in = enc_in + att_out

        if seq_mask is not None:
            enc_in *= seq_mask.float()

        enc_in = self.norm_2(enc_in)
        fc_out = self._pwffn(enc_in)
        enc_out = enc_in + fc_out

        if seq_mask is not None:
            enc_out *= seq_mask.float()

        return enc_out


class TransformerEncoder(nn.Module):
    def __init__(self, max_pos_embeddings, nb_layers, nb_heads, d_model, d_ff, att_drop=0.1):
        super(TransformerEncoder, self).__init__()

        d_k = d_v = d_model // nb_heads

        self.pos_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(PositionEncoding(max_pos_embeddings, d_model, pad_idx=0)), freeze=True)

        self._encoder_stack = nn.ModuleList([
            EncoderLayer(d_model, d_k, d_v, d_ff, nb_heads)
            for _ in range(nb_layers)
        ])

        self._dropout = nn.Dropout(att_drop)
        self.norm_out = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, embed_input, non_pad_mask):
        '''
        :param embed_input:  (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        if non_pad_mask is None:
            att_mask = None
            seq_mask = None
        else:
            att_mask = (non_pad_mask == 0)  # 填充部分的mask(uint8类型)
            seq_mask = non_pad_mask[:, :, None]  # (bz, seq_len, 1)

        seq_len = embed_input.size(1)
        seq_range = torch.arange(seq_len, dtype=torch.long, device=embed_input.device) \
            .unsqueeze(dim=0)  # (1, seq_len)
        # [bz, seq_len, d_model]
        encoder_out = embed_input + self.pos_embedding(seq_range)

        if self.training:
            encoder_out = self._dropout(encoder_out)  # 0.1

        for encoder in self._encoder_stack:
            # [bz, len_q, d_model]
            encoder_out = encoder(encoder_out, att_mask=att_mask, seq_mask=seq_mask)

        encoder_out = self.norm_out(encoder_out)
        return encoder_out