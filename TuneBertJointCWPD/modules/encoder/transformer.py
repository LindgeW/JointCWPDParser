import math
import numpy as np
import torch
import torch.nn as nn


# PE(pos, 2i) = sin(pos/10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
def PositionEncoding(max_len, d_model, pad_idx=None):
    pe = np.asarray([[pos / np.power(10000, 2.*(i//2) / d_model) for i in range(d_model)]
                     for pos in range(max_len)], dtype=np.float32)
    pe[:, 0::2] = np.sin(pe[:, 0::2])  # start : end : step
    pe[:, 1::2] = np.cos(pe[:, 1::2])

    if pad_idx is not None:
        pe[pad_idx] = 0

    return pe


class SelfAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1. / math.sqrt(d_k)
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
        # [bz, len_q, Q] x [bz, K, len_k] -> [bz, len_q, len_k]
        att_weights = torch.matmul(q, k.transpose(-1, -2)).mul(self.scale)

        if att_mask is not None:
            att_weights = att_weights.masked_fill(att_mask, -1e9)  # float('-inf')

        x = att_weights.new_full(att_weights.shape, -1e9, requires_grad=False)
        fw_mask = x.tril(diagonal=-1)  # 下三角矩阵
        bw_mask = x.triu(diagonal=1)  # 上三角矩阵
        fw_att_weights = att_weights + fw_mask
        bw_att_weights = att_weights + bw_mask

        # [bz, len_q, len_k]
        # soft_att_weights = self.softmax(att_weights)
        soft_fw_att_weights = self.softmax(fw_att_weights)
        soft_bw_att_weights = self.softmax(bw_att_weights)

        if self.training:
            # soft_att_weights = self._dropout(soft_att_weights)
            soft_fw_att_weights = self._dropout(soft_fw_att_weights)
            soft_bw_att_weights = self._dropout(soft_bw_att_weights)

        # [bz, len_q, len_k] * [bz, len_v, V] -> [bz, len_q, V]
        # att_out = torch.matmul(soft_att_weights, v)
        att_out = torch.matmul(soft_fw_att_weights, v) + torch.matmul(soft_bw_att_weights, v)
        # att_out = torch.cat((torch.matmul(soft_fw_att_weights, v) + torch.matmul(soft_bw_att_weights, v)), dim=-1)

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

        self._layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self._dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self._linear_qs.weight, mean=0, std=math.sqrt(2. / (self._d_model+self._d_k)))
        nn.init.normal_(self._linear_ks.weight, mean=0, std=math.sqrt(2. / (self._d_model+self._d_k)))
        nn.init.normal_(self._linear_vs.weight, mean=0, std=math.sqrt(2. / (self._d_model+self._d_v)))
        # nn.init.normal_(self._linear_out.weight, mean=0, std=math.sqrt(2. / (self._d_model + self._d_v)))
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
        residual = q

        bz, len_q, _ = q.size()
        bz, len_k, _ = k.size()
        bz, len_v, _ = v.size()
        nb_heads = self._nb_heads
        # [bz, len_q, d_k * nb_heads] -> [bz, nb_heads, len_q, d_k]
        q_fc = self._linear_qs(q).reshape(bz, len_q, nb_heads, -1).transpose(1, 2)
        # q_fc = q_fc.reshape(bz*nb_heads, len_q, -1)
        # [bz, len_k, d_k * nb_heads] -> [bz, nb_heads, len_k, d_k]
        k_fc = self._linear_ks(k).reshape(bz, len_k, nb_heads, -1).transpose(1, 2)
        # k_fc = k_fc.reshape(bz*nb_heads, len_k, -1)
        # [bz, len_v, d_v * nb_heads] -> [bz, nb_heads, len_v, d_v]
        v_fc = self._linear_vs(v).reshape(bz, len_v, nb_heads, -1).transpose(1, 2)
        # v_fc = v_fc.reshape(bz*nb_heads, len_v, -1)

        if att_mask is not None:
            # (bz, 1, len_k)
            att_mask = att_mask[:, None, None, :]
            # att_mask = att_mask[:, None, :].repeat(nb_heads, 1, 1)

        # (bz * nb_heads, len_q, d_v)
        att_out = self._self_attention(q_fc, k_fc, v_fc, att_mask).reshape(bz, nb_heads, len_q, -1)
        att_out = att_out.transpose(1, 2).reshape(bz, len_q, -1)
        # [bz, len_q, nb_heads*d_v] -> [bz, len_q, d_model]
        multi_head = self._linear_out(att_out)

        if self.training:
            multi_head = self._dropout(multi_head)

        return self._layer_norm(residual + multi_head)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels=d_in,
                      out_channels=d_ff,
                      kernel_size=1),  # 权重共享
            nn.ReLU(),
            nn.Conv1d(in_channels=d_ff,
                      out_channels=d_in,
                      kernel_size=1)
        )

        self._layer_norm = nn.LayerNorm(d_in, eps=1e-6)

        self._dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        '''
        :param inputs: [bz, len_q, d_model]
        :return: [bz, len_q, d_model]
        '''
        residual = inputs

        # [bz, len_q, d_model] -> [bz, d_model, len_q]
        fc_in = inputs.transpose(1, 2)

        # [bz, d_model, len_q]
        fc_out = self.ffn(fc_in)

        # [bz, len_q, d_model]
        out = fc_out.transpose(1, 2)

        if self.training:
            out = self._dropout(out)

        return self._layer_norm(residual + out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, nb_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # multi_head self-attention
        self._multi_head_att = MultiHeadAttention(d_model=d_model,
                                                  d_k=d_k,
                                                  d_v=d_v,
                                                  nb_heads=nb_heads,
                                                  dropout=dropout)
        # feedforward
        self._pwffn = PositionwiseFeedForward(d_in=d_model,
                                              d_ff=d_ff,
                                              dropout=dropout)

    def forward(self, enc_in, att_mask=None, seq_mask=None):
        '''
        :param enc_in: [bz, len_k, d_model]
        :param att_mask: [bz, len_k] attention mask, 将attention填充部分mask
        :param seq_mask: [bz, len_q, 1] 将序列填充部分mask
        :return: [bz, len_q, d_model]
        '''
        # [bz, len_q, d_model]
        multi_head = self._multi_head_att(enc_in, enc_in, enc_in, att_mask)
        if seq_mask is not None:
            multi_head *= seq_mask.float()

        # [bz, len_q, d_model]
        out = self._pwffn(multi_head)
        if seq_mask is not None:
            out *= seq_mask.float()

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, max_pos_embeddings, nb_layers, nb_heads, d_model, d_ff, dropout=0.1, att_drop=0.1):
        super(TransformerEncoder, self).__init__()

        d_k = d_v = d_model // nb_heads

        self.pos_embedding = nn.Embedding.from_pretrained(torch.from_numpy(PositionEncoding(max_pos_embeddings, d_model, pad_idx=0)), freeze=True)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self._encoder_stack = nn.ModuleList([
            EncoderLayer(d_model, d_k, d_v, d_ff, nb_heads, att_drop)
            for _ in range(nb_layers)
        ])

        self._dropout = nn.Dropout(dropout)

    def forward(self, embed_input, non_pad_mask=None):
        '''
        :param embed_input:  (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)  pad部分为0
        :return:
        '''
        if non_pad_mask is None:
            att_mask = None
            seq_mask = None
        else:
            att_mask = (non_pad_mask == 0)   # 填充部分的mask(uint8类型)
            seq_mask = non_pad_mask[:, :, None]  # (bz, seq_len, 1)

        seq_len = embed_input.size(1)
        seq_range = torch.arange(seq_len, dtype=torch.long, device=embed_input.device) \
            .unsqueeze(dim=0)  # (1, seq_len)
        # [bz, seq_len, d_model]
        embed_input = embed_input + self.pos_embedding(seq_range)

        encoder_out = self.layer_norm(embed_input)
        all_enc_outs = [encoder_out]
        if self.training:
            encoder_out = self._dropout(encoder_out)  # 0.1

        for encoder in self._encoder_stack:
            # [bz, len_q, d_model]
            encoder_out = encoder(encoder_out, att_mask=att_mask, seq_mask=seq_mask)
            all_enc_outs.append(encoder_out)

        return encoder_out, all_enc_outs


import matplotlib.pyplot as plt
def show_bar(atts):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(atts, cmap='bone')
    fig.colorbar(cax)
    # ax.set_xticklabels(['a', 'b', 'c', 'e'])
    # ax.set_yticklabels(['天', '津', '大', '学'])
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    plt.show()

import seaborn as sns
def show_bar2(atts):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    heatmap = sns.heatmap(atts, cmap='bone', cbar=True)
    # heatmap = sns.heatmap(atts, annot=True, fmt='f')
    # heatmap = sns.heatmap(atts, center=0)  # center=0 对于有正有负的数据而言颜色差异更大
    # heatmap.set_xticklabels(['a', 'b', 'c', 'd'])
    # heatmap.set_yticklabels(['天', '津', '大', '学'])
    plt.show()

if __name__ == '__main__':
    pe = PositionEncoding(50, 100)
    # print(pe[:2])
    # show_bar(pe)
    show_bar2(pe)