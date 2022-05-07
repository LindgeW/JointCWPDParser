import torch
import torch.nn as nn
import math


class NonlinearMLP(nn.Module):
    def __init__(self, in_feature, out_feature, activation=None, bias=True):
        super(NonlinearMLP, self).__init__()

        if activation is None:
            self.activation = lambda x: x
        else:
            assert callable(activation)
            self.activation = activation

        self.bias = bias
        self.linear = nn.Linear(in_features=in_feature,
                                out_features=out_feature,
                                bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        if self.bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        linear_out = self.linear(inputs)
        return self.activation(linear_out)


class Biaffine(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 bias=(True, True)  # True = 1  False = 0
                 ):
        super(Biaffine, self).__init__()
        self.in_features = in_features  # mlp_arc_size / mlp_label_size
        self.out_features = out_features  # 1 / rel_size
        self.bias = bias

        # arc / label: mlp_size + 1
        self.linear_input_size = in_features + bias[0]
        # arc: mlp_size
        # label: (mlp_size + 1) * rel_size
        self.linear_output_size = out_features * (in_features + bias[1])

        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.zeros_(self.linear.weight)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        if self.bias[0]:
            ones = input1.data.new_ones(batch_size, len1, 1)
            input1 = torch.cat((input1, ones), dim=-1)
            # dim1 += 1
        if self.bias[1]:
            ones = input2.data.new_ones(batch_size, len2, 1)
            input2 = torch.cat((input2, ones), dim=-1)
            # dim2 += 1

        # (bz, len1, dim1+1) -> (bz, len1, linear_output_size)
        affine = self.linear(input1)

        # (bz, len1 * self.out_features, dim2)
        affine = affine.reshape(batch_size, len1 * self.out_features, -1)

        # (bz, len1 * out_features, dim2) * (bz, dim2, len2)
        # -> (bz, len1 * out_features, len2) -> (bz, len2, len1 * out_features)
        biaffine = torch.bmm(affine, input2.transpose(1, 2)).transpose(1, 2)

        # (bz, len2, len1, out_features)    # out_features: 1 or rel_size
        biaffine = biaffine.reshape((batch_size, len2, len1, -1)).squeeze(-1)

        return biaffine


class BiaffineAttention(nn.Module):
    def __init__(self, in1_features, in2_features, num_label, bias=True):
        super(BiaffineAttention, self).__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, num_label, bias=bias)
        self.linear = nn.Linear(in1_features + in2_features, num_label, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, head, dep):
        """
        :param head: [batch, seq_len, hidden] 输入特征1, 即label-head
        :param dep: [batch, seq_len, hidden] 输入特征2, 即label-dep
        :return output: [batch, seq_len, num_cls] 每个元素对应类别的概率图
        """
        output = self.bilinear(head, dep)  # (bz, seq_len, num_cls)
        # biaff_score = output + self.linear(head)
        biaff_score = output + self.linear(torch.cat((head, dep), dim=-1))  # (bz, seq_len, num_cls)
        biaff_score = biaff_score.transpose(1, 2)  # (bz, num_cls, seq_len)
        att_weigths = self.softmax(biaff_score)
        att_out = torch.bmm(att_weigths, dep)  # (bz, num_cls, hidden)
        return att_out


# additive attention
class AdditiveAttention(nn.Module):
    def __init__(self, in_features, att_hidden, out_features, bias=True):
        super(AdditiveAttention, self).__init__()

        self.out_size = out_features
        self.linear1 = nn.Linear(in_features=in_features,
                                 out_features=att_hidden, bias=bias)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(in_features=att_hidden,
                                 out_features=out_features, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask=None):
        '''
        :param inputs: (bz, seq_len, in_features)
        :param mask: (bz, seq_len)  填充为0
        :return:
        '''
        # (bz, seq_len, att_hidden) -> (bz, seq_len, out_size)
        add_score = self.linear2(self.tanh(self.linear1(inputs)))
        # (bz, out_size, seq_len)
        add_score = add_score.transpose(1, 2)
        if mask is not None:
            pad_mask = (mask == 0)
            add_score = add_score.masked_fill(pad_mask[:, None, :], -1e9)
        att_weights = self.softmax(add_score)
        # (bz, out_size, seq_len) x (bz, seq_len, in_features)
        # -> (bz, out_size, in_features)
        att_out = torch.bmm(att_weights, inputs)
        return att_out


# multiplicative attention
class DotProductAttention(nn.Module):
    def __init__(self, k_dim, dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.scale = 1. / math.sqrt(k_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        :param q: (bz, q_len, q_dim)
        :param k: (bz, k_len, k_dim)
        :param v: (bz, v_len, v_dim)
        k_len == v_len  v_dim == q_dim
        :param mask: (bz, k_len)  填充部分为0
        :return: (bz, q_len, v_dim)
        '''
        # (bz, q_len, k_len)
        att_score = torch.bmm(q, k.transpose(1, 2)).mul(self.scale)

        if mask is not None:
            att_score.masked_fill_(~mask[:, None, :], -1e9)

        att_weights = self.softmax(att_score)

        if self.training:
            att_weights = self.dropout(att_weights)

        # (bz, q_len, v_dim)
        att_out = torch.bmm(att_weights, v)

        return att_out



