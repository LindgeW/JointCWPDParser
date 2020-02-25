import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ArcBiaffine(nn.Module):
    def __init__(self, hidden_size, bias=True):
        """
        :param hidden_size: 输入的特征维度
        :param bias: 是否使用bias. Default: ``True``
        """
        super(ArcBiaffine, self).__init__()
        self.U = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.has_bias = bias
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U)
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def forward(self, dep, head):
        """
        :param head: arc-head tensor [batch, length, hidden]
        :param dep: arc-dependent tensor [batch, length, hidden]
        :return output: tensor [bacth, length, length]
        """
        output = dep.matmul(self.U)
        output = output.bmm(head.transpose(-1, -2))
        if self.has_bias:
            output = output + head.matmul(self.bias).unsqueeze(1)
        return output


class LabelBiaffine(nn.Module):
    def __init__(self, in1_features, in2_features, num_label, bias=True):
        """
        :param in1_features: 输入的特征1维度
        :param in2_features: 输入的特征2维度
        :param num_label: 边类别的个数
        :param bias: 是否使用bias. Default: ``True``
        """
        super(LabelBiaffine, self).__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, num_label, bias=bias)
        self.lin = nn.Linear(in1_features + in2_features, num_label, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bilinear.weight)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x1, x2):
        """
        :param x1: [batch, seq_len, hidden] 输入特征1, 即label-head
        :param x2: [batch, seq_len, hidden] 输入特征2, 即label-dep
        :return output: [batch, seq_len, num_cls] 每个元素对应类别的概率图
        """
        output = self.bilinear(x1, x2)
        output = output + self.lin(torch.cat(tuple([x1, x2]), dim=2).contiguous())
        return output


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
        '''
        :param input1: dep
        :param input2: head
        :return:
        '''
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        if self.bias[0]:
            ones = input1.data.new_ones(batch_size, len1, 1)
            input1 = torch.cat((input1, ones), dim=-1)
        if self.bias[1]:
            ones = input2.data.new_ones(batch_size, len2, 1)
            input2 = torch.cat((input2, ones), dim=-1)

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


class Biaffine2(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 bias=(True, True)  # True = 1  False = 0
                 ):
        super(Biaffine2, self).__init__()
        self.bias = bias

        self.W = nn.Parameter(torch.zeros(in_features + bias[0],
                                          in_features + bias[1],
                                          out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        # nn.init.zeros_(self.W)

    def forward(self, input1, input2):
        '''
        :param input1: dep
        :param input2: head
        :return:
        '''
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        if self.bias[0]:
            ones = input1.data.new_ones(batch_size, len1, 1)
            input1 = torch.cat((input1, ones), dim=-1)
        if self.bias[1]:
            ones = input2.data.new_ones(batch_size, len2, 1)
            input2 = torch.cat((input2, ones), dim=-1)

        biaffine = torch.einsum('bxi,ijo,byj->bxyo', input1, self.W, input2).squeeze(-1)

        return biaffine

# ================== Attention Layers =================== #


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
# class AdditiveAttention(nn.Module):
#     def __init__(self, in_features, att_hidden):
#         super(AdditiveAttention, self).__init__()
#
#         self.linear1 = nn.Linear(in_features=in_features,
#                                  out_features=att_hidden, bias=True)
#         self.tanh = nn.Tanh()
#         self.linear2 = nn.Linear(in_features=att_hidden,
#                                  out_features=1, bias=False)
#
#     def forward(self, inputs, mask=None):
#         '''
#         :param inputs: (bz, seq_len, in_features)
#         :param mask: (bz, seq_len)  填充为0
#         :return: att_out (bz, in_features)
#         '''
#         # (bz, seq_len, att_hidden) -> (bz, seq_len, 1)
#         add_score = self.tanh(self.linear1(inputs))
#         add_score = self.linear2(add_score).squeeze()  # (bz, seq)
#         if mask is not None:
#             add_score.masked_fill_(~mask, -1e9)
#         att_weights = F.softmax(add_score, dim=1)
#         # (bz, 1, seq_len) x (bz, seq_len, in_features)
#         # -> (bz, 1, in_features)
#         att_out = torch.bmm(att_weights.unsqueeze(dim=1), inputs).squeeze(dim=1)

#         #(bz, seq_len) x (bz, seq_len, in_features) -> (bz, seq_len, in_features)
#         att_out = (att_weights * inputs).sum(dim=1)
#         return att_out

class AdditiveAttention(nn.Module):
    def __init__(self, in_features, att_hidden=100, dropout=0.1):
        super(AdditiveAttention, self).__init__()

        self.linear1 = nn.Linear(in_features=in_features,
                                 out_features=att_hidden,
                                 bias=False)
        self.linear2 = nn.Linear(in_features=in_features,
                                 out_features=att_hidden)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(in_features=att_hidden,
                             out_features=1,
                             bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        :param q: (bz, len_q, dim_k)
        :param k: (bz, len_kv, dim_k)
        :param v: (bz, len_kv, dim_v)
        :param mask: (bz, len_kv)  填充为0
        :return: att_out (bz, len_q, dim_v)
        '''
        len_q, len_kv = q.size(1), k.size(1)
        _q = q.unsqueeze(2).repeat(1, 1, len_kv, 1)  # (bz, len_q, len_kv, dim_k)
        _k = q.unsqueeze(1).repeat(1, len_q, 1, 1)  # (bz, len_q, len_kv, dim_k)
        add_score = self.tanh(self.linear1(_q) + self.linear2(_k))
        att_score = self.out(add_score).squeeze(-1)  # (bz, len_q, len_kv)
        if mask is not None:
            att_score.masked_fill_(~mask[:, None, :], -1e9)

        # (bz, len_q, len_kv)
        att_weights = F.softmax(att_score, dim=-1)

        if self.training:
            att_weights = self.dropout(att_weights)

        att_out = torch.matmul(att_weights, v)

        return att_out  # (bz, len_q, dim_v)


# multiplicative attention
# 普通版
class DotProductAttention(nn.Module):
    def __init__(self, k_dim, dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.scale = 1. / math.sqrt(k_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        :param q: (bz, q_len, k_dim)
        :param k: (bz, kv_len, k_dim)
        :param v: (bz, kv_len, v_dim)
        :param mask: (bz, kv_len)  填充部分为0
        :return: (bz, q_len, v_dim)
        '''
        # (bz, q_len, k_len)
        att_score = torch.bmm(q, k.transpose(1, 2)).mul(self.scale)

        if mask is not None:
            att_score.masked_fill_(~mask[:, None, :], -1e9)

        # [bz, len_q, len_k]
        att_weights = self.softmax(att_score)

        if self.training:
            att_weights = self.dropout(att_weights)

        # (bz, q_len, v_dim)
        att_out = torch.bmm(att_weights, v)
        return att_out


# 添加门机制
class GateDotProductAttention(nn.Module):
    def __init__(self, k_dim, dropout=0.1):
        super(GateDotProductAttention, self).__init__()
        self.scale = 1. / math.sqrt(k_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.in_gates = nn.ModuleList([
            nn.Linear(in_features=k_dim, out_features=k_dim, bias=False),
            nn.Linear(in_features=k_dim, out_features=k_dim, bias=False)
        ])

        self.out_gates = nn.ModuleList([
            nn.Linear(in_features=k_dim, out_features=k_dim),
            nn.Linear(in_features=k_dim, out_features=k_dim)
        ])

        # self.fc = nn.Linear(in_features=2*k_dim, out_features=k_dim, bias=False)

    def _fusion_gate(self, in_h, out_h, idx=0):
        return (self.in_gates[idx](in_h) + self.out_gates[idx](out_h)).sigmoid()

    def forward(self, q, k, v, mask=None):
        '''
        :param q: (bz, q_len, k_dim)
        :param k: (bz, kv_len, k_dim)
        :param v: (bz, kv_len, v_dim)
        :param mask: (bz, kv_len)  填充部分为0
        :return: (bz, q_len, v_dim)
        '''
        # (bz, q_len, k_len)
        att_score = torch.bmm(q, k.transpose(1, 2)).mul(self.scale)

        # if mask is not None:
        #     att_score.masked_fill_(~mask[:, None, :], -1e9)

        x = att_score.new_ones(att_score.shape, requires_grad=False).byte()
        fw_att_score = att_score.masked_fill(x.tril(diagonal=-1), -1e9)  # 下三角矩阵
        bw_att_score = att_score.masked_fill(x.triu(diagonal=1), -1e9)  # 上三角矩阵

        # [bz, len_q, len_k]
        fw_att_weights = self.softmax(fw_att_score)
        bw_att_weights = self.softmax(bw_att_score)

        if self.training:
            fw_att_weights = self.dropout(fw_att_weights)
            bw_att_weights = self.dropout(bw_att_weights)

        # (bz, q_len, v_dim)
        fw_v = torch.bmm(fw_att_weights, v)
        bw_v = torch.bmm(bw_att_weights, v)
        fw_gate = self._fusion_gate(v, fw_v, 0)
        bw_gate = self._fusion_gate(v, bw_v, 1)
        fw_v = fw_gate * fw_v + (1 - fw_gate) * v
        bw_v = bw_gate * bw_v + (1 - bw_gate) * v
        att_out = torch.cat((fw_v, bw_v), dim=-1).contiguous()

        return att_out


# leverage local and global
class LocalDotProductAttention(nn.Module):
    def __init__(self, k_dim, win_size=12, dropout=0.1):
        super(LocalDotProductAttention, self).__init__()
        self.inf = -1e9
        self.scale = 1. / math.sqrt(k_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.win_size = win_size

        self.gate_fc = nn.Linear(in_features=k_dim, out_features=1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate_fc.weight, 0, 0.1)

    def _fusion_gate(self, x, y, q):
        gate = self.gate_fc(q).sigmoid()  # gate scalar
        return x * gate + y * (1 - gate)

    def forward(self, q, k, v, mask=None):
        '''
        :param q: (bz, q_len, k_dim)
        :param k: (bz, kv_len, k_dim)
        :param v: (bz, kv_len, v_dim)
        :param mask: (bz, kv_len)  填充部分为0
        :return: (bz, q_len, v_dim)
        '''
        # (bz, q_len, k_len)
        att_score = torch.matmul(q, k.transpose(1, 2)).mul(self.scale)

        if mask is not None:
            att_score.masked_fill_(~mask[:, None, :], self.inf)

        x = att_score.new_ones(att_score.shape, requires_grad=False).byte()
        brand_mask = x.triu(-self.win_size) * x.tril(self.win_size)
        local_att_score = att_score.masked_fill(~brand_mask, self.inf)

        # (bz, q_len, k_len)
        global_att_weights = self.softmax(att_score)
        local_att_weights = self.softmax(local_att_score)

        if self.training:
            global_att_weights = self.dropout(global_att_weights)
            local_att_weights = self.dropout(local_att_weights)

        # (bz, q_len, v_dim)
        global_att_out = torch.matmul(global_att_weights, v)
        local_att_out = torch.matmul(local_att_weights, v)

        fusion_out = self._fusion_gate(global_att_out, local_att_out, q)
        return fusion_out


if __name__ == '__main__':
    add_att = AdditiveAttention(10, 5)
    x = torch.rand(3, 10)
    y = torch.rand(3, 10)
    print(add_att(x, y).shape)



