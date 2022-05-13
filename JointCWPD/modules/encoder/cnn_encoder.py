import torch
import torch.nn as nn


class ResidualConv(nn.Module):
    def __init__(self, conv_layer, dropout=0.2):
        super(ResidualConv, self).__init__()
        assert isinstance(conv_layer, nn.Conv1d)
        self.conv_layer = conv_layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        return inputs + self.dropout(self.conv_layer(inputs))


class ConvEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_convs, kernel_size, dropout=0.2):
        super(ConvEncoder, self).__init__()

        assert in_features == out_features  # 如果输入输出维度不一致，则无法使用残差连接
        # 保证卷积前后序列长度不变：kernel_size = 2 * pad + 1
        assert kernel_size % 2 == 1
        padding = kernel_size // 2

        self.conv_layers = nn.Sequential()
        for i in range(num_convs):
            rconv = ResidualConv(nn.Conv1d(in_channels=in_features,
                                           out_channels=out_features,
                                           kernel_size=kernel_size,
                                           padding=padding), dropout)
            self.conv_layers.add_module(name=f'conv_{i}', module=rconv)

            self.conv_layers.add_module(name=f'activation_{i}', module=nn.ReLU())

    def forward(self, embed_inputs, non_pad_mask=None):
        '''
        :param embed_inputs: (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        if non_pad_mask is not None:
            # (bz, seq_len, embed_dim) * (bz, seq_len, 1)  广播
            embed_inputs *= non_pad_mask.unsqueeze(dim=-1)

        # (bz, 2*embed_dim, seq_len)
        embed = embed_inputs.transpose(1, 2)

        conv_out = self.conv_layers(embed)

        # (bz, seq_len, 2*embed_dim)
        enc_out = conv_out.transpose(1, 2)

        return enc_out


# 提取k个最大值并保持相对顺序不变
class KMaxPool1d(nn.Module):
    def __init__(self, top_k: int):
        super(KMaxPool1d, self).__init__()
        self.top_k = top_k

    def forward(self, inputs):
        assert inputs.dim() == 3
        # torch.topk和torch.sort均返回的是：values, indices
        top_idxs = torch.topk(inputs, k=self.top_k, dim=2)[1]
        sorted_top_idxs = top_idxs.sort(dim=2)[0]
        # gather: 沿给定轴dim, 将输入索引张量index指定位置的值进行聚合(抽取)
        return inputs.gather(dim=2, index=sorted_top_idxs)


class ConvStack(nn.Module):
    def __init__(self, in_features, out_features, num_convs=3, filter_size=100, kernel_sizes=(1, 3, 5), dropout=0.1):
        super(ConvStack, self).__init__()

        self.conv_stack = nn.Sequential()
        for i in range(num_convs):
            conv_i = ConvEncoder(in_features, out_features, filter_size, kernel_sizes, dropout)
            self.conv_stack.add_module(f'conv_{i}', conv_i)
            self.conv_stack.add_module(f'activate_{i}', nn.ReLU())

    def forward(self, inputs):
        return self.conv_stack(inputs)