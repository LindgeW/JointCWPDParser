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
