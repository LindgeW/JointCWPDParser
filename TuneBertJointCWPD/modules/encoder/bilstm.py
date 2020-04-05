from .rnn_encoder import RNNEncoder
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# class BiLSTMEncoder(nn.Module):
#     def __init__(self, args):
#         super(BiLSTMEncoder, self).__init__()
#
#         self.bilstm = RNNEncoder(input_size=args.wd_embed_dim + args.tag_embed_dim,
#                                  hidden_size=args.hidden_size // 2,
#                                  num_layers=args.lstm_depth,
#                                  dropout=args.lstm_drop,
#                                  batch_first=True,
#                                  bidirectional=True,
#                                  rnn_type='lstm')
#
#     def forward(self, embed_inputs, non_pad_mask=None):
#         '''
#         :param embed_inputs: (bz, seq_len, embed_dim)
#         :param non_pad_mask: (bz, seq_len)
#         :return:
#         '''
#         # lstm_out: (bz, seq_len, lstm_size * num_directions)
#         # lstm_hidden: (num_layers, batch, lstm_size * num_directions)
#         lstm_out, lstm_hidden = self.bilstm(embed_inputs, mask=non_pad_mask)
#
#         return lstm_out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.,
                 batch_first=True, bidirectional=False):
        super(GRU, self).__init__()

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=batch_first,
                          bidirectional=bidirectional)

    def forward(self, x, seq_lens=None, hx=None):
        '''
        :param x: (bz, seq_len, embed_dim)
        :param seq_lens: (bz, seq_len)
        :param hx: 初始隐层
        :return:
        '''

        if seq_lens is not None and not isinstance(x, rnn.PackedSequence):
            sort_lens, sort_idxs = torch.sort(seq_lens, dim=0, descending=True)
            pack_embed = rnn.pack_padded_sequence(x[sort_idxs], lengths=sort_lens, batch_first=True)
            pack_enc_out, hx = self.gru(pack_embed, hx)
            enc_out, _ = rnn.pad_packed_sequence(pack_enc_out, batch_first=True)
            _, unsort_idxs = torch.sort(sort_idxs, dim=0, descending=False)
            enc_out = enc_out[unsort_idxs]
        else:
            enc_out, hx = self.gru(x, hx)

        return enc_out, hx


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.2,
                 bidirectional=True,
                 batch_first=True):
        super(BiLSTMEncoder, self).__init__()

        self.bilstm = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=batch_first)

    def forward(self, embed_inputs, non_pad_mask=None):
        '''
        :param embed_inputs: (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        if non_pad_mask is None:
            non_pad_mask = embed_inputs.data.new_full(embed_inputs.shape[:2], 1)

        seq_lens = non_pad_mask.data.sum(dim=1)
        sort_lens, sort_idxs = torch.sort(seq_lens, dim=0, descending=True)
        pack_embed = pack_padded_sequence(embed_inputs[sort_idxs], lengths=sort_lens, batch_first=True)
        pack_enc_out, _ = self.bilstm(pack_embed)
        enc_out, _ = pad_packed_sequence(pack_enc_out, batch_first=True)
        _, unsort_idxs = torch.sort(sort_idxs, dim=0, descending=False)

        return enc_out[unsort_idxs]
