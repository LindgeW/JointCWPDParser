from .rnn_encoder import RNNEncoder
import torch
import torch.nn as nn
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


class BiLSTMEncoder(nn.Module):
    def __init__(self, args):
        super(BiLSTMEncoder, self).__init__()

        self.bilstm = nn.LSTM(input_size=3 * args.embed_size,
                              hidden_size=args.hidden_size // 2,
                              num_layers=args.lstm_depth,
                              dropout=args.lstm_drop,
                              batch_first=True,
                              bidirectional=True)

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
