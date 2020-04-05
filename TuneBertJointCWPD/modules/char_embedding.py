import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
'''
1、根据词表建立字符表
2、构建词索引到字符索引buffer
3、建立词索引到词长度buffer
'''
class CharCNNEmbedding(nn.Module):
    def __init__(self, vocab, embed_size=50, char_embed_dim=50, include_word_start_end=True):
        super(CharCNNEmbedding, self).__init__()
        # 根据词表建立字符表
        char_vocab = vocab.build_char_vocab(include_word_start_end=include_word_start_end)

        # 字典中最大词长
        max_wd_len = max(map(lambda wd: len(wd[0]), vocab))
        if include_word_start_end:
            max_wd_len += 2
        # 词索引转字符索引
        self.char_pad_idx = char_vocab.padding_idx
        self.register_buffer('word_to_char_idxs', torch.zeros(len(vocab), max_wd_len, dtype=torch.long).fill_(self.char_pad_idx))
        self.register_buffer('word_lengths', torch.zeros(len(vocab), dtype=torch.long))
        for wd, idx in vocab:
            if include_word_start_end:
                chars = ['<c>'] + list(wd) + ['</c>']
            else:
                chars = list(wd)
            self.word_to_char_idxs[idx, :len(chars)] = torch.LongTensor(char_vocab.word2index(chars))
            self.word_lengths[idx] = len(chars)

        self.char_embedding = nn.Embedding(num_embeddings=len(char_vocab),
                                           embedding_dim=char_embed_dim,
                                           padding_idx=0)

        win_sizes = [1, 3, 5]
        filter_nums = [30, 40, 50]
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=char_embed_dim,
                      out_channels=filter_nums[i],
                      kernel_size=w),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
            # nn.AdaptiveAvgPool1d(output_size=1)
        ) for i, w in enumerate(win_sizes)])

        self.linear = nn.Linear(in_features=sum(filter_nums),
                                out_features=embed_size)

        self.dropout = nn.Dropout(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.char_embedding.weight)

    def forward(self, word_idxs):
        '''
        :param word_idxs: (bz, seq_len)
        :return: (bz, seq_len, embed_size)
        '''
        bz, seq_len = word_idxs.size()
        # max_seq_len = wd_mask.sum(dim=1).max().item()
        wd_lens = self.word_lengths[word_idxs]  # (bz, seq_len)
        max_wd_len = wd_lens.max()  # 实际最大长度
        chars = self.word_to_char_idxs[word_idxs]  # (bz, seq_len, max_wd_len)
        chars = chars[:, :, :max_wd_len]
        # (bz, seq_len, max_wd_len, char_embed_size)
        # -> (bz * seq_len, max_wd_len, char_embed_size)
        char_embed = self.char_embedding(chars).reshape(bz*seq_len, max_wd_len, -1)
        # (bz*seq_len, char_embed_size, max_wd_len)
        char_embed = char_embed.transpose(1, 2)
        # (bz*seq_len, conv_size)
        conv_outs = torch.cat(tuple(conv(char_embed).squeeze(-1) for conv in self.convs), dim=-1).contiguous()

        # (bz, seq_len, embed_size)
        embed_out = self.linear(conv_outs).reshape(bz, seq_len, -1)

        if self.training:
            embed_out = self.dropout(embed_out)

        return embed_out


class CharLSTMEmbedding(nn.Module):
    def __init__(self, vocab, embed_size=50, char_embed_dim=50, hidden_size=50, bidirectional=True, include_word_start_end=True):
        super(CharLSTMEmbedding, self).__init__()
        # 根据词表建立字符表
        char_vocab = vocab.build_char_vocab(include_word_start_end=include_word_start_end)

        # 字典中最大词长
        max_wd_len = max(map(lambda wd: len(wd[0]), vocab))
        if include_word_start_end:
            max_wd_len += 2
        # 词索引转字符索引
        self.char_pad_idx = char_vocab.padding_idx
        self.register_buffer('word_to_char_idxs', torch.zeros(len(vocab), max_wd_len, dtype=torch.long).fill_(self.char_pad_idx))
        self.register_buffer('word_lengths', torch.zeros(len(vocab), dtype=torch.long))
        for wd, idx in vocab:
            if include_word_start_end:
                chars = ['<c>'] + list(wd) + ['</c>']
            else:
                chars = list(wd)
            self.word_to_char_idxs[idx, :len(chars)] = torch.LongTensor(char_vocab.word2index(chars))
            self.word_lengths[idx] = len(chars)

        self.char_embedding = nn.Embedding(num_embeddings=len(char_vocab),
                                           embedding_dim=char_embed_dim,
                                           padding_idx=0)

        lstm_hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.lstm = LSTM(input_size=char_embed_dim,
                         hidden_size=lstm_hidden_size,
                         num_layers=1,
                         batch_first=True,
                         bidirectional=bidirectional)

        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=embed_size)

        self.dropout = nn.Dropout(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.char_embedding.weight)

    def forward(self, word_idxs):
        '''
        :param word_idxs: (bz, seq_len)
        :return: (bz, seq_len, embed_size)
        '''
        bz, seq_len = word_idxs.size()
        wd_lens = self.word_lengths[word_idxs]  # (bz, seq_len)
        max_wd_len = wd_lens.max()  # 实际最大长度
        chars = self.word_to_char_idxs[word_idxs]  # (bz, seq_len, max_wd_len)
        chars = chars[:, :, :max_wd_len]
        char_mask = chars.eq(self.char_pad_idx)  # pad部分mask为1
        # (bz, seq_len, max_wd_len, char_embed_size)
        # -> (bz * seq_len, max_wd_len, char_embed_size)
        char_embed = self.char_embedding(chars).reshape(bz*seq_len, max_wd_len, -1)

        char_lens = char_mask.eq(0).sum(dim=-1).reshape(bz*seq_len)  # (bz * seq_len, )
        lstm_out = self.lstm(char_embed, char_lens)[0]  # (bz*seq_len, max_wd_len, hidden_size)

        lstm_out = F.relu(lstm_out)

        # pad部分置成-inf，以免对max_pool造成干扰
        # 如果是avg_pool，pad部分置成0
        mask_lstm_out = lstm_out.masked_fill(char_mask.reshape(bz*seq_len, max_wd_len, 1), -1e9)
        mask_lstm_out = mask_lstm_out.transpose(1, 2)  # (bz*seq_len, hidden_size, max_wd_len)
        out = F.max_pool1d(mask_lstm_out, kernel_size=mask_lstm_out.size(-1)).squeeze(dim=-1)
        # (bz, seq_len, embed_size)
        embed_out = self.linear(out).reshape(bz, seq_len, -1)

        if self.training:
            embed_out = self.dropout(embed_out)

        return embed_out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.,
                 batch_first=True, bidirectional=False):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
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
            pack_enc_out, hx = self.lstm(pack_embed, hx)
            enc_out, _ = rnn.pad_packed_sequence(pack_enc_out, batch_first=True)
            _, unsort_idxs = torch.sort(sort_idxs, dim=0, descending=False)
            enc_out = enc_out[unsort_idxs]
        else:
            enc_out, hx = self.lstm(x, hx)

        return enc_out, hx
