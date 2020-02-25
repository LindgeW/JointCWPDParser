from .layer import *
from .encoder.transformer import TransformerEncoder
from .encoder.transformer_xl_nomem import TransformerXL
from .dropout import *
import torch
from .scale_mix import ScalarMix
import torch.nn.functional as F
from .crf import CRF
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import rnn


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


class ConvEncoder(nn.Module):
    def __init__(self, in_features, out_features, filter_size=100, kernel_sizes=(1, 3, 5), dropout=0.1):
        super(ConvEncoder, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_features,
                      out_channels=filter_size,
                      kernel_size=w,
                      padding=w//2) for w in kernel_sizes])

        self.k_max_pool = nn.AdaptiveMaxPool1d(output_size=out_features)
        # self.k_max_pool = KMaxPool1d(out_features)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        '''
        :param inputs: (bz, seq_len, input_size)
        :return:
        '''

        conv_in = inputs.transpose(1, 2)
        conv_out = torch.cat(tuple([conv(conv_in) for conv in self.convs]), dim=1).contiguous()

        out = self.k_max_pool(conv_out.transpose(1, 2))

        if self.training:
            out = self.dropout(out)
        return out


class ParserModel(nn.Module):
    def __init__(self, args, bert_embedding=None):
        super(ParserModel, self).__init__()

        self.args = args

        self.bert_embedding = bert_embedding

        # self.conv_encoder = ConvEncoder(in_features=d_model, out_features=d_model)

        # self.gru = GRU(input_size=args.d_model, hidden_size=args.d_model // 2,
        #                dropout=args.enc_drop,
        #                bidirectional=True)
        #
        # self.transformer = TransformerEncoder(max_pos_embeddings=args.max_pos_embeddings,
        #                                       nb_layers=args.encoder_layer,
        #                                       nb_heads=args.nb_heads,
        #                                       d_model=args.d_model,
        #                                       d_ff=args.d_ff,
        #                                       dropout=args.enc_drop,
        #                                       att_drop=args.att_drop)

        self.transformer_xl = TransformerXL(d_model=args.d_model,
                                            d_head=args.d_model // args.nb_heads,
                                            n_head=args.nb_heads,
                                            d_inner=args.d_ff,
                                            n_layer=args.encoder_layer,
                                            dropout=args.att_drop)

        self.scale = ScalarMix(mixture_size=args.encoder_layer+1)

        # self.fc = nn.Linear(in_features=d_model, out_features=args.hidden_size)

        self.att_layer = DotProductAttention(k_dim=args.d_model)

        self.tag_embedding = nn.Parameter(torch.zeros(args.tag_size, args.tag_embed_dim))
        self.tag_mlp = nn.Linear(in_features=args.d_model, out_features=args.tag_size)
        self.tag_crf = CRF(num_tags=args.tag_size, batch_first=True)

        # in_features = args.hidden_size + args.tag_embed_dim
        in_features = args.d_model + args.tag_embed_dim
        self._activation = nn.ReLU()
        # self._activation = nn.LeakyReLU(0.1)
        # self._activation = nn.ELU()

        self.mlp_arc = NonlinearMLP(in_feature=in_features,
                                    out_feature=args.arc_mlp_size * 2,
                                    activation=self._activation)
        self.mlp_lbl = NonlinearMLP(in_feature=in_features,
                                    out_feature=args.label_mlp_size * 2,
                                    activation=self._activation)

        self.arc_biaffine = Biaffine(args.arc_mlp_size,
                                     1, bias=(True, False))

        self.label_biaffine = Biaffine(args.label_mlp_size,
                                       args.rel_size, bias=(True, True))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.tag_embedding)
        # nn.init.normal_(self.tag_embedding)
        with torch.no_grad():
            self.tag_embedding[0].fill_(0)

    def forward(self, bert_ids, bert_lens, bert_mask):
        mask = bert_lens.ne(0)
        bert_embed = self.bert_embedding(bert_ids, bert_lens, bert_mask)

        # last_out, enc_outs = self.transformer(bert_embed, mask)
        last_out, enc_outs = self.transformer_xl(bert_embed, mask)
        enc_out = self.scale(enc_outs)

        att_out = self.att_layer(last_out, enc_out, enc_out, mask)
        tag_score = self.tag_mlp(att_out)
        tag_probs = F.softmax(tag_score, dim=-1)
        tag_embed = torch.matmul(tag_probs, self.tag_embedding)

        dep_embed = torch.cat((enc_out, tag_embed), dim=-1).contiguous()

        if self.training:
            dep_embed = timestep_dropout(dep_embed, self.args.embed_drop)

        arc_input, lbl_input = dep_embed, dep_embed

        arc_feat = self.mlp_arc(arc_input)
        lbl_feat = self.mlp_lbl(lbl_input)

        arc_head, arc_dep = arc_feat.chunk(2, dim=-1)
        lbl_head, lbl_dep = lbl_feat.chunk(2, dim=-1)

        if self.training:
            arc_head = timestep_dropout(arc_head, self.args.arc_mlp_drop)
            arc_dep = timestep_dropout(arc_dep, self.args.arc_mlp_drop)
        arc_score = self.arc_biaffine(arc_dep, arc_head)

        if self.training:
            lbl_head = timestep_dropout(lbl_head, self.args.label_mlp_drop)
            lbl_dep = timestep_dropout(lbl_dep, self.args.label_mlp_drop)
        lbl_score = self.label_biaffine(lbl_dep, lbl_head)

        return tag_score, arc_score, lbl_score

    def tag_loss(self, tag_score, gold_tags, char_mask=None):
        lld = self.tag_crf(tag_score, tags=gold_tags, mask=char_mask)
        return lld.neg()

    def tag_decode(self, tag_emissions, char_mask):
        # return best segment tags
        best_tag_seq = self.tag_crf.decode(tag_emissions, mask=char_mask)
        return pad_sequence(best_tag_seq, batch_first=True, padding_value=0)

