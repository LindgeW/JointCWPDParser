import torch.nn as nn
from .layer import NonlinearMLP, Biaffine
from .dropout import timestep_dropout
import torch
from .scale_mix import ScalarMix
import torch.nn.functional as F
from .crf import CRF
from torch.nn.utils.rnn import pad_sequence


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.tag_embedding = nn.Parameter(torch.zeros(args.tag_size, args.tag_embed_dim))
        self.tag_mlp = nn.Linear(in_features=args.d_model, out_features=args.tag_size)
        # self.tag_crf = CRF(num_tags=args.tag_size, batch_first=True)
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

        self.scale_tag = ScalarMix(mixture_size=args.encoder_layer)
        self.scale_dep = ScalarMix(mixture_size=args.encoder_layer)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.tag_mlp.weight)
        nn.init.xavier_uniform_(self.tag_embedding)
        with torch.no_grad():
            self.tag_embedding[0].fill_(0)

    def forward(self, bert_out, mask):
        tag_out = self.scale_tag(bert_out)
        dep_out = self.scale_dep(bert_out)
        tag_score = self.tag_mlp(tag_out)
        tag_probs = F.softmax(tag_score, dim=-1)
        tag_embed = torch.matmul(tag_probs, self.tag_embedding)

        dep_embed = torch.cat((dep_out, tag_embed), dim=-1).contiguous()
        if self.training:
            dep_embed = timestep_dropout(dep_embed, self.args.embed_drop)

        arc_feat = self.mlp_arc(dep_embed)
        lbl_feat = self.mlp_lbl(dep_embed)

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

    def tag_loss(self, tag_score, gold_tags, char_mask=None, reduction='mean'):
        lld = self.tag_crf(tag_score, tags=gold_tags, mask=char_mask, reduction=reduction)
        return lld.neg()

    def tag_decode(self, tag_emissions, char_mask):
        # return best segment tags
        best_tag_seq = self.tag_crf.decode(tag_emissions, mask=char_mask)
        return pad_sequence(best_tag_seq, batch_first=True, padding_value=0)


class ParserModel(nn.Module):
    def __init__(self, model, bert=None):
        super(ParserModel, self).__init__()
        self.bert = bert
        self.model = model

    def forward(self, bert_ids, bert_lens, bert_mask):
        mask = bert_lens.gt(0)
        bert_embed = self.bert(bert_ids, bert_lens, bert_mask)
        tag_score, arc_score, lbl_score = self.model(bert_embed, mask)
        return tag_score, arc_score, lbl_score

    def tag_loss(self, tag_score, gold_tags, char_mask=None):
        return self.model.tag_loss(tag_score, gold_tags, char_mask)

    def tag_decode(self, tag_emissions, char_mask):
        return self.model.tag_decode(tag_emissions, char_mask)
