import torch.nn as nn
from .layer import NonlinearMLP, Biaffine
from .encoder.transformer_RPE import TransformerXL
from .encoder.bilstm import BiLSTMEncoder
from .CPUEmbedding import CPUEmbedding
from .dropout import *
import torch
from .scale_mix import ScalarMix
import torch.nn.functional as F
from .crf import CRF
from torch.nn.utils.rnn import pad_sequence


class ParserModel(nn.Module):
    def __init__(self, args, char_embed_weights=None, bichar_embed_weights=None):
        super(ParserModel, self).__init__()
        self.args = args
        self.char_embedding = nn.Embedding(num_embeddings=args.char_vocab_size,
                                           embedding_dim=args.char_embed_dim,
                                           padding_idx=0)
        self.bichar_embedding = nn.Embedding(num_embeddings=args.bichar_vocab_size,
                                           embedding_dim=args.char_embed_dim,
                                           padding_idx=0)

        self.pre_char_embedding = CPUEmbedding(num_embeddings=char_embed_weights.shape[0],
                                            embedding_dim=args.char_embed_dim,
                                            padding_idx=0)
        self.pre_char_embedding.weight.data.copy_(torch.from_numpy(char_embed_weights))
        self.pre_char_embedding.weight.requires_grad = False

        self.pre_bichar_embedding = CPUEmbedding(num_embeddings=bichar_embed_weights.shape[0],
                                            embedding_dim=args.char_embed_dim,
                                            padding_idx=0)
        self.pre_bichar_embedding.weight.data.copy_(torch.from_numpy(bichar_embed_weights))
        self.pre_bichar_embedding.weight.requires_grad = False

        d_model = 2 * args.char_embed_dim

        self.transformer_xl = TransformerXL(d_model=d_model,
                                            d_head=d_model // args.nb_heads,
                                            n_head=args.nb_heads,
                                            d_inner=args.d_ff,
                                            n_layer=args.encoder_layer,
                                            dropout=args.att_drop)

        self.scale_tag = ScalarMix(mixture_size=args.encoder_layer+1)
        self.scale_dep = ScalarMix(mixture_size=args.encoder_layer + 1)

        self.tag_embedding = nn.Parameter(torch.zeros(args.tag_size, args.tag_embed_dim))
        self.tag_mlp = nn.Linear(in_features=d_model, out_features=args.tag_size)
        self.tag_crf = CRF(num_tags=args.tag_size, batch_first=True)

        in_features = d_model + args.tag_embed_dim
        # self._activation = nn.ReLU()
        self._activation = nn.LeakyReLU(0.1)

        self.mlp_arc = NonlinearMLP(in_feature=in_features,
                                    out_feature=args.arc_mlp_size * 2,
                                    activation=self._activation)
        self.mlp_lbl = NonlinearMLP(in_feature=in_features,
                                    out_feature=args.label_mlp_size * 2,
                                    activation=self._activation)

        self.arc_biaffine = Biaffine(args.arc_mlp_size, 1, bias=(True, False))
        self.label_biaffine = Biaffine(args.label_mlp_size, args.rel_size, bias=(True, True))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.tag_mlp.weight)
        nn.init.xavier_uniform_(self.char_embedding.weight)
        nn.init.xavier_uniform_(self.bichar_embedding.weight)
        nn.init.xavier_uniform_(self.tag_embedding)
        with torch.no_grad():
            self.tag_embedding[0].fill_(0)

    def forward(self, ngrams, extngrams, mask=None):
        unigram, bigram = ngrams
        unigram_embed = self.char_embedding(unigram)
        bigram_embed = self.bichar_embedding(bigram)

        ext_unigram, ext_bigram = extngrams
        extunigram_embed = self.pre_char_embedding(ext_unigram)
        extbigram_embed = self.pre_bichar_embedding(ext_bigram)

        unigram_embed_ = unigram_embed + extunigram_embed
        bigram_embed_ = bigram_embed + extbigram_embed
        ngram_embed = torch.cat((unigram_embed_, bigram_embed_), dim=-1).contiguous()

        last_out, enc_outs = self.transformer_xl(ngram_embed, mask)
        enc_out_seg = self.scale_tag(enc_outs)
        enc_out_dep = self.scale_dep(enc_outs)

        tag_score = self.tag_mlp(enc_out_seg)
        tag_probs = F.softmax(tag_score, dim=-1)
        tag_embed = torch.matmul(tag_probs, self.tag_embedding)
        dep_embed = torch.cat((enc_out_dep, tag_embed), dim=-1).contiguous()

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