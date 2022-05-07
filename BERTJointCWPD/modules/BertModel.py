import torch
import torch.nn as nn
from transformers import BertModel
from .scale_mix import ScalarMix


class BertEmbedding(nn.Module):
    def __init__(self, model_path, nb_layers, out_dim, merge='linear', requires_grad=False):
        super(BertEmbedding, self).__init__()
        assert merge.lower() in ['none', 'linear', 'mean']
        self.merge = merge.lower()
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)
        self.requires_grad = requires_grad
        self.bert_layers = self.bert.config.num_hidden_layers
        self.nb_layers = nb_layers if nb_layers < self.bert_layers else self.bert_layers
        self.hidden_size = self.bert.config.hidden_size
        self.scale = ScalarMix(self.nb_layers)
        self.proj = nn.Linear(in_features=self.hidden_size, out_features=out_dim, bias=False)
        for p in self.bert.named_parameters():
            p[1].requires_grad = False
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, bert_ids, bert_lens, bert_mask):
        '''
        :param bert_ids: (bz, bpe_seq_len) subword indexs
        :param bert_lens: (bz, seq_len)  每个token经过bpe切词后的长度
        :param bert_mask: (bz, bep_seq_len)  经过bpe切词
        :return:
        '''
        bz, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)

        with torch.no_grad():
            _, _, all_enc_outs = self.bert(bert_ids, attention_mask=bert_mask)
            # _, _, all_enc_outs = self.bert_util(bert_ids)
            top_enc_outs = all_enc_outs[-self.nb_layers:]

        if self.merge == 'linear':
            bert_out = self.scale(top_enc_outs)  # (bz, seq_len, 768)
        elif self.merge == 'mean':
            bert_out = sum(top_enc_outs) / len(top_enc_outs)
        else:
            # bert_out = last_enc_out
            bert_out = top_enc_outs[-2]

        # 根据bert piece长度切分
        bert_chunks = bert_out[bert_mask].split(bert_lens[mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
        bert_embed = bert_out.new_zeros(bz, seq_len, self.hidden_size)
        # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
        bert_embed = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)
        return self.proj(bert_embed)