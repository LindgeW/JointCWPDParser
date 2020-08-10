from transformers.modeling_bert import *
from transformers import BertModel
from .layer import NonlinearMLP


class BertEmbedding(nn.Module):
    def __init__(self, model_path, nb_layers, out_dim):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)
        self.bert_layers = self.bert.config.num_hidden_layers
        self.nb_layers = nb_layers if nb_layers < self.bert_layers else self.bert_layers
        self.hidden_size = self.bert.config.hidden_size
        # self.proj = nn.Linear(in_features=self.hidden_size, out_features=out_dim)
        # self.projs = nn.ModuleList([NonlinearMLP(self.hidden_size, out_dim, bias=False) for _ in range(self.nb_layers)])
        self.projs = nn.ModuleList([nn.Linear(self.hidden_size, out_dim) for _ in range(self.nb_layers)])

    def forward(self, bert_ids, bert_lens, bert_mask):
        '''
        :param bert_ids: (bz, bpe_seq_len) subword indexs
        :param bert_lens: (bz, seq_len)  每个token经过bpe切词后的长度
        :param bert_mask: (bz, bep_seq_len)  经过bpe切词
        :return:
        '''
        bz, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)

        _, _, all_enc_outs = self.bert(bert_ids, attention_mask=bert_mask)
        bert_outs = all_enc_outs[-self.nb_layers:]

        proj_hiddens = []
        for i, bert_out in enumerate(bert_outs):
            # 根据bert piece长度切分
            bert_chunks = bert_out[bert_mask].split(bert_lens[mask].tolist())
            bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
            bert_embed = bert_out.new_zeros(bz, seq_len, self.hidden_size)
            # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
            bert_embed_ = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)
            proj_hiddens.append(self.projs[i](bert_embed_))

        return proj_hiddens

        # 根据bert piece长度切分
        # bert_out = all_enc_outs[-1:]
        # bert_chunks = bert_out[bert_mask].split(bert_lens[mask].tolist())
        # bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
        # bert_embed = bert_out.new_zeros(bz, seq_len, self.hidden_size)
        # # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
        # bert_embed = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)
        # return self.proj(bert_embed)




