import os
from datautil.dependency import read_deps
from collections import Counter
from functools import wraps
from transformers import BertTokenizer
import pickle


def create_vocab(data_path, bert_vocab_path):
    assert os.path.exists(data_path)
    root_rel = ''
    tag_counter, rel_counter = Counter(), Counter()
    with open(data_path, 'r', encoding='utf-8') as fr:
        for deps in read_deps(fr):
            for dep in deps:
                tag_counter[dep.tag] += 1
                if dep.head != 0:
                    rel_counter[dep.dep_rel] += 1
                elif root_rel == '':
                    root_rel = dep.dep_rel
                    rel_counter[dep.dep_rel] += 1
                elif root_rel != dep.dep_rel:
                    print('root = ' + root_rel + ', rel for root = ' + dep.dep_rel)
    return DepVocab(bert_vocab_path, tag_counter, rel_counter, root_rel)


def _check_build_bert_vocab(func):
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        if self.bert_tokenizer is None:
            self.build_vocab()
        return func(self, *args, **kwargs)

    return _wrapper


class DepVocab(object):
    def __init__(self, bert_vocab_path=None,
                 tag_counter: Counter = None,
                 rel_counter: Counter = None,
                 root_rel='root',
                 padding='<pad>',
                 unknown='<unk>'):
        
        self.root_rel = root_rel
        self.root_form = '<'+root_rel.lower()+'>'
        self.padding = padding
        self.unknown = unknown

        self._tag2idx = None
        self._idx2tag = None
        self._rel2idx = None
        self._idx2rel = None

        self.bert_vocab = bert_vocab_path
        self.tag_counter = tag_counter
        self.rel_counter = rel_counter
        self.bert_tokenizer = None

    def build_vocab(self):
        if self.tag_counter is not None:
            if self._tag2idx is None:
                self._tag2idx = dict()
                if self.padding is not None:
                    self._tag2idx[self.padding] = len(self._tag2idx)
                if self.root_rel is not None:
                    self._tag2idx[self.root_rel] = len(self._tag2idx)
                if self.unknown is not None:
                    self._tag2idx[self.unknown] = len(self._tag2idx)

            for tag in self.tag_counter.keys():
                if tag not in self._tag2idx:
                    self._tag2idx[tag] = len(self._tag2idx)
            self._idx2tag = dict((idx, tag) for tag, idx in self._tag2idx.items())
            del self.tag_counter

        if self.rel_counter is not None:
            if self._rel2idx is None:
                self._rel2idx = dict()
                if self.padding is not None:
                    self._rel2idx[self.padding] = len(self._rel2idx)
                if self.root_rel is not None:
                    self._rel2idx[self.root_rel] = len(self._rel2idx)

            for rel in self.rel_counter.keys():
                if rel not in self._rel2idx:
                    self._rel2idx[rel] = len(self._rel2idx)
            self._idx2rel = dict((idx, rel) for rel, idx in self._rel2idx.items())
            del self.rel_counter

        if self.bert_tokenizer is None and self.bert_vocab is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_vocab)
            print("Load bert_util vocabulary finished !!!")
        return self

    @_check_build_bert_vocab
    def bert2id(self, tokens: list):
        '''将原始token序列转换成bert bep ids'''
        def transform(token):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))

        bert_ids, bert_lens = [], []
        tokenizer = self.bert_tokenizer
        cls_tokens = [tokenizer.cls_token] + tokens
        bert_piece_ids = map(transform, cls_tokens)
        for piece in bert_piece_ids:
            if not piece:
                piece = transform(tokenizer.pad_token)
            bert_ids.extend(piece)
            bert_lens.append(len(piece))
        bert_mask = [1] * len(bert_ids)
        return bert_ids, bert_lens, bert_mask

    # @_check_build_bert_vocab
    # def bert_ids(self, seqs):
    #     def transform(token):
    #         return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
    #
    #     tokenizer = self.bert_tokenizer
    #     bert_ids, bert_lens = [], []
    #     # seqs = [[self.key_words['cls']] + list(seq) for seq in seqs]
    #     seqs_iter = map(lambda x: [self.key_words['cls']] + list(x), seqs)
    #     for seq in seqs_iter:
    #         seq = list(map(transform, seq))  # 将每个序列中的原始token转换bpe id
    #         # seq = [transform(token) for token in seq]
    #         # seq = [piece if piece else transform(self.key_words['pad']) for piece in seq]
    #         bert_ids.append(sum(seq, []))  # 将每个bpe id list合并一个
    #         bert_lens.append([len(piece) for piece in seq])
    #     bert_mask = [[1] * len(piece) for piece in bert_ids]
    #
    #     return bert_ids, bert_lens, bert_mask

    @_check_build_bert_vocab
    def save(self, path):
        with open(path, 'wb') as fw:
            pickle.dump(self, fw)

    @_check_build_bert_vocab
    def tag2index(self, tag):
        if isinstance(tag, list):
            return [self._tag2idx.get(p, self._tag2idx[self.unknown]) for p in tag]
        else:
            return self._tag2idx.get(tag, self._tag2idx[self.unknown])

    @_check_build_bert_vocab
    def index2tag(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2tag.get(i, self.unknown) for i in idxs]
        else:
            return self._idx2tag.get(idxs, self.unknown)

    @_check_build_bert_vocab
    def rel2index(self, rels):
        if isinstance(rels, list):
            return [self._rel2idx.get(rel) for rel in rels]
        else:
            return self._rel2idx.get(rels)

    @_check_build_bert_vocab
    def index2rel(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2rel.get(i) for i in idxs]
        else:
            return self._idx2rel.get(idxs)

    @property
    @_check_build_bert_vocab
    def tag_size(self):
        return len(self._tag2idx)

    @property
    @_check_build_bert_vocab
    def rel_size(self):
        return len(self._rel2idx)

