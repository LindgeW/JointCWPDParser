import os
from datautil.dependency import read_deps
from collections import Counter
import numpy as np
from functools import wraps


def ngram_generator(token_lst: list, n=2):
    ngrams = []
    for i in range(len(token_lst)-n+1):
        ngrams.append(''.join(token_lst[i:i+n]))
    return ngrams


def create_vocab(data_path, min_count=2):
    assert os.path.exists(data_path)

    root_rel = ''
    char_counter, bichar_counter = Counter(), Counter()
    tag_counter, rel_counter = Counter(), Counter()
    with open(data_path, 'r', encoding='utf-8') as fr:
        for deps in read_deps(fr):
            unigrams = []
            for dep in deps:
                unigrams.append(dep.form)
                tag_counter[dep.tag] += 1
                if dep.head != 0:
                    rel_counter[dep.dep_rel] += 1
                elif root_rel == '':
                    root_rel = dep.dep_rel
                    rel_counter[dep.dep_rel] += 1
                elif root_rel != dep.dep_rel:
                    print('root = ' + root_rel + ', rel for root = ' + dep.dep_rel)
            bigrams = ngram_generator(unigrams, n=2)
            # trigrams = ngram_generator(unigrams, n=3)
            char_counter.update(unigrams)
            bichar_counter.update(bigrams)

    char_vocab = DepVocab(char_counter, tag_counter, rel_counter, root_rel, min_count=min_count)
    bichar_vocab = DepVocab(bichar_counter, min_count=min_count)
    return char_vocab, bichar_vocab


# 检查词表是否被创建
def _check_build_vocab(func):
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        if self._word2idx is None:
            self.build_vocab()
        return func(self, *args, **kwargs)
    return _wrapper


class DepVocab(object):
    def __init__(self, wd_counter: Counter,
                 tag_counter: Counter = None,
                 rel_counter: Counter = None,
                 root_rel='root',
                 padding='<pad>',
                 unknown='<unk>',
                 min_count=2):
        
        self.root_rel = root_rel
        self.root_form = '<'+root_rel.lower()+'>'
        self.padding = padding
        self.unknown = unknown

        self._word2idx = None
        self._idx2wd = None
        self._extwd2idx = None
        self._extidx2wd = None

        self._tag2idx = None
        self._idx2tag = None

        self._rel2idx = None
        self._idx2rel = None

        self.wd_counter = wd_counter
        self.tag_counter = tag_counter
        self.rel_counter = rel_counter
        self.min_count = min_count

    def build_vocab(self):
        if self._word2idx is None:
            self._word2idx = dict()
            if self.padding is not None:
                self._word2idx[self.padding] = len(self._word2idx)
            if self.root_form is not None:
                self._word2idx[self.root_form] = len(self._word2idx)
            if self.unknown is not None:
                self._word2idx[self.unknown] = len(self._word2idx)

        for wd, freq in self.wd_counter.items():
            if freq >= self.min_count and wd not in self._word2idx:
                self._word2idx[wd] = len(self._word2idx)
        self._idx2wd = dict((idx, wd) for wd, idx in self._word2idx.items())

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
        return self

    @_check_build_vocab
    def get_embedding_weights(self, embed_path):
        if not os.path.exists(embed_path):
            print('embedding path does not exist!')
            return None

        if self._extwd2idx is None:
            self._extwd2idx = dict()
            if self.padding is not None:
                self._extwd2idx[self.padding] = len(self._extwd2idx)
            if self.root_form is not None:
                self._extwd2idx[self.root_form] = len(self._extwd2idx)
            if self.unknown is not None:
                self._extwd2idx[self.unknown] = len(self._extwd2idx)

        vec_size = 0
        with open(embed_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split()
                if len(tokens) > 10:
                    wd = tokens[0]
                    if vec_size == 0:
                        vec_size = len(tokens[1:])
                    if wd not in self._extwd2idx:
                        self._extwd2idx[wd] = len(self._extwd2idx)

        self._extidx2wd = dict((idx, wd) for wd, idx in self._extwd2idx.items())

        oov_ratio = 0
        for wd in self._word2idx.keys():
            if wd not in self._extwd2idx:
                oov_ratio += 1
        print('oov ratio: %.2f%%' % (100 * (oov_ratio-3) / (len(self._word2idx)-3)))

        wd_count = len(self._extwd2idx)
        embed_weights = np.zeros((wd_count, vec_size), dtype=np.float32)
        unk_idx = self._extwd2idx[self.unknown]
        with open(embed_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split()
                if len(tokens) > 10:
                    idx = self._extwd2idx[tokens[0]]
                    vec = np.asarray(tokens[1:], dtype=np.float32)
                    embed_weights[idx] = vec
                    embed_weights[unk_idx] += vec
        # 已知词的词向量初始化的均值初始化未知向量
        embed_weights[unk_idx] /= wd_count
        embed_weights /= np.std(embed_weights)
        return embed_weights

    @_check_build_vocab
    def word2index(self, wds):
        if isinstance(wds, list):
            return [self._word2idx.get(wd, self._word2idx[self.unknown]) for wd in wds]
        else:
            return self._word2idx.get(wds, self._word2idx[self.unknown])

    @_check_build_vocab
    def index2word(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2wd.get(i, self.unknown) for i in idxs]
        else:
            return self._idx2wd.get(idxs, self.unknown)

    def extwd2index(self, wds):
        if isinstance(wds, list):
            return [self._extwd2idx.get(wd, self._extwd2idx[self.unknown]) for wd in wds]
        else:
            return self._extwd2idx.get(wds, self._extwd2idx[self.unknown])

    def extidx2word(self, idxs):
        if isinstance(idxs, list):
            return [self._extidx2wd.get(i, self.unknown) for i in idxs]
        else:
            return self._extidx2wd.get(idxs, self.unknown)

    @_check_build_vocab
    def tag2index(self, tag):
        if isinstance(tag, list):
            return [self._tag2idx.get(p, self._tag2idx[self.unknown]) for p in tag]
        else:
            return self._tag2idx.get(tag, self._tag2idx[self.unknown])

    @_check_build_vocab
    def index2tag(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2tag.get(i, self.unknown) for i in idxs]
        else:
            return self._idx2tag.get(idxs, self.unknown)

    @_check_build_vocab
    def rel2index(self, rels):
        if isinstance(rels, list):
            return [self._rel2idx.get(rel) for rel in rels]
        else:
            return self._rel2idx.get(rels)

    @_check_build_vocab
    def index2rel(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2rel.get(i) for i in idxs]
        else:
            return self._idx2rel.get(idxs)

    @property
    @_check_build_vocab
    def vocab_size(self):
        return len(self._word2idx)
    
    @property
    def extvocab_size(self):
        return len(self._extwd2idx)

    @property
    @_check_build_vocab
    def tag_size(self):
        return len(self._tag2idx)

    @property
    @_check_build_vocab
    def rel_size(self):
        return len(self._rel2idx)

    @property
    @_check_build_vocab
    def padding_idx(self):
        return self._word2idx.get(self.padding)

    @_check_build_vocab
    def __len__(self):
        return len(self._word2idx)

    @_check_build_vocab
    def __iter__(self):
        for wd, idx in self._word2idx.items():
            yield wd, idx

    @_check_build_vocab
    def __contains__(self, item):
        return item in self._word2idx

    @_check_build_vocab
    def __getitem__(self, item):
        if item in self._word2idx:
            return self._word2idx.get(item)
        if self.unknown is not None:
            return self._word2idx[self.unknown]




