from typing import List
from pprint import pprint


# ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
class DepNode(object):
    def __init__(self, id,
                 form,
                 lemma,
                 cpos_tag,
                 pos_tag,
                 feats,
                 head,
                 dep_rel,
                 phead,
                 pdep_rel):
        self.id = int(id)
        self.form = form
        self.lemma = lemma
        self.cpos_tag = cpos_tag  #
        self.pos_tag = pos_tag  #
        self.feats = feats
        self.head = int(head)
        self.dep_rel = dep_rel
        self.phead = phead
        self.pdep_rel = pdep_rel

    def __str__(self):
        return '\t'.join([
            self.id,
            self.form,
            self.lemma,
            self.cpos_tag,
            self.pos_tag,
            self.feats,
            self.head,
            self.dep_rel,
            self.phead,
            self.pdep_rel
        ])


def is_proj(deps):
    if deps[0].id == 1:
        deps = [None] + deps

    n = len(deps)
    for i in range(1, n):
        hi = deps[i].head
        for j in range(i+1, hi):
            hj = deps[j].head
            if (hj - hi) * (hj - i) > 0:
                return False
    return True


def read_conll(path):
    deps = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split('\t')
            if tokens is None or line.strip() == '':
                yield deps
                deps = []
            else:
                assert len(tokens) == 10
                deps.append(DepNode(*tokens))


def save_conll(path, deps):
    with open(path, 'a', encoding='utf-8') as fw:
        for dep in deps:
            fw.write(str(dep))
            fw.write('\n')
        fw.write('\n')


# 将混合标签的词性bi转成bmes形式
# NR#b NR#i NR#i -> NR#b NR#m NR#e
def transform_pos_tag(deps: List[DepNode]):
    is_start = False
    one_wd = []
    end_idx = len(deps)-1
    for i, dep in enumerate(deps):
        tag_bound = dep.cpos_tag.split('#')[1]
        one_wd.append(dep)
        if tag_bound == 'b':
            is_start = True
            if i == end_idx or deps[i+1].cpos_tag.split('#')[1] != 'i':
                one_wd[0].cpos_tag = one_wd[0].cpos_tag.split('#')[0] + '#s'
                one_wd = []
                is_start = False
        elif tag_bound == 'i' and (i == end_idx or deps[i+1].cpos_tag.split('#')[1] != 'i'):
            if is_start:
                if len(one_wd) > 1:
                    one_wd[-1].cpos_tag = one_wd[-1].cpos_tag.split('#')[0] + '#e'
                    for ch in one_wd[1:-1]:
                        ch.cpos_tag = ch.cpos_tag.split('#')[0] + '#m'
                one_wd = []
                is_start = False


def process(in_path, out_path):
    for deps in read_conll(in_path):
        transform_pos_tag(deps)
        save_conll(out_path, deps)


if __name__ == '__main__':
    process('../data/ctb70/test.ctb70-yiou.charconll_new.r',
            '../data/ctb50/test.ctb50.charconll_new.txt')
