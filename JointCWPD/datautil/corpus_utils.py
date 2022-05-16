from typing import List
from pprint import pprint


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
            str(self.id),
            self.form,
            self.lemma,
            self.cpos_tag,
            self.pos_tag,
            self.feats,
            str(self.head),
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
    wds = set()
    deps = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split('\t')
            if tokens is None or len(tokens) == 0 or line.strip() == '':
                yield deps, wds
                deps = []
            elif len(tokens) == 10:
                if tokens[6] == '_':
                    tokens[6] = '-1'
                wds.add(tokens[1])
                deps.append(DepNode(*tokens))
            else:
                pass
    if len(deps) > 0:
        yield deps, wds


def save_conll(path, deps):
    with open(path, 'a', encoding='utf-8') as fw:
        for dep in deps:
            fw.write(str(dep))
            fw.write('\n')
        fw.write('\n')


# 将混合标签的词性bi转成bmes形式
# NR#b NR#i NR#i -> NR#b NR#m NR#e
def transform_tag(deps: List[DepNode]):
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


def cws_from_dep(deps: List[DepNode], app='#in'):
    wds = []
    one_wd = []
    seq_len = len(deps)
    for i, dep in enumerate(deps):
        one_wd.append(dep)
        if (i+1 == seq_len) \
                or (dep.head != deps[i+1].id and dep.id != deps[i+1].head) \
                or (dep.dep_rel != app and deps[i+1].dep_rel != app):
            wds.append(one_wd)
            one_wd = []
    return wds


def cws_from_postag(deps: List[DepNode]):
    wds = []
    one_wd = []
    is_start = False
    for i, dep in enumerate(deps):
        if '#' not in dep.cpos_tag:
            is_start = False
            continue

        tag_bound = dep.cpos_tag.split('#')[1]
        if tag_bound == 's':
            wds.append([dep])
            is_start = False
        elif tag_bound == 'b':
            one_wd = [dep]
            is_start = True
        elif tag_bound == 'm':
            if is_start:
                one_wd.append(dep)
        elif tag_bound == 'e':
            if is_start:
                one_wd.append(dep)
                wds.append(one_wd)
                one_wd = []
            is_start = False
    return wds


def idx_head_in_wd(ch_deps: List[DepNode]):
    if len(ch_deps) == 1:
        return 0
    head_ch = -1
    for cd in ch_deps:
        if not (ch_deps[0].id <= cd.head <= ch_deps[-1].id):
            head_ch = cd.id - ch_deps[0].id
            break
    return head_ch


def count(in_path, char=False):
    ws = 0
    sent = 0
    for deps, wds in read_conll(in_path):
        sent += 1
        if char:
            wdps = cws_from_postag(deps)
            ws += len(wdps)
        else:
            ws += len(wds)
    print('word:', ws)
    print('sent:', sent)
    return ws


def process(in_path, out_path):
    fw = open('../data/ctb50/train.ctb50.conll', 'a', encoding='utf-8')
    for deps in read_conll(in_path):
        # transform_tag(deps)
        # save_conll(out_path, deps)
        wdps = cws_from_postag(deps)
        id = 1
        l = 0
        deps = []
        itab = {dep[idx_head_in_wd(dep)].id: i+1 for i, dep in enumerate(wdps)}
        for dep in wdps:
            l += len(dep)
            form = ''.join([d.form for d in dep])
            pos_tag = cpos_tag = dep[0].cpos_tag.split('#')[0]
            head_ch = dep[idx_head_in_wd(dep)]
            dep_rel = head_ch.dep_rel
            if head_ch.head != 0:
                head = itab[head_ch.head]
            else:
                head = 0
            dep_node = DepNode(id, form, '_', cpos_tag, pos_tag, '_', head, dep_rel, '_', '_')
            fw.write(str(dep_node)+'\n')
            deps.append(dep_node)
            id += 1
        fw.write('\n')
    fw.close()

from collections import Counter
def check_len(deps):
    len_counter = Counter()
    for d in deps:
        len_counter[len(d)] += 1
    if 1 in len_counter and 2 in len_counter and 3 in len_counter:
        return True
    else:
        return False

def process2(in_path, out_path):
    for deps, wds in read_conll(in_path):
        chs = []
        if 10 <= len(deps) <= 11 and check_len(cws_from_postag(deps)):
            for d in deps:
                chs.append(d.form)
                print(d, end='  ')
            print(''.join(chs))
            print('\n\n')