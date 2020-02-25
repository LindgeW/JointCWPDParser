from datautil.dependency import Dependency
from typing import List


# 计算F1值
def calc_f1(num_gold, num_pred, num_correct, eps=1e-10):
    precision = num_correct / num_pred
    recall = num_correct / num_gold
    f1 = (2. * precision * recall) / (precision + recall + eps)
    return f1


# 找到一棵字符级别的依存树中找到单词序列
# def cws_from_dep(deps: List[Dependency], app='#in'):
#     wds = []
#     one_wd = []
#     seq_len = len(deps)
#     for i, dep in enumerate(deps):
#         one_wd.append(dep)
#         if (i+1 == seq_len) or dep.id != deps[i+1].head or deps[i+1].dep_rel != app:
#             wds.append(one_wd)
#             one_wd = []
#     return wds


# 利用依存标签获取词的边界（字符级语料--词的最后一个字符为其他字符的head）
def cws_from_dep(deps: List[Dependency], app='#in'):
    wds = []
    one_wd = []
    seq_len = len(deps)
    for i, dep in enumerate(deps):
        one_wd.append(dep)
        if (i+1 == seq_len) or dep.head != deps[i+1].id or dep.dep_rel != app:
            wds.append(one_wd)
            one_wd = []
    return wds


# # 利用词性标签获取分词的边界bi（不考虑词性标签）
# # NR#b  NR#i  DEG#b VV#b  NN#b  NN#i  NN#i
def cws_from_postag_bi(deps: List[Dependency]):
    wds = []
    one_wd = []
    is_start = False
    end_idx = len(deps) - 1
    for i, dep in enumerate(deps):
        if '#' not in dep.tag:
            is_start = False
            continue
        tag_bound = dep.tag.split('#')[1]
        one_wd.append(dep)
        if tag_bound == 'b':
            is_start = True
            if i == end_idx or '#' not in deps[i+1].tag or deps[i+1].tag.split('#')[1] != 'i':
                wds.append(one_wd)
                one_wd = []
                is_start = False
        elif tag_bound == 'i' and (i == end_idx or '#' not in deps[i+1].tag or deps[i+1].tag.split('#')[1] != 'i'):
            if is_start:
                wds.append(one_wd)
                is_start = False
            one_wd = []
    return wds


# 利用词性标签获取分词的边界bmes（不考虑词性标签的正确性）
# NR#b  NR#e  DEG#s VV#s  NN#b  NN#m  NN#e
def cws_from_postag(deps: List[Dependency]):
    wds = []
    one_wd = []
    is_start = False
    for i, dep in enumerate(deps):
        if '#' not in dep.tag:
            is_start = False
            continue

        tag_bound = dep.tag.split('#')[1]
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


# 利用依存标签计算分词的F1分数
def calc_seg_f1(gold_seg_lst: List, pred_seg_lst: List):
    start_idx = 1 if gold_seg_lst[0][0].id == 0 else 0

    num_gold = len(gold_seg_lst) - start_idx
    num_pred = len(pred_seg_lst) - start_idx  # 排除<root>

    gold_bounds, pred_bounds = [], []
    for gold_segs in gold_seg_lst[start_idx:]:
        if len(gold_segs) > 1:
            gold_bounds.append((gold_segs[0].id, gold_segs[-1].id))
        else:
            gold_bounds.append(gold_segs[0].id)

    for pred_segs in pred_seg_lst[start_idx:]:
        if len(pred_segs) > 1:
            pred_bounds.append((pred_segs[0].id, pred_segs[-1].id))
        else:
            pred_bounds.append(pred_segs[0].id)

    correct_pred = 0
    for gold in gold_bounds:
        if gold in pred_bounds:
            correct_pred += 1

    return num_gold, num_pred, correct_pred


# 计算词性的F1分数：分词正确+词性正确
def pos_tag_f1(gold_seg_lst: List, pred_seg_lst: List):
    start_idx = 1 if gold_seg_lst[0][0].id == 0 else 0
    num_gold, num_pred, correct_pred = 0, 0, 0

    num_gold = len(gold_seg_lst) - start_idx
    num_pred = len(pred_seg_lst) - start_idx

    for gold_seg in gold_seg_lst[start_idx:]:
        gold_span = (gold_seg[0].id, gold_seg[-1].id)
        for pred_seg in pred_seg_lst[start_idx:]:
            pred_span = (pred_seg[0].id, pred_seg[-1].id)
            if gold_span == pred_span:
                is_correct = True
                for pred_c, gold_c in zip(pred_seg, gold_seg):
                    if pred_c.tag != gold_c.tag:
                        is_correct = False
                        break
                if is_correct:
                    correct_pred += 1

    return num_gold, num_pred, correct_pred


def parser_metric(gold_seg_lst: List, pred_seg_lst: List):
    # 不考虑标点符号
    ignore_tags = {'``', "''", ':', ',', '.', 'PU', 'PU#b', 'PU#i', 'PU#m', 'PU#e', 'PU#s'}

    start_idx = 1 if gold_seg_lst[0][0].id == 0 else 0

    nb_gold_arcs, nb_pred_arcs = 0, 0  # 实际的弧的数量，预测的弧的数量
    nb_arc_correct, nb_rel_correct = 0, 0

    for pred_seg in pred_seg_lst[start_idx:]:
        if pred_seg[0].tag in ignore_tags:
            continue
        nb_pred_arcs += 1

    for gold_seg in gold_seg_lst[start_idx:]:
        if gold_seg[0].tag in ignore_tags:
            continue
        nb_gold_arcs += 1

        gold_span = (gold_seg[0].id, gold_seg[-1].id)
        for pred_seg in pred_seg_lst[start_idx:]:
            pred_span = (pred_seg[0].id, pred_seg[-1].id)
            if gold_span == pred_span:
                if gold_seg[-1].head == pred_seg[-1].head:
                    nb_arc_correct += 1
                    if gold_seg[-1].dep_rel == pred_seg[-1].dep_rel:
                        nb_rel_correct += 1

    return nb_gold_arcs, nb_pred_arcs, nb_arc_correct, nb_rel_correct


from pprint import pprint
if __name__ == '__main__':
    # tag_lst = 'NR#b xxx NR#b xxx NR#m NR#m NR#e NR#m NR#m NR#e DEG#b NR#m NN#b NN#e tooy NR#s DEG#s NN#s NR#b NR#e NR#b'.split(' ')
    tag_lst = 'NR#s xxx NR#b NR#m NR#m NR#m NR#e NR#s DEG#s NR#b NR#m NR#e NR#e NR#s NR#b NR#s'.split(' ')
    # tag_lst = 'NR#b NR#b NR#b xxx NR#i NR#b NR#i NR#b NR#b NR#i DEG#b NR#i NN#b NN#i NR#i DEG#i NN#i xxx NR#b NR#i NR#b NR#i'.split(' ')

    print(len(tag_lst))
    deps = [Dependency(sid=i, form='OK', head=i, dep_rel='OK', tag=tag) for i, tag in enumerate(tag_lst)]
    # wd_deps = cws_from_postag_bi(deps)
    wd_deps = cws_from_postag(deps)
    pprint(wd_deps)


