from .dependency import Dependency
from typing import List


def calc_prf(num_gold, num_pred, num_correct, eps=1e-10):
    p = num_correct / (num_pred + eps)
    r = num_correct / (num_gold + eps)
    f1 = (2. * num_correct) / (num_pred + num_gold + eps)
    return p, r, f1


def cws_from_tag(deps: List[Dependency]):
    wds = []
    one_wd = []
    is_start = False
    for i, dep in enumerate(deps):
        if '#' not in dep.tag and dep.tag not in 'bmes':
            is_start = False
            continue

        tag_bound = dep.tag.split('#')[1] if '#' in dep.tag else dep.tag
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
                tags = ['#' in w.tag for w in one_wd]
                if all(tags) or not any(tags):
                    wds.append(one_wd)
                one_wd = []
            is_start = False
    return wds


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
                break

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

        head_idx = idx_head_in_wd(gold_seg)
        gold_span = (gold_seg[0].id, gold_seg[-1].id)
        for pred_seg in pred_seg_lst[start_idx:]:
            pred_span = (pred_seg[0].id, pred_seg[-1].id)
            if gold_span == pred_span:
                if gold_seg[head_idx].head == pred_seg[head_idx].head:
                    nb_arc_correct += 1
                    if gold_seg[head_idx].dep_rel == pred_seg[head_idx].dep_rel:
                        nb_rel_correct += 1
                break
    return nb_gold_arcs, nb_pred_arcs, nb_arc_correct, nb_rel_correct


def idx_head_in_wd(ch_deps: List[Dependency]):
    if len(ch_deps) == 1:
        return 0
    head_ch = -1
    b, e = ch_deps[0].id, ch_deps[-1].id
    for cd in ch_deps:
        if not (b <= cd.head <= e):
            head_ch = cd.id - b
            break
    return head_ch