import os
from .dependency import read_deps
import numpy as np
import torch


# Dependency对象列表的列表：[[], [], [], ...]
def load_dataset(path, vocab=None, max_len=512):
    assert os.path.exists(path)
    dataset = []
    with open(path, 'r', encoding='utf-8') as fr:
        for deps in read_deps(fr, vocab):
            if len(deps) < max_len:
                dataset.append(deps)
    return dataset


def batch_iter(dataset: list, batch_size, shuffle=False):
    if shuffle:
        np.random.shuffle(dataset)

    nb_batch = int(np.ceil(len(dataset) / batch_size))
    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        yield batch_data


def batch_variable(batch_data, dep_vocab, device=torch.device('cpu')):
    batch_size = len(batch_data)
    max_seq_len = max(len(deps) for deps in batch_data)
    tag_idxs = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    head_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    rel_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)

    # bert_seqs = []
    bert_ids, bert_lens, bert_masks = [], [], []
    for i, deps in enumerate(batch_data):
        forms = []
        for j, dep in enumerate(deps):
            if dep.id != 0:
                forms.append(dep.form)
            tag_idxs[i, j] = dep_vocab.tag2index(dep.tag)
            head_idx[i, j] = dep.head
            rel_idx[i, j] = dep_vocab.rel2index(dep.dep_rel)
        # bert_seqs.append(forms)

        bert_id, bert_len, bert_mask = dep_vocab.bert2id(forms)
        bert_ids.append(bert_id)
        bert_lens.append(bert_len)
        bert_masks.append(bert_mask)

    # bert_ids, bert_lens, bert_mask = dep_vocab.bert_ids(bert_seqs)
    bert_ids = pad_sequence(bert_ids, dtype=torch.long, device=device)
    bert_lens = pad_sequence(bert_lens, dtype=torch.int, device=device)
    bert_masks = pad_sequence(bert_masks, dtype=torch.bool, device=device)
    return (bert_ids, bert_lens, bert_masks), tag_idxs, head_idx, rel_idx


def pad_sequence(seqs: list, device=torch.device('cpu'), dtype=torch.int, padding_val=0):
    max_len = max(len(s) for s in seqs)
    pad_seqs = torch.zeros(len(seqs), max_len, dtype=dtype, device=device).fill_(padding_val)
    for i, seq in enumerate(seqs):
        pad_seqs[i, :len(seq)] = torch.tensor(seq, dtype=dtype, device=device)
    return pad_seqs