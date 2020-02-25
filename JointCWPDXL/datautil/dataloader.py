import os
from .dependency import read_deps
import numpy as np
import torch


def ngram_producer(line: str, n=2):
    assert n < 1
    ngrams = []
    for i in range(len(line)):
        ngrams.append(line[i: i+n])
    return ngrams


def ngram_generator(token_lst: list, n=2, start_idx=0):
    ngrams = []
    if start_idx < 0:
        start_idx = 0
    for i in range(len(token_lst)):
        if i < start_idx:
            ngrams.append(token_lst[i])
        else:
            ngrams.append(''.join(token_lst[i:i+n]))
    return ngrams


# Dependency对象列表的列表：[[], [], [], ...]
def load_dataset(path, vocab=None):
    assert os.path.exists(path)
    dataset = []
    with open(path, 'r', encoding='utf-8') as fr:
        for deps in read_deps(fr, vocab):
            dataset.append(deps)

    return dataset


def batch_iter(dataset: list, batch_size, shuffle=False):
    if shuffle:
        np.random.shuffle(dataset)

    nb_batch = (len(dataset) + batch_size - 1) // batch_size

    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        if shuffle:
            np.random.shuffle(batch_data)
        yield batch_data


def batch_variable(batch_data, dep_vocab, device=torch.device('cpu')):
    batch_size = len(batch_data)

    max_seq_len = max(len(deps) for deps in batch_data)

    unigram_idxs = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    bigram_idxs = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    extunigram_idxs = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    extbigram_idxs = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)

    tag_idxs = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)

    head_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    rel_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    non_pad_mask = torch.zeros((batch_size, max_seq_len),dtype=torch.uint8, device=device)

    for i, deps in enumerate(batch_data):
        seq_len = len(deps)
        char_form = [dep.form for dep in deps]
        bi_form = ngram_generator(char_form, n=2, start_idx=1)

        unigram_idxs[i, :seq_len] = torch.tensor(dep_vocab.word2index(char_form), device=device)
        bigram_idxs[i, :seq_len] = torch.tensor(dep_vocab.word2index(bi_form), device=device)
        extunigram_idxs[i, :seq_len] = torch.tensor(dep_vocab.extwd2index(char_form), device=device)
        extbigram_idxs[i, :seq_len] = torch.tensor(dep_vocab.extwd2index(bi_form), device=device)

        tag_idxs[i, :seq_len] = torch.tensor(dep_vocab.tag2index([dep.tag for dep in deps]), device=device)

        head_idx[i, :seq_len] = torch.tensor([dep.head for dep in deps], device=device)
        rel_idx[i, :seq_len] = torch.tensor(dep_vocab.rel2index([dep.dep_rel for dep in deps]), device=device)
        non_pad_mask[i, :seq_len].fill_(1)

    ngram_idxs = unigram_idxs, bigram_idxs
    extngram_idxs = extunigram_idxs, extbigram_idxs

    return ngram_idxs, extngram_idxs, tag_idxs, head_idx, rel_idx, non_pad_mask


def get_seq_batch(tensors, max_len, step_len):
    for j in range(0, max_len-1, step_len):
        yield (t[j: j+step_len] for t in tensors)
