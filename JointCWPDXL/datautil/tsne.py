import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def load_wd2vec(path):
    assert os.path.exists(path)
    wds = []
    vecs = []
    i = 0
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split()
            if len(tokens) > 2:
                wds.append(tokens[0])
                vecs.append(np.array(tokens[1:], dtype=np.float32).tolist())
                i += 1
                if i > 1000:
                    break

    vecs = standardization(np.asarray(vecs))

    return wds, vecs


def tsne_reduce(wds, vecs):
    tsne_vecs = TSNE(n_components=2, init='pca').fit_transform(vecs)
    print(tsne_vecs.shape)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 15))
    plt.scatter(tsne_vecs[:, 0], tsne_vecs[:, 1])
    for i, wd in enumerate(wds):
        plt.text(tsne_vecs[i, 0] + 0.1, tsne_vecs[i, 1] + 0.2, s=wd, fontsize=8)
    plt.savefig('./vec2.png')
    plt.show()


if __name__ == '__main__':
    wds, vecs = load_wd2vec('../data/giga.100.txt')
    tsne_reduce(wds, vecs)
