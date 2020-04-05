import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def test1():
    tag = np.array([[0.09775623679161072, -0.9170374870300293, -0.12046116590499878, 0.4151001572608948, -0.4906068742275238, 1.0860775709152222, -1.9194791316986084]])
    dep = np.array([[0.5591347217559814, -2.310115337371826, -2.024622917175293, -1.4636127948760986, 0.6141133904457092, 0.8987528085708618, -0.6187851428985596]])
    tag_ = F.softmax(torch.from_numpy(tag), dim=1)
    dep_ = F.softmax(torch.from_numpy(dep), dim=1)

    ax1 = plt.subplot(211)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.imshow(tag_, cmap=plt.cm.hot_r)
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['left'].set_color('none')
    ax1.spines['bottom'].set_color('none')
    ax1.yaxis.set_ticklabels('none')
    ax1.set_xticklabels(['h0', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    ax1.set_yticks([])
    plt.xticks(fontsize=12)
    plt.title('CWS & POS-tagging', fontdict={'family': 'Microsoft YaHei', 'size': 13}, loc='center')
    ax2 = plt.subplot(212)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.imshow(dep_, cmap=plt.cm.hot_r, aspect=1)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['left'].set_color('none')
    ax2.spines['bottom'].set_color('none')
    ax2.yaxis.set_ticklabels('none')
    ax2.set_xticklabels(['h0', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    ax2.set_yticks([])
    plt.xticks(fontsize=12)
    plt.title('Dependency parsing', fontdict={'family': 'Microsoft YaHei', 'size': 13}, loc='center')
    plt.colorbar()
    # plt.savefig('./map.svg')
    plt.show()


def test2():
    tune_tag = np.array([[-0.3287504017353058, -0.43384358286857605, -0.6369794607162476, -0.7293740510940552, -0.045044273138046265, 0.5225287079811096, 0.6732711791992188]])
    tune_dep = np.array([[-0.33941972255706787, -0.24522434175014496, 0.1323101967573166, 0.3380222022533417, 0.2715478837490082, 0.05004581809043884, -0.10043632239103317]])

    tag_ = F.softmax(torch.from_numpy(tune_tag), dim=1)
    dep_ = F.softmax(torch.from_numpy(tune_dep), dim=1)

    fig, axes = plt.subplots(2, 1)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    im = axes[0].imshow(tag_, cmap=plt.cm.hot_r)
    axes[0].yaxis.set_ticklabels('none')
    axes[0].set_xticklabels(['${L_0}$', '${L_0}$', '$L_1$', '$L_2$', '$L_3$', '$L_4$', '$L_5$', '$L_6$'], fontsize=13)
    axes[0].set_yticks([])
    axes[0].set_title('CWS & POS-tagging', fontdict={'family': 'Microsoft YaHei', 'size': 14}, loc='center')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    axes[1].imshow(dep_, cmap=plt.cm.hot_r)
    fig.subplots_adjust(wspace=0, hspace=0)
    axes[1].yaxis.set_ticklabels('none')
    axes[1].set_xticklabels(['${L_0}$', '${L_0}$', '$L_1$', '$L_2$', '$L_3$', '$L_4$', '$L_5$', '$L_6$'], fontsize=13)
    axes[1].set_yticks([])
    axes[1].set_title('Dependency parsing', fontdict={'family': 'Microsoft YaHei', 'size': 14}, loc='center')
    fig.colorbar(im, ax=axes.ravel().tolist())
    # plt.savefig('./tune_map.svg')
    plt.show()


def test_display_attention():
    i = 0
    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    for l in range(6):
        for h in range(8):
            att_weights = np.loadtxt(f'../xl_layer_ctb7/xl_layer{l}_head{h}.out')
            ax = plt.subplot(6, 8, i+1)
            plt.imshow(att_weights, cmap=plt.cm.hot_r)
            # plt.xticks(range(13))
            # plt.yticks(range(13))
            plt.xticks([])
            plt.yticks([])
            # cax=ax.matshow(att_weights, cmap='hot_r')
            ax.tick_params(labelsize=6)
            # ax.set_xticklabels(['', 'root', '上', '海','浦','东','开','发','与','法','制','建','设','同','步'])
            # ax.set_yticklabels(['', 'I', 'love', 'you', '!'])
            # fig.colorbar(cax)
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            i += 1
            if h == 0:
                plt.ylabel(f'Layer{l}')
            if l == 0:
                plt.title(f'Head{h}')
                # plt.xlabel('Head'+str(h))

    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.savefig('./trans_layer.svg')
    plt.show()
    plt.close()
