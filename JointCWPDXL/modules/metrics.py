'''
PRF、UAS、LAS
UAS = R_udp = nb_arc / nb_arcs
LAS = R_ldp = nb_lbl / nb_arcs
'''


class Metrics(object):
    def __init__(self, nb_gold=None, nb_pred=None, nb_correct=None):
        self.nb_gold = nb_gold
        self.nb_pred = nb_pred
        self.nb_correct = nb_correct
        self.eps = 1e-12

    @property
    def precision(self):
        return 1. * self.nb_correct / (self.nb_pred + self.eps)

    @property
    def recall(self):
        return 1. * self.nb_correct / (self.nb_gold + self.eps)

    @property
    def F1(self):
        f1 = (2. * self.precision * self.recall) / (self.precision + self.recall + self.eps)
        return f1

    def get_metrics(self):
        result = {'precision': self.precision, 'recall': self.recall, 'f1': self.F1}
        return result

