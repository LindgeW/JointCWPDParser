
# CONLL标注格式包含10列，分别为：
# ID   FORM    LEMMA   CPOSTAG POSTAG  FEATS   HEAD    DEPREL  PHEAD   PDEPREL
# 只用到前８列，其含义分别为：
# 1    ID      当前词在句子中的序号，从1开始
# 2    FORM    当前字词或标点
# 3    LEMMA   当前词（或标点）的原型或词干，在中文中，此列与FORM相同
# 4    CPOSTAG 当前词的词性（粗粒度） coarse-grained
# 5    POSTAG  当前词的词性（细粒度）
# 6    FEATS   句法特征，在本次评测中，此列未被使用，全部以下划线代替。
# 7    HEAD    当前词的中心词 (语法父词索引，ROOT为0)
# 8    DEPREL  当前词与中心词的依存关系

# 创建词表时不需要vocab，读取语料时需要vocab
def read_deps(file_reader, vocab=None) -> list:
    min_count = 0
    if vocab is None:
        deps = []
    else:
        min_count = 1
        deps = [Dependency(0, vocab.root_form, vocab.root_rel, 0, vocab.root_rel)]

    for line in file_reader:
        try:
            tokens = line.strip().split('\t')
            if line.strip() == '' or len(tokens) < 10:
                if len(deps) > min_count:
                    yield deps

                if vocab is None:
                    deps = []
                else:
                    deps = [Dependency(0, vocab.root_form, vocab.root_rel, 0, vocab.root_rel)]
            elif len(tokens) == 10:
                if tokens[6] == '_':
                    tokens[6] = '-1'
                deps.append(Dependency(int(tokens[0]), tokens[1], tokens[3],
                                       int(tokens[6]), tokens[7]))
        except Exception as e:
            print('异常：', e)

    if len(deps) > min_count:
        yield deps


class Dependency(object):
    def __init__(self, sid: int,
                 form: str,
                 tag: str,
                 head: int,
                 dep_rel: str):
        self.id = sid       # 当前词ID
        self.form = form    # 当前词（或标点）
        self.tag = tag      # 词性
        self.head = head    # 当前词的head (ROOT为0)
        self.dep_rel = dep_rel  # 当前词与head之间的依存关系

    def __str__(self):
        return '\t'.join([str(self.id), self.form, self.tag, str(self.head), self.dep_rel])

    def __repr__(self):
        return str(self.__dict__)
