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
