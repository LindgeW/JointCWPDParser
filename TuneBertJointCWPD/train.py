import torch
import torch.nn as nn
import numpy as np
import random
from conf.config import get_data_path, args_config
from datautil.dataloader import load_dataset
from vocab.dep_vocab import create_vocab
from modules.model import BaseModel, ParserModel
from modules.parser import BiaffineParser
from modules.BertModel import BertEmbedding


def set_seeds(seed=1349):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_seeds(1347)
    print('cuda available:', torch.cuda.is_available())
    print('cuDnn available:', torch.backends.cudnn.enabled)
    print('GPU numbers:', torch.cuda.device_count())

    data_path = get_data_path("./conf/datapath.json")
    dep_vocab = create_vocab(data_path['data']['train_data'],
                             data_path['pretrained']['bert_model'])

    train_data = load_dataset(data_path['data']['train_data'], dep_vocab)
    print('train data size:', len(train_data))
    dev_data = load_dataset(data_path['data']['dev_data'], dep_vocab)
    print('dev data size:', len(dev_data))
    test_data = load_dataset(data_path['data']['test_data'], dep_vocab)
    print('test data size:', len(test_data))

    args = args_config()
    args.tag_size = dep_vocab.tag_size
    args.rel_size = dep_vocab.rel_size

    bert_embed = BertEmbedding(data_path['pretrained']['bert_model'], nb_layers=args.encoder_layer, out_dim=args.d_model)
    base_model = BaseModel(args)
    parser_model = ParserModel(base_model, bert_embed)
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        # if torch.cuda.device_count() > 1:
        #     parser_model = nn.DataParallel(parser_model, device_ids=list(range(torch.cuda.device_count() // 2)))
    else:
        args.device = torch.device('cpu')

    bert_embed = bert_embed.to(args.device)
    base_model = base_model.to(args.device)
    parser_model = parser_model.to(args.device)
    biff_parser = BiaffineParser(parser_model)
    biff_parser.summary()
    print('模型参数量：', sum(p.numel() for p in parser_model.parameters() if p.requires_grad))
    biff_parser.train(train_data, dev_data, test_data, args, dep_vocab)

