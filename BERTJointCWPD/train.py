import torch
import numpy as np
from conf.config import get_data_path, args_config
from datautil.dataloader import load_dataset
from vocab.dep_vocab import create_vocab
from modules.model import ParserModel
from modules.parser import BiaffineParser
from modules.BertModel import BertEmbedding


if __name__ == '__main__':
    np.random.seed(3046)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1344)
    torch.cuda.manual_seed_all(1344)

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

    bert_embed = BertEmbedding(data_path['pretrained']['bert_model'], nb_layers=8, out_dim=args.d_model)
    parser_model = ParserModel(args, bert_embed)
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        # if torch.cuda.device_count() > 1:
        #     parser_model = nn.DataParallel(parser_model, device_ids=list(range(torch.cuda.device_count() // 2)))
    else:
        args.device = torch.device('cpu')

    bert_embed = bert_embed.to(args.device)
    parser_model = parser_model.to(args.device)
    biff_parser = BiaffineParser(parser_model)
    biff_parser.summary()
    print('模型参数量：', sum(p.numel() for p in parser_model.parameters() if p.requires_grad))
    biff_parser.train(train_data, dev_data, test_data, args, dep_vocab)

