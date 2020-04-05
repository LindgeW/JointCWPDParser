import torch
import torch.nn as nn
import numpy as np
from conf.config import get_data_path, args_config
from datautil.dataloader import load_dataset
from vocab.dep_vocab import create_vocab
from modules.model import ParserModel
from modules.parser import JointDParser


if __name__ == '__main__':
    np.random.seed(3046)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1344)
    torch.cuda.manual_seed_all(1344)

    print('cuda available:', torch.cuda.is_available())
    print('cuDnn available:', torch.backends.cudnn.enabled)
    print('GPU numbers:', torch.cuda.device_count())

    data_path = get_data_path("./conf/datapath.json")
    char_vocab, bichar_vocab = create_vocab(data_path['data']['train_data'])
    char_embed_weights = char_vocab.get_embedding_weights(data_path['pretrained']['char_embedding'])
    bichar_embed_weights = bichar_vocab.get_embedding_weights(data_path['pretrained']['bichar_embedding'])

    train_data = load_dataset(data_path['data']['train_data'], char_vocab)
    print('train data size:', len(train_data))
    dev_data = load_dataset(data_path['data']['dev_data'], char_vocab)
    print('dev data size:', len(dev_data))
    test_data = load_dataset(data_path['data']['test_data'], char_vocab)
    print('test data size:', len(test_data))

    args = args_config()
    args.char_vocab_size = char_vocab.vocab_size
    args.bichar_vocab_size = bichar_vocab.vocab_size
    args.tag_size = char_vocab.tag_size
    args.rel_size = char_vocab.rel_size

    parser_model = ParserModel(args, char_embed_weights, bichar_embed_weights)
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        # if torch.cuda.device_count() > 1:
        #     parser_model = nn.DataParallel(parser_model, device_ids=list(range(torch.cuda.device_count() // 2)))
    else:
        args.device = torch.device('cpu')

    parser_model = parser_model.to(args.device)
    print('模型参数量：', sum(p.numel() for p in parser_model.parameters()))
    print('模型参数量：', sum(p.nelement() for p in parser_model.parameters()))

    biff_parser = JointDParser(parser_model)
    biff_parser.summary()

    biff_parser.train(train_data, dev_data, test_data, args, char_vocab, bichar_vocab)

