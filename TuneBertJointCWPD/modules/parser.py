import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
# from .optimizer import WarmupLinearSchedule
from ..datautil.char_utils import calc_prf, cws_from_tag, calc_seg_f1, pos_tag_f1, parser_metric
from ..datautil.dataloader import batch_iter, batch_variable
from ..datautil.dependency import Dependency
from ..log.logger_ import logger
from ..modules.decode_alg.eisner import eisner



class BiaffineParser(object):
    def __init__(self, parser_model):
        super(BiaffineParser, self).__init__()
        assert isinstance(parser_model, nn.Module)
        self.parser_model = parser_model

    def summary(self):
        logger.info(self.parser_model)

    def train(self, train_data, dev_data, test_data, args, vocab):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.parser_model.bert.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.01, 'lr': args.bert_lr},
            {'params': [p for n, p in self.parser_model.bert.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0, 'lr': args.bert_lr},
            {'params': [p for n, p in self.parser_model.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.base_lr},
            {'params': [p for n, p in self.parser_model.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.base_lr}
        ]
        args.max_step = args.epoch * ((len(train_data) + args.batch_size - 1) // (args.batch_size * args.update_steps))
        # args.warmup_step = args.max_step // 10
        print('max step:', args.max_step)
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.bert_lr, eps=args.eps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.max_step)
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=args.max_step)

        test_best_uas, test_best_las = 0, 0
        test_best_tag_f1, test_best_seg_f1, test_best_udep_f1, test_best_ldep_f1 = 0, 0, 0, 0
        for ep in range(1, 1 + args.epoch):
            self.parser_model.train()
            train_loss = 0
            all_arc_acc, all_rel_acc, all_arcs = 0, 0, 0
            start_time = time.time()
            for i, batch_data in enumerate(batch_iter(train_data, args.batch_size, True)):
                batcher = batch_variable(batch_data, vocab, args.device)
                # batcher = (x.to(args.device) for x in batcher)
                (bert_ids, bert_lens, bert_mask), true_tags, true_heads, true_rels = batcher
                tag_score, arc_score, rel_score = self.parser_model(bert_ids, bert_lens, bert_mask)
                tag_loss = self.calc_tag_loss(tag_score, true_tags, bert_lens.gt(0))
                dep_loss = self.calc_dep_loss(arc_score, rel_score, true_heads, true_rels, bert_lens.gt(0))
                loss = tag_loss + dep_loss
                if args.update_steps > 1:
                    loss = loss / args.update_steps
                loss_val = loss.data.item()
                train_loss += loss_val
                loss.backward()

                arc_acc, rel_acc, nb_arcs = self.calc_acc(arc_score, rel_score, true_heads, true_rels, bert_lens.gt(0))
                all_arc_acc += arc_acc
                all_rel_acc += rel_acc
                all_arcs += nb_arcs
                ARC = all_arc_acc * 100. / all_arcs
                REL = all_rel_acc * 100. / all_arcs

                if (i + 1) % args.update_steps == 0 or (i == args.max_step - 1):
                    nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parser_model.parameters()),
                                             max_norm=args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    self.parser_model.zero_grad()

                logger.info('Iter%d ARC: %.2f%%, REL: %.2f%%' % (i + 1, ARC, REL))
                logger.info('time cost: %.2fs, train loss: %.2f' % ((time.time() - start_time), loss_val))

            train_loss /= len(train_data)
            arc = all_arc_acc * 100. / all_arcs
            rel = all_rel_acc * 100. / all_arcs

            dev_uas, dev_las, tag_f1, seg_f1, udep_f1, ldep_f1 = self.evaluate(dev_data, args, vocab)
            logger.info('[Epoch %d] train loss: %.3f, ARC: %.2f%%, REL: %.2f%%' % (ep, train_loss, arc, rel))
            logger.info('Dev data -- UAS: %.2f%%, LAS: %.2f%%' % (100. * dev_uas, 100. * dev_las))
            logger.info('Dev data -- TAG: %.2f%%, Seg F1: %.2f%%, UDEP F1: %.2f%%, LDEP F1: %.2f%%' % (
            100. * tag_f1, 100. * seg_f1, 100. * udep_f1, 100. * ldep_f1))

            test_uas, test_las, test_tag_f1, test_seg_f1, test_udep_f1, test_ldep_f1 = self.evaluate(test_data, args,
                                                                                                     vocab)
            if test_best_uas < test_uas:
                test_best_uas = test_uas
            if test_best_las < test_las:
                test_best_las = test_las
            if test_best_tag_f1 < test_tag_f1:
                test_best_tag_f1 = test_tag_f1
            if test_best_seg_f1 < test_seg_f1:
                test_best_seg_f1 = test_seg_f1
            if test_best_udep_f1 < test_udep_f1:
                test_best_udep_f1 = test_udep_f1
            if test_best_ldep_f1 < test_ldep_f1:
                test_best_ldep_f1 = test_ldep_f1
            logger.info('Test data -- UAS: %.2f%%, LAS: %.2f%%' % (100. * test_uas, 100. * test_las))
            logger.info('Test data -- Tag F1: %.2f%%, Seg F1: %.2f%%, UDEP F1: %.2f%%, LDEP F1: %.2f%%' % (
            100. * test_tag_f1, 100. * test_seg_f1, 100. * test_udep_f1, 100. * test_ldep_f1))

        logger.info('Final test performance -- UAS: %.2f%%, LAS: %.2f%%' % (100. * test_best_uas, 100. * test_best_las))
        logger.info('Final test performance -- Tag F1: %.2f%%, Seg F1: %.2f%%, UDEP F1: %.2f%%, LDEP F1: %.2f%%' % (
        100. * test_best_tag_f1, 100. * test_best_seg_f1, 100. * test_best_udep_f1, 100. * test_best_ldep_f1))

    def evaluate(self, test_data, args, vocab):
        self.parser_model.eval()
        all_gold_seg, all_pred_seg, all_seg_correct = 0, 0, 0
        all_gold_tag, all_pred_tag, all_tag_correct = 0, 0, 0
        all_gold_arc, all_pred_arc, all_arc_correct, all_rel_correct = 0, 0, 0, 0
        with torch.no_grad():
            for batch_data in batch_iter(test_data, args.test_batch_size):
                batcher = batch_variable(batch_data, vocab, args.device)
                # batcher = (x.to(args.device) for x in batcher)
                (bert_ids, bert_lens, bert_mask), true_tags, true_heads, true_rels = batcher
                tag_score, arc_score, rel_score = self.parser_model(bert_ids, bert_lens, bert_mask)
                pred_tags = tag_score.data.argmax(dim=-1)
                pred_heads, pred_rels = self.decode(arc_score, rel_score, bert_lens.gt(0))
                for i, sent_dep_tree in enumerate(
                        self.dep_tree_iter(batch_data, pred_tags, pred_heads, pred_rels, vocab)):
                    gold_seg_lst = cws_from_tag(batch_data[i])
                    pred_seg_lst = cws_from_tag(sent_dep_tree)

                    num_gold_seg, num_pred_seg, num_seg_correct = calc_seg_f1(gold_seg_lst, pred_seg_lst)
                    all_gold_seg += num_gold_seg
                    all_pred_seg += num_pred_seg
                    all_seg_correct += num_seg_correct

                    num_gold_tag, num_pred_tag, num_tag_correct = pos_tag_f1(gold_seg_lst, pred_seg_lst)
                    all_gold_tag += num_gold_tag
                    all_pred_tag += num_pred_tag
                    all_tag_correct += num_tag_correct

                    num_gold_arc, num_pred_arc, num_arc_correct, num_rel_correct = parser_metric(gold_seg_lst,
                                                                                                 pred_seg_lst)
                    all_arc_correct += num_arc_correct
                    all_rel_correct += num_rel_correct
                    all_gold_arc += num_gold_arc
                    all_pred_arc += num_pred_arc

        seg_f1 = calc_prf(all_gold_seg, all_pred_seg, all_seg_correct)[2]
        tag_f1 = calc_prf(all_gold_tag, all_pred_tag, all_tag_correct)[2]
        udep_metric = calc_prf(all_gold_arc, all_pred_arc, all_arc_correct)
        ldep_metric = calc_prf(all_gold_arc, all_pred_arc, all_rel_correct)
        udep_f1 = udep_metric[2]
        uas = udep_metric[1]
        ldep_f1 = ldep_metric[2]
        las = ldep_metric[1]
        return uas, las, tag_f1, seg_f1, udep_f1, ldep_f1

    def dep_tree_iter(self, batch_gold_trees, pred_tags, pred_heads, pred_rels, vocab):
        if torch.is_tensor(pred_tags):
            pred_tags = pred_tags.data.cpu().numpy()
        if torch.is_tensor(pred_heads):
            pred_heads = pred_heads.data.cpu().numpy()
        if torch.is_tensor(pred_rels):
            pred_rels = pred_rels.data.cpu().numpy()

        for sent_tree, tags, heads, rels in zip(batch_gold_trees, pred_tags, pred_heads, pred_rels):
            sent_dep_tree = []
            for idx in range(len(sent_tree)):
                sent_dep_tree.append(
                    Dependency(sent_tree[idx].id, sent_tree[idx].form, vocab.index2tag(tags[idx]), heads[idx],
                               vocab.index2rel(rels[idx])))
            yield sent_dep_tree

    def decode(self, pred_arc_score, pred_rel_score, mask):
        '''
        :param pred_arc_score: (bz, seq_len, seq_len)
        :param pred_rel_score: (bz, seq_len, seq_len, rel_size)
        :param mask: (bz, seq_len)  pad部分为0
        :return: pred_heads (bz, seq_len)
                 pred_rels (bz, seq_len)
        '''
        bz, seq_len, _ = pred_arc_score.size()
        # pred_heads = mst_decode(pred_arc_score, mask)
        mask[:, 0] = 0  # mask out <root>
        pred_heads = eisner(pred_arc_score, mask)
        pred_rels = pred_rel_score.data.argmax(dim=-1)
        # pred_rels = pred_rels.gather(dim=-1, index=pred_heads.unsqueeze(-1)).squeeze(-1)
        pred_rels = pred_rels[torch.arange(bz, dtype=torch.long, device=pred_arc_score.device).unsqueeze(1),
                              torch.arange(seq_len, dtype=torch.long, device=pred_arc_score.device).unsqueeze(0),
                              pred_heads].contiguous()
        return pred_heads, pred_rels

    def calc_tag_loss(self, pred_tags, true_tags, non_pad_mask):
        masked_true_tags = true_tags.masked_fill(~non_pad_mask, -1)
        tag_loss = F.cross_entropy(pred_tags.transpose(1, 2), masked_true_tags, ignore_index=-1)
        return tag_loss

    def calc_dep_loss(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask):
        '''
        :param pred_arcs: (bz, seq_len, seq_len)
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_heads: (bz, seq_len)  包含padding
        :param true_rels: (bz, seq_len)
        :param non_pad_mask: (bz, seq_len) 有效部分mask
        :return:
        '''
        # non_pad_mask[:, 0] = 0  # mask out <root>
        pad_mask = (non_pad_mask == 0)

        bz, seq_len, _ = pred_arcs.size()
        masked_true_heads = true_heads.masked_fill(pad_mask, -1)
        arc_loss = F.cross_entropy(pred_arcs.transpose(1, 2), masked_true_heads, ignore_index=-1)

        bz, seq_len, seq_len, rel_size = pred_rels.size()

        out_rels = pred_rels[torch.arange(bz, device=pred_arcs.device, dtype=torch.long).unsqueeze(1),
                             torch.arange(seq_len, device=pred_arcs.device, dtype=torch.long).unsqueeze(0),
                             true_heads].contiguous()

        masked_true_rels = true_rels.masked_fill(pad_mask, -1)
        # (bz*seq_len, rel_size)  (bz*seq_len, )
        rel_loss = F.cross_entropy(out_rels.transpose(1, 2), masked_true_rels, ignore_index=-1)
        return arc_loss + rel_loss

    def calc_acc(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask=None):
        '''a
        :param pred_arcs: (bz, seq_len, seq_len)
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_heads: (bz, seq_len)  包含padding
        :param true_rels: (bz, seq_len)
        :param non_pad_mask: (bz, seq_len) 非填充部分mask
        :return:
        '''
        # non_pad_mask[:, 0] = 0  # mask out <root>
        bz, seq_len, seq_len, rel_size = pred_rels.size()

        # (bz, seq_len)
        pred_heads = pred_arcs.data.argmax(dim=2)
        arc_acc = ((pred_heads == true_heads) * non_pad_mask).sum().item()

        total_arcs = non_pad_mask.sum().item()

        out_rels = pred_rels[torch.arange(bz, device=pred_arcs.device, dtype=torch.long).unsqueeze(1),
                             torch.arange(seq_len, device=pred_arcs.device, dtype=torch.long).unsqueeze(0),
                             true_heads].contiguous()
        pred_rels = out_rels.data.argmax(dim=2)
        rel_acc = ((pred_rels == true_rels) * non_pad_mask).sum().item()

        return arc_acc, rel_acc, total_arcs

    # def calc_loss(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask):
    #     '''
    #     :param pred_arcs: (bz, seq_len, seq_len)
    #     :param pred_rels:  (bz, seq_len, seq_len, rel_size)
    #     :param true_heads: (bz, seq_len)  包含padding
    #     :param true_rels: (bz, seq_len)
    #     :param non_pad_mask: (bz, seq_len) 有效部分mask
    #     :return:
    #     '''
    #     non_pad_mask = non_pad_mask.byte()
    #     non_pad_mask[:, 0] = 0
    #
    #     pred_heads = pred_arcs[non_pad_mask]  # (bz, seq_len)
    #     true_heads = true_heads[non_pad_mask]   # (bz, )
    #     pred_rels = pred_rels[non_pad_mask]  # (bz, seq_len, rel_size)
    #     pred_rels = pred_rels[torch.arange(len(pred_rels), dtype=torch.long), true_heads]  # (bz, rel_size)
    #     true_rels = true_rels[non_pad_mask]     # (bz, )
    #
    #     arc_loss = F.cross_entropy(pred_heads, true_heads)
    #     rel_loss = F.cross_entropy(pred_rels, true_rels)
    #
    #     return arc_loss + rel_loss

    # def calc_acc(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask=None):
    #     '''
    #     :param pred_arcs: (bz, seq_len, seq_len)
    #     :param pred_rels:  (bz, seq_len, seq_len, rel_size)
    #     :param true_heads: (bz, seq_len)  包含padding
    #     :param true_rels: (bz, seq_len)
    #     :param non_pad_mask: (bz, seq_len)
    #     :return:
    #     '''
    #     non_pad_mask = non_pad_mask.byte()
    #
    #     pred_heads = pred_arcs[non_pad_mask]  # (bz, seq_len)
    #     true_heads = true_heads[non_pad_mask]  # (bz, )
    #     pred_heads = pred_heads.data.argmax(dim=-1)
    #     arc_acc = true_heads.eq(pred_heads).sum().item()
    #     total_arcs = non_pad_mask.sum().item()
    #
    #     pred_rels = pred_rels[non_pad_mask]  # (bz, seq_len, rel_size)
    #     pred_rels = pred_rels[torch.arange(len(pred_rels)), true_heads]  # (bz, rel_size)
    #     pred_rels = pred_rels.data.argmax(dim=-1)
    #     true_rels = true_rels[non_pad_mask]  # (bz, )
    #     rel_acc = true_rels.eq(pred_rels).sum().item()
    #
    #     return arc_acc, rel_acc, total_arcs
