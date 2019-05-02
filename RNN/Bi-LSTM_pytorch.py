#!/usr/bin/env python
# encoding: utf-8
"""
@author: dstch
@license: (C) Copyright 2013-2019, Regulus Tech.
@contact: dstch@163.com
@file: Bi-LSTM_pytorch.py
@time: 2019/5/2 23:21
@desc: https://zhuanlan.zhihu.com/p/47802053
                                                                                                                                       
                                                                                                                                       
                                                            ...]`...                                                        ....]]..   
                                                            .=@@/@^*                                                        ..=@@/@^   
    .......,`...........                                ....=@@.                        .......*......,]]`..                .=@@       
    .....,]@@@@@@@@`....                                ...=@/..                        ..,]/@@@@@@@@@@/[[..                =@/.       
.....,/@@@@@^......@@...                                ..,@@...                    .../@@/[....=@^.                    ...,@@..       
.../@@@`..=@......=@@...                                ..@@....                    ..,@/.......@/..                    ...@@...       
..=@@*....@^...../@@    ..]]]].........,]................=@^.................,].    ...\@....../@...    ..]]]]............/@^.......   
...@\...,@@\,]*/@@^.    =@/.@^..../@@@@/@^..,@@`.../@@..,@/.../@/...,@@^.=@/.@@.    ....\^....=@^...    =@/.@^...@@`=@@..,@/../@@@^.   
    [...=@@@@`[[`......=@^..@^..,@@^..=@@`.,@@^.../@@.../@`../@/...,@@^..@@^.[`.            ..@@    ...=@^..@^..@@..=@^../@`,@/.=@^.   
    ...,@/.\@@\......*=@@@@@[..,@@`..,@@@..=@/.../@@`..,@@.../@`..,@@/...,@@....            .=@^    .*=@@@@@[..@@...*[..,@/@@`..@@*.   
    .=\@@...*\@@`.....@@....,@.@@`..=@\@^../@*.,@\@^*..=@^].,@^..//@@*....=@\...        . .,@@/.    ..@@....,@=@^......./@@/...=@\/.   
    .@@@......,@@@`..*@@.*,@/.=@^..@/*@@...@@.=@.@@@...@@/..=@^,@^=@@^....,@/...        ...=@@`.    .*@@.*,@/.@@^..,@^..@@/...,@@^..   
    ,@@`        ,@@@`*[@/@/....@@/@`./@^...@@@^..@@^..,@@...=@@@..=@/,@@@@@.            ...@@@..    ..[@/@/...,@@]@@`...\@....@@^...   
    ....        ..,@@@\/@..... .....,@/.......................*.......*.....            ........    ................................   
                    ,@@/        ....@@..                                                                                               
                    ....        .../@^..                                                                                               
                    .. .........../@^...                                                                                               
                    ...,].......,@@`....                                                                                               
                        ,@@].*]@@/..                                                                                                   
                        ...,[[[`...*                                                                                                   

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import numpy as np
import os
import torch.optim as optim

torch.manual_seed(123456)


def get_padding(sentences, max_len):
    """
    :param sentences: raw sentence --> index_padded sentence
                    [2, 3, 4], 5 --> [2, 3, 4, 0, 0]
    :param max_len: number of steps to unroll for a LSTM
    :return: sentence of max_len size with zero paddings
    """
    seq_len = np.zeros((0,))
    padded = np.zeros((0, max_len))
    for sentence in sentences:
        num_words = len(sentence)
        num_pad = max_len - num_words
        ''' Answer 60=45+15'''
        if max_len == 60 and num_words > 60:
            sentence = sentence[:45] + sentence[num_words - 15:]
            sentence = np.asarray(sentence, dtype=np.int64).reshape(1, -1)
        else:
            sentence = np.asarray(sentence[:max_len], dtype=np.int64).reshape(1, -1)
        if num_pad > 0:
            zero_paddings = np.zeros((1, num_pad), dtype=np.int64)
            sentence = np.concatenate((sentence, zero_paddings), axis=1)
        else:
            num_words = max_len

        padded = np.concatenate((padded, sentence), axis=0)
        seq_len = np.concatenate((seq_len, [num_words]))
    return padded.astype(np.int64), seq_len.astype(np.int64)


def get_mask_matrix(seq_lengths, max_len):
    """
    [5, 2, 4,... 7], 10 -->
            [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             ...,
             [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
            ]
    :param seq_lengths:
    :param max_len:
    :return:
    """
    mask_matrix = np.ones((0, max_len))
    for seq_len in seq_lengths:
        num_mask = max_len - seq_len
        mask = np.ones((1, seq_len), dtype=np.int64)
        if num_mask > 0:
            zero_paddings = np.zeros((1, num_mask), dtype=np.int64)
            mask = np.concatenate((mask, zero_paddings), axis=1)
        mask_matrix = np.concatenate((mask_matrix, mask), axis=0)

    return mask_matrix.astype(np.int64)


class YDataset(object):
    def __init__(self, features, labels, to_pad=True, max_len=40):
        """
        All sentences are indexes of words!
        :param features: list containing sequences to be padded and batched
        :param labels:
        """
        self.features = features
        self.labels = labels
        self.pad_max_len = max_len
        self.seq_lens = None
        self.mask_matrix = None

        assert len(features) == len(self.labels)

        self._num_examples = len(self.labels)
        self._epochs_completed = 0
        self._index_in_epoch = 0

        if to_pad:
            if max_len:
                self._padding()
                self._mask()
            else:
                print("Need more information about padding max_length")

    def __len__(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def _padding(self):
        self.features, self.seq_lens = get_padding(self.features, max_len=self.pad_max_len)

    def _mask(self):
        self.mask_matrix = get_mask_matrix(self.seq_lens, max_len=self.pad_max_len)

    def _shuffle(self, seed):
        """
        After each epoch, the data need to be shuffled
        :return:
        """
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)

        self.features = self.features[perm]
        self.seq_lens = self.seq_lens[perm]
        self.mask_matrix = self.mask_matrix[perm]
        self.labels = self.labels[perm]

    def next_batch(self, batch_size, seed=123456):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            '''  shuffle feature  and labels'''
            self._shuffle(seed=seed)

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        features = self.features[start:end]
        seq_lens = self.seq_lens[start:end]
        mask_matrix = self.mask_matrix[start:end]
        labels = self.labels[start:end]

        return features, seq_lens, mask_matrix, labels


class BiLSTM(nn.Module):
    """
        Implementation of BLSTM Concatenation for sentiment classification task
    """

    def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
        super(BiLSTM, self).__init__()

        self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
                                embedding_dim=embeddings.size(1),
                                padding_idx=0)
        self.emb.weight = nn.Parameter(embeddings)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # sen encoder
        self.sen_len = max_len
        self.sen_rnn = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        self.output = nn.Linear(2 * self.hidden_dim, output_dim)

    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

        # (batch_size, max_len, 1, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
        fw_out = fw_out.view(batch_size * max_len, -1)
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
        bw_out = bw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
        """
        :param sen_batch: (batch, sen_length), tensor for sentence sequence
        :param sen_lengths:
        :param sen_mask_matrix:
        :return:
        """

        ''' Embedding Layer | Padding | Sequence_length 40'''
        sen_batch = self.emb(sen_batch)

        batch_size = len(sen_batch)

        ''' Bi-LSTM Computation '''
        sen_outs, _ = self.sen_rnn(sen_batch.view(batch_size, -1, self.input_dim))
        sen_rnn = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, sen_len, 2*hid)

        ''' Fetch the truly last hidden layer of both sides
        '''
        sentence_batch = self.bi_fetch(sen_rnn, sen_lengths, batch_size, self.sen_len)  # (batch_size, 2*hid)

        representation = sentence_batch
        out = self.output(representation)
        out_prob = F.softmax(out.view(batch_size, -1))

        return out_prob


class LSTM(nn.Module):
    """
        Implementation of BLSTM Concatenation for Stance Classification Task
        Final representation is concatenation of last hidden layer of both sentence and ask blstm
    """

    def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
        super(LSTM, self).__init__()

        self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
                                embedding_dim=embeddings.size(1),
                                padding_idx=0)
        self.emb.weight = nn.Parameter(embeddings)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # sen encoder
        self.sen_len = max_len
        self.sen_rnn = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=False)

        self.output = nn.Linear(self.hidden_dim, output_dim)

    def _fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 1, -1)

        # (batch_size, max_len, 1, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
        fw_out = fw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        return fw_out

    def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
        """
        :param sen_batch: (batch, sen_length), tensor for sentence sequence
        :param sen_lengths:
        :param sen_mask_matrix:
        :return:
        """

        ''' Embedding Layer | Padding | Sequence_length 40'''
        sen_batch = self.emb(sen_batch)

        batch_size = len(sen_batch)

        ''' Bi-LSTM Computation '''
        sen_outs, _ = self.sen_rnn(sen_batch.view(batch_size, -1, self.input_dim))

        # Batch_first only change viewpoint, may not be contiguous
        sen_rnn = sen_outs.contiguous().view(batch_size, -1, self.hidden_dim)  # (batch, sen_len, 2*hid)

        ''' Fetch the truly last hidden layer of both sides
        '''
        sentence_batch = self._fetch(sen_rnn, sen_lengths, batch_size, self.sen_len)  # (batch_size, hid)

        representation = sentence_batch
        out = self.output(representation)
        out_prob = F.softmax(out.view(batch_size, -1))

        return out_prob


def train(model, training_data, args, optimizer, criterion):
    model.train()
    """
    在训练模型和测试模型时，会使用model.train()和model.eval()，因为这两个方法是针对在网络训练和测试时采用不同方式的情况，
    比如Batch Normalization 和 Dropout。
    """

    batch_size = args.batch_size

    sentences, sentences_seqlen, sentences_mask, labels = training_data

    # print batch_size, len(sentences), len(labels)

    assert batch_size == len(sentences) == len(labels)

    ''' Prepare data and prediction'''
    sentences_, sentences_seqlen_, sentences_mask_ = var_batch(args, batch_size, sentences, sentences_seqlen,
                                                               sentences_mask)
    labels_ = Variable(torch.LongTensor(labels))
    if args.cuda:
        labels_ = labels_.cuda()

    assert len(sentences) == len(labels)

    """
    代码中training_data是一个batch的数据，其中包括输入的句子sentences（句子中每个词以词下标表示），
    输入句子的长度sentences_seqlen，输入的句子对应的情感类别labels。 训练模型前，先清空遗留的梯度值，
    再根据该batch数据计算出来的梯度进行更新模型。
    """
    model.zero_grad()
    probs = model(sentences_, sentences_seqlen_, sentences_mask_)
    loss = criterion(probs.view(len(labels_), -1), labels_)

    loss.backward()
    optimizer.step()


def test_prf(pred, labels):
    """
    4. log and return prf scores
    :return:
    """
    total = len(labels)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    for i in range(total):
        pred_all[pred[i]] += 1
        if pred[i] == labels[i]:
            pred_right[pred[i]] += 1
        gold[labels[i]] += 1

    print("  Prediction:", pred_all, " Right:", pred_right, " Gold:", gold)
    ''' -- for all labels -- '''
    print("  ****** Neg|Neu|Pos ******")
    accuracy = 1.0 * sum(pred_right) / total
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                             formation=False,
                             metric_type="macro")
    print("    Accuracy on test is %d/%d = %f" % (sum(pred_right), total, accuracy))
    print("    Precision: %s\n    Recall   : %s\n    F1 score : %s\n    Macro F1 score on test (Neg|Neu|Pos) is %f" \
          % (p, r, f1, macro_f1))
    return accuracy


def var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask):
    """
    Transform the input batch to PyTorch variables
    :return:
    """
    # dtype = torch.from_numpy(sentences, dtype=torch.cuda.LongTensor)
    sentences_ = Variable(torch.LongTensor(sentences).view(batch_size, args.sen_max_len))
    sentences_seqlen_ = Variable(torch.LongTensor(sentences_seqlen).view(batch_size, 1))
    sentences_mask_ = Variable(torch.LongTensor(sentences_mask).view(batch_size, args.sen_max_len))

    if args.cuda:
        sentences_ = sentences_.cuda()
        sentences_seqlen_ = sentences_seqlen_.cuda()
        sentences_mask_ = sentences_mask_.cuda()

    return sentences_, sentences_seqlen_, sentences_mask_


def test(model, dataset, args, data_part="test"):
    """
    :param model:
    :param args:
    :param dataset:
    :param data_part:
    :return:
    """

    tvt_set = dataset[data_part]
    tvt_set = YDataset(tvt_set["xIndexes"],
                       tvt_set["yLabels"],
                       to_pad=True, max_len=args.sen_max_len)

    test_set = tvt_set
    sentences, sentences_seqlen, sentences_mask, labels = test_set.next_batch(len(test_set))

    assert len(test_set) == len(sentences) == len(labels)

    tic = time.time()

    model.eval()
    ''' Prepare data and prediction'''
    batch_size = len(sentences)
    sentences_, sentences_seqlen_, sentences_mask_ = \
        var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask)

    probs = model(sentences_, sentences_seqlen_, sentences_mask_)

    _, pred = torch.max(probs, dim=1)

    if args.cuda:
        pred = pred.view(-1).cpu().data.numpy()
    else:
        pred = pred.view(-1).data.numpy()

    tit = time.time() - tic
    print("  Predicting {:d} examples using {:5.4f} seconds".format(len(test_set), tit))

    labels = np.asarray(labels)
    ''' log and return prf scores '''
    accuracy = test_prf(pred, labels)

    return accuracy


def cal_prf(pred, right, gold, formation=True, metric_type=""):
    """
    :param pred: predicted labels
    :param right: predicting right labels
    :param gold: gold labels
    :param formation: whether format the float to 6 digits
    :param metric_type:
    :return: prf for each label
    """
    ''' Pred: [0, 2905, 0]  Right: [0, 2083, 0]  Gold: [370, 2083, 452] '''
    num_class = len(pred)
    precision = [0.0] * num_class
    recall = [0.0] * num_class
    f1_score = [0.0] * num_class

    for i in range(num_class):
        ''' cal precision for each class: right / predict '''
        precision[i] = 0 if pred[i] == 0 else 1.0 * right[i] / pred[i]

        ''' cal recall for each class: right / gold '''
        recall[i] = 0 if gold[i] == 0 else 1.0 * right[i] / gold[i]

        ''' cal recall for each class: 2 pr / (p+r) '''
        f1_score[i] = 0 if precision[i] == 0 or recall[i] == 0 \
            else 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        if formation:
            precision[i] = precision[i].__format__(".6f")
            recall[i] = recall[i].__format__(".6f")
            f1_score[i] = f1_score[i].__format__(".6f")

    ''' PRF for each label or PRF for all labels '''
    if metric_type == "macro":
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    elif metric_type == "micro":
        precision = 1.0 * sum(right) / sum(pred) if sum(pred) > 0 else 0
        recall = 1.0 * sum(right) / sum(gold) if sum(recall) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def pickle2dict(in_file):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(in_file, 'r') as f:
        your_dict = pickle.load(f)
    return your_dict


def main(args):
    # define location to save the model
    if args.save == "__":
        # LSTM_100_40_8
        args.save = "saved_model/%s_%d_%d_%d" % \
                    (args.model, args.nhid, args.sen_max_len, args.batch_size)

    in_dir = "data/mr/"
    dataset = pickle2dict(in_dir + "features_glove.pkl")

    if args.is_test:
        with open(args.save + "/model.pt") as f:
            model = torch.load(f)
        test(model, dataset, args)

    else:
        ''' make sure the folder to save models exist '''
        if not os.path.exists(args.save):
            os.mkdir(args.save)

        embeddings = pickle2dict(in_dir + "embeddings_glove.pkl")
        dataset["embeddings"] = embeddings
        emb_np = np.asarray(embeddings, dtype=np.float32)  # from_numpy
        emb = torch.from_numpy(emb_np)

        models = {"LSTM": LSTM, "BLSTM": BiLSTM}  # , "CNN": CNN}
        model = models[args.model](embeddings=emb,
                                   input_dim=args.embsize,
                                   hidden_dim=args.nhid,
                                   num_layers=args.nlayers,
                                   output_dim=2,
                                   max_len=args.sen_max_len,
                                   dropout=args.dropout)

        if torch.cuda.is_available():
            if not args.cuda:
                print("Waring: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(args.seed)
                model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        training_set = dataset["training"]
        training_set = YDataset(training_set["xIndexes"],
                                training_set["yLabels"],
                                to_pad=True,
                                max_len=args.sen_max_len)

        best_acc_test, best_acc_valid = -np.inf, -np.inf
        batches_per_epoch = int(len(training_set) / args.batch_size)
        print("--------------\nEpoch 0 begins!")
        max_train_steps = int(args.epochs * batches_per_epoch * 10)
        tic = time.time()
        print("-----------------------------", max_train_steps, len(training_set), args.batch_size)

        for step in range(max_train_steps):

            training_batch = training_set.next_batch(args.batch_size)

            train(model, training_batch, args, optimizer, criterion)

            if (step + 1) % batches_per_epoch == 0:
                print("  using %.5f seconds" % (time.time() - tic))
                tic = time.time()
                ''' Test after each epoch '''
                print("\n  Begin to predict the results on Validation")
                acc_score = test(model, dataset, args, data_part="validation")

                print("  ----Old best acc score on validation is %f" % best_acc_valid)
                if acc_score > best_acc_valid:
                    print("  ----New acc score on validation is %f" % acc_score)
                    best_acc_valid = acc_score
                    with open(args.save + "/model.pt", 'wb') as to_save:
                        torch.save(model, to_save)

                    acc_test = test(model, dataset, args)
                    print("  ----Old best acc score on test is %f" % best_acc_test)
                    if acc_test > best_acc_test:
                        best_acc_test = acc_test
                        print("  ----New acc score on test is %f" % acc_test)

                print("--------------\nEpoch %d begins!" % (training_set.epochs_completed + 1))

        # print the final result
        with open(args.save + "/model.pt") as f:
            model = torch.load(f)
        test(model, dataset, args)
