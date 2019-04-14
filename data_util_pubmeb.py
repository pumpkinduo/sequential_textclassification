from __future__ import absolute_import
from __future__ import division
#把下一个新版本的特性导入到当前版本
from __future__ import print_function
from gensim.models import Doc2Vec

import numpy as np

import json

padToken,goToken,eosToken,unknownToken = 0,1,2,3

class Batch:
    def __init__(self):
        self.decoder_inputs = []
        self.decoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []
        self.word_inputs = []
        self.word_inputs_length = []



def get_sentence_int_to_vocab():


    tags_vocab = ['background', 'objective', 'methods', 'results', 'conclusions']

    tags_int_to_vocab = {}

    for index_no, word in enumerate(tags_vocab):
        tags_int_to_vocab[index_no] = word
    tags_vocab_to_int = {word: index_no for index_no, word in tags_int_to_vocab.items()}

    return tags_vocab_to_int,tags_int_to_vocab

def loadTrainDataset():
    tags_vocab_to_int, tags_int_to_vocab = get_sentence_int_to_vocab()
    f = open("PUBMEB_dataset/pubtrain.txt", "r")

    f = f.readlines()

    abstract = "".join(f).split("\n\n")

    i = 1
    dataset = []
    for abs in abstract:
        data = []
        targets = []
        sentences = []
        for sen in abs.split("\n")[1:]:
            if sen.split("\t")[0] != "":
                targets.append(tags_vocab_to_int[sen.split("\t")[0].lower()])

                sentences.append(i)
                i+=1
        data.append(sentences)
        data.append(targets)
        dataset.append(data)

    return tags_vocab_to_int,tags_int_to_vocab,dataset,i

def loadValiDataset():
    tags_vocab_to_int, tags_int_to_vocab, dataset, i = loadTrainDataset()
    f = open("PUBMEB_dataset/pubvali.txt", "r")

    f = f.readlines()

    abstract = "".join(f).split("\n\n")

    dataset = []

    for abs in abstract:

        data = []
        targets = []
        sentences = []
        for sen in abs.split("\n")[1:]:
            if sen.split("\t")[0] != "":
                targets.append(tags_vocab_to_int[sen.split("\t")[0].lower()])

            sentences.append(i)
            i += 1
        data.append(sentences)
        data.append(targets)
        dataset.append(data)

    return dataset,i


def loadwordmatrix():
    #train和vali的word embedding
    f = open("PUBMEB_dataset/pubmeb_wordmatrix.json")
    sentenceembedding = []
    for line in f.readlines():
        embed = json.loads(line)
        sentenceembedding.append(np.array(embed))
    sentenceembedding = np.reshape(sentenceembedding, [-1, 200])

    return sentenceembedding
def loadtestwordmatrix():
    #测试集的word embedding
    f = open("PUBMEB_dataset/pubmebtest_wordmatrix.json")
    sentenceembedding = []
    for line in f.readlines():
        embed = json.loads(line)
        sentenceembedding.append(np.array(embed))
    sentenceembedding = np.reshape(sentenceembedding, [-1, 200])
    return sentenceembedding
def loadwordtraindataset():

    _, _, _, _, dataset, _= loadTrainDataset()

    f = open("PUBMEB_dataset/pubmebtrain_wordindex.json")
    f = f.readlines()
    for i,line in enumerate(f):

        line = json.loads(line)
        for j,sentence in enumerate(line):

            sentence_ = []
            sentence_.append(dataset[i][0][j])
            sentence_.append(sentence)
            dataset[i][0][j]=sentence_


    return dataset
def loadwordvalidataset():
    dataset, _ = loadValiDataset()

    f = open("PUBMEB_dataset/pubmebvali_wordindex.json")
    f = f.readlines()
    for i, line in enumerate(f):

        line = json.loads(line)
        for j, sentence in enumerate(line):
            sentence_ = []
            sentence_.append(dataset[i][0][j])
            sentence_.append(sentence)
            dataset[i][0][j] = sentence_

    return dataset


def createBatch(samples):
    '''
    :param samples: 一个batch的数据
    :return: 可直接传入feeddict的一个batch的数据格式
    '''
    # print("samples:",samples[1])
    batch = Batch()
    batch.decoder_inputs_length = [len(sample[0]) for sample in samples]
    batch.decoder_targets_length = [len(sample[1]) for sample in samples]
    batch.word_inputs_length = [[len(sentence[1]) for sentence in sample[0] ] for sample in samples]


    max_source_length = max(batch.decoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)
    max_word_inputs = max([max(sentence) for sentence in batch.word_inputs_length])

    a = []
    a.append(0)
    for j,sample in enumerate(samples):

        source= sample[0]
        batch.decoder_inputs.append([sentence[0] for sentence in source]+[0]*(max_source_length - len(source)))

        target = sample[1]

        batch.decoder_targets.append(target+[0]*(max_target_length - len(target)))

        batch.word_inputs.append([sentence[1] +[0]*(max_word_inputs-len(sentence[1])) for sentence in source])
        for i in range((max_source_length - len(source))):
            batch.word_inputs[j].append(a+[0]*(len(batch.word_inputs[j][0])-1))
            batch.word_inputs_length[j].append(0)

    return batch

def getBatches(data,batch_size):

    batches = []
    data_len = len(data)

    def genNextSamples():
        for i in range(0,data_len,batch_size):
            yield data[i:min(i+batch_size,data_len)]

    for sample in genNextSamples():
        batch = createBatch(sample)
        batches.append(batch)
    return batches





