import tensorflow as tf
from data_util_NICTAword import getBatches,loadwordvalidataset,loadmatrix,loadwordmatrix,loadwordtraindataset,get_sentence_int_to_vocab
from LSTM_CRF import LSTM_CRFModel
#进度条显示
from tqdm import tqdm
import math
import time
import os

start = time.clock()

tf.app.flags.DEFINE_integer("rnn_size",200,"Number of hidden units in each layer")
tf.app.flags.DEFINE_integer("batch_size",40,"Batch Size")
tf.app.flags.DEFINE_integer("embedding_size",200,"Embedding dimensions of encoder and decoder inputs")
tf.app.flags.DEFINE_float("learning_rate",0.001,"Learning rate")
tf.app.flags.DEFINE_integer("num_layers",1,"Number of layers in each encoder and decoder")
tf.app.flags.DEFINE_integer("numEpochs",30,"Maximum # of training epochs")



FLAGS = tf.app.flags.FLAGS


sentence_int_to_vocab,sentence_vocab_to_int,tags_vocab_to_int,tags_int_to_vocab = get_sentence_int_to_vocab()
train_data = loadwordtraindataset()
vali_data = loadwordvalidataset()
word_embeddingmatrix = loadwordmatrix()

with tf.Session() as sess:


    model = LSTM_CRFModel(word_embeddingmatrix,FLAGS.rnn_size,FLAGS.embedding_size,
                         FLAGS.learning_rate, tags_vocab_to_int, tags_int_to_vocab,
                         max_gradient_norm=5.0)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    current_step = 0

    train_summary_writer = tf.summary.FileWriter("train",graph=sess.graph)

    for e in range(FLAGS.numEpochs):

        print("Epoch{}/{}-------------".format(e+1,FLAGS.numEpochs))
        train_batches = getBatches(train_data,FLAGS.batch_size)

        vali_batches = getBatches(vali_data,40)
        for train_Batch in tqdm(train_batches,desc = "Training"):
            train_acc,trainloss,trainsummary,train_f1 = model.train(sess,train_Batch)
            current_step += 1

            trainperplexity = math.exp(float(trainloss)) if trainloss<300 else float("inf")
            tqdm.write("----Step %d -- trainloss %.2f -- trainacc %.2f --trainf1 %.2f" %(current_step,trainloss,train_acc,train_f1))

            train_summary_writer.add_summary(trainsummary,current_step)






