import tensorflow as tf
from selfatt import SelfAttentive

class  LSTM_CRFModel():
    def __init__(self,word_embeddingmatrix,rnn_size,embedding_size,learning_rate,tar_to_idx,idx_to_tar,max_gradient_norm=5.0):
        self.learning_rate = learning_rate
        self.word_embeddingmatrix = word_embeddingmatrix
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.tar_to_idx = tar_to_idx
        self.tar_vocab_size = len(self.tar_to_idx)
        self.idx_to_tar = idx_to_tar
        self.max_gradient_norm = max_gradient_norm
        self.build_model()

    def _create_rnn_cell(self):

        single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell,output_keep_prob=self.keep_prob_placeholder)
        return single_cell

    def build_model(self):
        self.decoder_inputs_length = tf.placeholder(tf.int32, [None], name='decoder_inputs_length')
        self.word_inputs = tf.placeholder(tf.int32, [None, None, None], name="word_inputs")
        self.word_inputs_length = tf.placeholder(tf.int32, [None, None], name="word_inputs_length")
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        self.batch_size = tf.placeholder(tf.int32,[],name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32,name='keep_prob_placeholder')


        decoder_cell = self._create_rnn_cell()

        with tf.variable_scope("inputprocess"):

            input_wordembedding = tf.nn.embedding_lookup(self.word_embeddingmatrix,self.word_inputs)
            input_wordembedding = tf.cast(input_wordembedding,tf.float32)

            s = tf.shape(input_wordembedding)#(batch,s_n,s_l,dim)
            #(batch*s_n)*1
            word_inputs_length = tf.reshape(self.word_inputs_length,[-1,])
            #(batch*s_n)*s_l*200
            input_wordembedding = tf.reshape(input_wordembedding, [-1, s[-2], self.embedding_size])

            word_output,word_state = tf.nn.bidirectional_dynamic_rnn(decoder_cell,decoder_cell,input_wordembedding,
                                                                            sequence_length=word_inputs_length,
                                                                            dtype=tf.float32)
            word_output = tf.concat(word_output,2)
        with tf.variable_scope("attention_based_pooling"):
            selfattention = SelfAttentive()
            A = selfattention.build_graph(2 * self.rnn_size, 150, word_output, reuse=False)
            word_output = tf.multiply(A, word_output)

        word_embedding = tf.reduce_sum(word_output,1)
        decoder_inputs = tf.reshape(word_embedding,[-1,s[1],2*self.rnn_size])


        with tf.variable_scope("sentencedecoder"):
            cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob_placeholder)

            decoder_outputs,decoder_state = tf.nn.bidirectional_dynamic_rnn(cell,cell,decoder_inputs,
                                                                            sequence_length=self.decoder_inputs_length,
                                                                            dtype=tf.float32)


            decoder_outputs = tf.concat(decoder_outputs,2)


        with tf.variable_scope("dense"):
            # （batch_size * seq_length) * 2rnnsize
            decoder_outputs = tf.reshape(decoder_outputs,[-1,2*self.rnn_size])

            W = tf.get_variable("W_dense", dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     shape=[2 * self.rnn_size, self.tar_vocab_size])
            b = tf.get_variable("b_dense", shape=[self.tar_vocab_size],
                                     dtype=tf.float32, initializer=tf.zeros_initializer())

            self.output = tf.matmul(decoder_outputs,W)+b

            output = tf.reshape(self.output,[self.batch_size,-1,self.tar_vocab_size])

        with tf.variable_scope("loss"):

            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                output, self.decoder_targets,self.decoder_inputs_length)
            self.trans_params = trans_params
            loss = tf.reduce_mean(-log_likelihood)
            mask = tf.sequence_mask(self.decoder_inputs_length)
            self.loss = loss+  0.00001* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tf.trainable_variables() ])

            # output = tf.boolean_mask(output, mask)
            viterbi_seq, viterbi_score = tf.contrib.crf.crf_decode(output, trans_params,self.decoder_inputs_length)
            output = tf.boolean_mask(viterbi_seq,mask)
            label = tf.boolean_mask(self.decoder_targets, mask)
            correct_predictions = tf.equal(tf.cast(output, tf.int32), label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

            tf.summary.scalar("trainloss", self.loss)
            tf.summary.scalar("acc", self.accuracy)
            self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope=None))

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss,trainable_params)
            clip_gradients,_ = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients,trainable_params))

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self,sess,batch):
        feed_dict = {
                     self.decoder_inputs_length:batch.decoder_inputs_length,
                     self.word_inputs: batch.word_inputs,
                     self.word_inputs_length: batch.word_inputs_length,
                     self.decoder_targets:batch.decoder_targets,
                     self.decoder_targets_length:batch.decoder_targets_length,
                     self.keep_prob_placeholder:0.5,
                     self.batch_size:len(batch.decoder_inputs)

                     }


        _,loss,summary,acc = sess.run([self.train_op,self.loss,self.summary_op,self.accuracy],feed_dict=feed_dict)

        return acc,loss,summary

    def vali(self,sess,batch):

        feed_dict = {
                     self.decoder_inputs_length: batch.decoder_inputs_length,
                     self.word_inputs: batch.word_inputs,
                     self.word_inputs_length: batch.word_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob_placeholder: 0.5,
                     self.batch_size: len(batch.decoder_inputs)


                    }

        loss,summary,acc,trans = sess.run([self.loss,self.summary_op,self.accuracy,self.trans_params],feed_dict=feed_dict)

        return loss,summary,acc,trans
