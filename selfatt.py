import tensorflow as tf


class SelfAttentive():

  def build_graph(self, u, d_a, H,reuse=True):
    with tf.variable_scope('SelfAttentive', reuse=reuse):


      initializer = tf.contrib.layers.xavier_initializer()



      W_s1 = tf.get_variable('W_s1', shape=[d_a, u],
          initializer=initializer)

      W_s2 = tf.get_variable('W_s2', shape=[1, d_a],
          initializer=initializer)


      A = tf.nn.softmax(
          tf.map_fn(
            lambda x: tf.matmul(W_s2, x),
            tf.tanh(
              tf.map_fn(
                lambda x: tf.matmul(W_s1, tf.transpose(x)),
                H))))
      A = tf.squeeze(A)  # batch*length

      A = tf.expand_dims(A, axis=2)  # batch*length*1

      return A