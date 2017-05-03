import tensorflow as tf
import numpy as np
n_samples = 4
n_features = 5
v_embeddings = tf.Variable([[1,2,3,4],[5,6,7,8],[1,1,1,1]])
example =[0,4]
vocabulary_size = 10
embedding_size = 1
# var = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
var = np.array([1,2,3,4],[5,6,7,8],[1,1,1,1])
embeddings = tf.Variable(var, dtype=tf.float64)
c = tf.Variable([1,2,3],dtype=tf.float64)
# embed = tf.nn.embedding_lookup_sparse(embeddings, example, None)
a = tf.nn.embedding_lookup(embeddings, [1,2])
b = tf.nn.embedding_lookup(embeddings, [3,4])


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # print(sess.run(embed)) # prints [  9.  36.  81.]
    # print(sess.run(a*b))
    print(sess.run(tf.nn.l2_loss(c-1)))