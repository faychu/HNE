import tensorflow as tf
import numpy as np
n_samples = 4
n_features = 5
dim = 3
X = tf.Variable([[1,0,0,0,1],[0,1,0.5,0,0.5],[1,0,0,0.5,0.5],[0,0,1,1,0]])
V = tf.Variable(tf.random_uniform([n_features,dim], -1.0, 1.0))
a = tf.matmul(X,V)
# a = tf.nn.embedding_lookup(V, [1,2])
b = tf.SparseTensor()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(X))
    print(sess.run(X*X))
    print(sess.run(V))
    print(sess.run(a))
    print(sess.run(b))