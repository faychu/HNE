import tensorflow as tf
import numpy as np
n_samples = 4
n_features = 5
dim = 3
X = tf.Variable([[1,0,0,0,1],[0,1,0.5,0,0.5],[1,0,0,0.5,0.5],[0,0,1,1,0]])
Xsp = tf.SparseTensor(values=[1,1,1,0.5,0.5,1,0.5,0.5,1,1],
                      indices=[[0,0],[0,4],[1,1],[1,2],[1,4],[2,0],[2,3],[2,4],[3,2],[3,3]],
                      dense_shape=[4,5])
a = [Xsp,Xsp]
# e = tf.slice(Xsp,[1,0],[2,5])
print(Xsp.indices.shape)
print(Xsp.dense_shape.shape==[2])
# d1 = tf.nn.embedding_lookup_sparse(Xsp,[1,2],[1])
V = tf.Variable(tf.random_uniform([n_features,dim], -1.0, 1.0))
# a = tf.matmul(X,V)
d = tf.nn.embedding_lookup(X, 1)
d2 = tf.nn.embedding_lookup(X, 2)

b = tf.SparseTensor(values=[1, 2], indices=[[0, 0], [1, 2]], dense_shape=[3, 4])

c=tf.sparse_to_dense([[0, 0], [1, 2]],[3,4],[1,2])
c1 = tf.sparse_tensor_to_dense(b,10)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(tf.nn.l2_loss(d-d2))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(X))
    print(sess.run(X+1))
    print(sess.run(Xsp))
    print(sess.run(X*X))
    print(sess.run(V))
    print(sess.run(a))
    # print(sess.run(b))
    # print(sess.run(c))
    # print(sess.run(c1))
    # print(sess.run(b.indices))
    # print(sess.run(d))
    # print(sess.run(e))
    # print(sess.run(d1))
    # print(sess.run(d1))
    # for i in range(1000):
    #     sess.run(train_step)
    #     print(sess.run(d))
    #     print(sess.run(d2))
    #     print(sess.run(X))
    # print(sess.run(d1))
    # print(sess.run(Xsp))

