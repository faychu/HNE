import tensorflow as tf
import numpy as np

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
n_samples = 4
n_features = 5
dim = 3
X = tf.Variable([[1,0,0,0,1],[0,1,0.5,0,0.5],[1,0,0,0.5,0.5],[0,0,1,1,0]])
embed = tf.nn.embedding_lookup(X,[0,1])
ones = tf.ones([1,5],dtype=tf.int32)
val = tf.Variable([[2,4]],dtype=tf.float32)
# mul = tf.matmul(val,ones,transpose_a=True)
a = Dotdict()
a.value = [1,2,3]
a.indices =[[0,0],[0,2],[0,3]]
Xsp = tf.SparseTensor(values=a.value,
                      indices=a.indices,
                      dense_shape=[1,4])
Xsp1 = tf.SparseTensor(values=[2,5],indices=[[0,1],[0,2]],dense_shape=[1,4])
b = [Xsp, Xsp, Xsp1]

b1=tf.sparse_concat(axis=0,sp_inputs=b)
bbb = tf.sparse_split(sp_input=b1,num_split=3,axis=0)
d = tf.Variable([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
print(b1)

chengji  = tf.sparse_tensor_dense_matmul(b1,d)*3
print(chengji)
# Xsp1 = tf.SparseTensor(values=a1,indices=b1,dense_shape=[10])
print(a.value)

a1 =  tf.Variable(tf.random_normal([10]))
s1 = tf.Variable([[1,2,3,4,5]],dtype=tf.float32)
s2 = tf.Variable([[4,25,6,0,0]],dtype=tf.float32)
sss = tf.concat([s2,s1],axis=0)
print(sss)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(X[0]))
    # print(sess.run(b1[0]))  'SparseTensor' object does not support indexing
    print(sess.run(tf.sparse_tensor_to_dense(Xsp)))
    print(sess.run(tf.sparse_tensor_to_dense(b1)))
    # print(sess.run(b1))
    print("######")
    # print(sess.run(b1.indices))
    # print(sess.run(chengji))
    # print(sess.run(embed))
    # print(sess.run(mul))
    # print(sess.run((mul*embed)**2*0.3))
    # print(sess.run(tf.shape(embed)))
    # print(sess.run(b1.dense_shape))
    print(sess.run(sss))
    print(sess.run(s1))
    sss = tf.concat([sss, s1], axis=0)
    print(sess.run(sss))
    # for i in bbb:
        # print("index:")
        # index = tf.transpose(i.indices)[1]
        # print(sess.run(index))
        # print(sess.run(tf.nn.embedding_lookup(X,index)))
        # print("value:")
        # print(sess.run(i.values))
        # v = tf.reshape(i.values, [-1, 1])
        # print(sess.run(v))
        # c = tf.matmul(v, ones)
        # print(sess.run(c))
        # x = tf.reduce_sum(X, axis=0)
        # print(sess.run(tf.shape(x)))
        # print(sess.run(x))

    print(ones)


