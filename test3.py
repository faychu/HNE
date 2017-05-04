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
ones = tf.ones([1,5],dtype=tf.float32)
val = tf.Variable([[2,4]],dtype=tf.float32)
mul = tf.matmul(val,ones,transpose_a=True)
a = Dotdict()
a.value = [1,2,3]
a.indices =[[0,0],[0,3],[0,8]]
Xsp = tf.SparseTensor(values=a.value,
                      indices=a.indices,
                      dense_shape=[1,10])

b = [Xsp,Xsp,Xsp]

b1=tf.sparse_concat(axis=0,sp_inputs=b)
bbb = tf.sparse_split(sp_input=b1,num_split=3,axis=0)
d = tf.Variable([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
print(b1)

chengji  = tf.sparse_tensor_dense_matmul(b1,d)*3
print(chengji)
# Xsp1 = tf.SparseTensor(values=a1,indices=b1,dense_shape=[10])
print(a.value)

a1 =  tf.Variable(tf.random_normal([10]))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(X[0]))
    # print(sess.run(b1[0]))  'SparseTensor' object does not support indexing
    print(sess.run(tf.sparse_tensor_to_dense(Xsp)))
    print(sess.run(tf.sparse_tensor_to_dense(b1)))
    # print(sess.run(b1))
    print("######")
    # print(sess.run(b1.indices))
    print(sess.run(chengji))
    print(sess.run(embed))
    print(sess.run(mul))
    print(sess.run((mul*embed)**2*0.3))
    print(sess.run(tf.shape(embed)))
    print(sess.run(bbb[1].values))
    for i in range(3):
        print(sess.run(tf.sparse_tensor_to_dense(bbb[i])))


