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
a = Dotdict()
a.value = [1,2,3]
a.indices =[[0,0],[0,3],[0,8]]
Xsp = tf.SparseTensor(values=a.value,
                      indices=a.indices,
                      dense_shape=[1,10])

b = [Xsp,Xsp,Xsp]
b1=tf.sparse_concat(axis=0,sp_inputs=b)

# Xsp1 = tf.SparseTensor(values=a1,indices=b1,dense_shape=[10])
print(a.value)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(Xsp))
    print(sess.run(tf.sparse_tensor_to_dense(Xsp)))
    print(sess.run(tf.sparse_tensor_to_dense(b1)))
    print(sess.run(b1))
    print("######")
    print(sess.run(b1.indices))


