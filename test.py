import tensorflow as tf
import numpy as np
'''
# example = tf.SparseTensor(indices=[[0], [1], [2]], values=[3, 6, 9], dense_shape=[3])
example =[0,4]
vocabulary_size = 10
embedding_size = 1
var = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
embeddings = tf.Variable(var)
c = tf.Variable([1,2,3],dtype=tf.float64)
# embed = tf.nn.embedding_lookup_sparse(embeddings, example, None)
a = tf.nn.embedding_lookup(embeddings, [1,2])
b = tf.nn.embedding_lookup(embeddings, [3,4])


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # print(sess.run(embed)) # prints [  9.  36.  81.]
    # print(sess.run(a*b))
    print(sess.run(tf.nn.l2_loss(c-1)))
'''

class A():
    def __init__(self):
        self.count = 11
    def add(self, num):
        self.count += num
        return self.count

a = A()
b = A()
print(a.add(100))
print(a.add(25))
print(b.count)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)  # 变量的初始值为截断正太分布
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)