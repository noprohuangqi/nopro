# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 10:01:07 2018

@author: 32002
"""

import tensorflow as tf

matrix1 = tf.constant([[3,3]])

matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1,matrix2)

sess = tf.Session()

result = sess.run(product)

print(result)

sess.close()



##with tf.Session() as sess:
#    result2=sess.run(product)
 #   print(result2)