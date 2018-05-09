# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 10:29:43 2018

@author: 32002
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)

input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))