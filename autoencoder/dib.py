'''
Created on Dec 9, 2016

@author: botpi
'''
import tensorflow as tf
import numpy as np
from apifish import *
import scipy.io
from scipy.misc import toimage
from apifish import *

print "begin"
dirdata = "../../data/fish/"
group = "train 64x36"
group = "LAG 128x72"
learning_rate = 0.0005
training_epochs = 100
img_width = 128 # 1280
img_height = 72 # 720
hidden = 1000

features = scipy.io.loadmat(group + "_resp")
images, labels = read_images(dirdata + group)

x0 = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 3])
x = tf.reshape(x0, [-1, img_height * img_width * 3])  

W1 = weight_variable([img_height * img_width * 3, hidden])
b1 = bias_variable([hidden])

W2 = weight_variable([hidden, img_height * img_width * 3])
b2 = bias_variable([img_height * img_width * 3])

h = tf.matmul(x, W1) + b1
pred = tf.matmul(h, W2) + b2
print "pred", pred

dib = tf.reshape(pred, [img_height, img_width, 3])  

cost = tf.reduce_mean(tf.reduce_sum((pred-x)**2, reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(x, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    img = dib.eval({x:images, 
        W1:features["W1"],
        b1:features["b1"][0],
        W2:features["W2"],
        b2:features["b2"][0]
      })
toimage(img).show()

print "end"