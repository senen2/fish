'''
Created on Dec 9, 2016

@author: botpi
'''
import tensorflow as tf
import numpy as np
from apifish import *
import scipy.io
from apifish import *

print "begin"
dirdata = "../../data/fish/"
group = "train 64x36"
group = "LAG 128x72"
learning_rate = 0.0005
training_epochs = 1
img_width = 128 # 1280
img_height = 72 # 720
hidden = 1000

images, labels = read_images(dirdata + group)

x0 = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 3])
x = tf.reshape(x0, [-1, img_height * img_width * 3])  

# W = tf.Variable(tf.zeros([img_height * img_width * 3, 1000]))
# b = tf.Variable(tf.zeros([1000]))

W1 = weight_variable([img_height * img_width * 3, hidden])
b1 = bias_variable([hidden])

W2 = weight_variable([hidden, img_height * img_width * 3])
b2 = bias_variable([img_height * img_width * 3])

h = tf.matmul(x, W1) + b1
pred = tf.matmul(h, W2) + b2
print "pred", pred

cost = tf.reduce_mean(tf.reduce_sum((pred-x)**2, reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(x, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in xrange(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={x0: images})
        if (epoch+1) % 1 == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)

    features = {}
    features["W1"] = W1.eval()
    features["b1"] = b1.eval()
    features["W2"] = W2.eval()
    features["b2"] = b2.eval()
    features["h"] = sess.run([h], feed_dict={x0: images})
    scipy.io.savemat(group + "_resp", features, do_compression=True) 

print "end"