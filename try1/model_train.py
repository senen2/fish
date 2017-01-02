'''
Model Train

Created on Dic 7, 2016

@author: botpi
'''

import tensorflow as tf
import numpy as np
from apifish import *
import scipy.io

def train_tf(images, labels, parameters, training_epochs = 100):
#    display_step = 100
    cv1_size = parameters["cv1_size"]
    cv2_size = parameters["cv2_size"]
    cv1_channels = parameters["cv1_channels"]
    cv2_channels = parameters["cv2_channels"]
    hidden = parameters["hidden"]
    img_width = parameters["img_width"]
    img_height = parameters["img_height"]
    categories = parameters["categories"]
    learning_rate = parameters["learning_rate"]
    dropout = parameters["dropout"]
    display_step = 1
    
    best_cost = 1e99
    best_acc = 0

    x = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 3])
    y = tf.placeholder(tf.float32, shape=[None, categories])  
      
    # First Convolutional Layer  
    W_conv1 = weight_variable([cv1_size, cv1_size, 3, cv1_channels])
    b_conv1 = bias_variable([cv1_channels])
    
    x_image = tf.reshape(x, [-1, img_width, img_height,3])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    print "h_conv1", h_conv1
    print "h_pool1", h_pool1
    
    # Second Convolutional Layer
    W_conv2 = weight_variable([cv2_size, cv2_size, cv1_channels, cv2_channels])
    b_conv2 = bias_variable([cv2_channels])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print "h_conv2", h_conv2
    print "h_pool2", h_pool2
    
    # Densely Connected Layer
    W_fc1 = weight_variable([img_width/4 * img_height/4 * cv2_channels, hidden])
    b_fc1 = bias_variable([hidden])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, img_width/4 * img_height/4  * cv2_channels])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Readout Layer
    W_fc2 = weight_variable([hidden, categories])
    b_fc2 = bias_variable([categories])
    
    pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    print "pred", pred

    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-20), reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    init = tf.initialize_all_variables()
    #init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
    
        # Training cycle
        for epoch in xrange(training_epochs):
    #         print epoch
            _, c = sess.run([optimizer, cost], feed_dict={x: images, y: labels, keep_prob: dropout})
            #if (epoch+1) % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)
                
            if c < best_cost:
                best_cost = c
                best_epoch_cost = epoch
                features = {}
                features["W_conv1"] = W_conv1.eval()
                features["b_conv1"] = b_conv1.eval()
                features["W_conv2"] = W_conv2.eval()
                features["b_conv2"] = b_conv2.eval()
                features["W_fc1"] = W_fc1.eval()
                features["b_fc1"] = b_fc1.eval()
                features["W_fc2"] = W_fc2.eval()
                features["b_fc2"] = b_fc2.eval()
                scipy.io.savemat("resp_best_cost", features, do_compression=True) 
            
            acc = accuracy.eval({x:images, y: labels, keep_prob: 1})     
            if acc > best_acc:
                best_acc = acc
                best_epoch_acc = epoch
                features = {}
                features["W_conv1"] = W_conv1.eval()
                features["b_conv1"] = b_conv1.eval()
                features["W_conv2"] = W_conv2.eval()
                features["b_conv2"] = b_conv2.eval()
                features["W_fc1"] = W_fc1.eval()
                features["b_fc1"] = b_fc1.eval()
                features["W_fc2"] = W_fc2.eval()
                features["b_fc2"] = b_fc2.eval()
                scipy.io.savemat("resp_best_acc", features, do_compression=True) 
                    
        print "Optimization Finished!"
    
        # Test model
     
        acc = accuracy.eval({x:images, y: labels, keep_prob: 1})    
        prob = pred.eval({x:images, y: labels, keep_prob: 1})
#         print auctf[0]
#         aucr = auctf[0].eval({x:images, y: labels, keep_prob: 1})
        
        features = {}
        features["W_conv1"] = W_conv1.eval()
        features["b_conv1"] = b_conv1.eval()
        features["W_conv2"] = W_conv2.eval()
        features["b_conv2"] = b_conv2.eval()
        features["W_fc1"] = W_fc1.eval()
        features["b_fc1"] = b_fc1.eval()
        features["W_fc2"] = W_fc2.eval()
        features["b_fc2"] = b_fc2.eval()
    
    print "best cost", best_cost, "epoch", best_epoch_cost
    print "best acc", best_acc, "epoch", best_epoch_acc

    return features, prob, acc, c
