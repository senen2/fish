'''
Model Train

Created on Jan 17, 2017

@author: botpi
'''

import tensorflow as tf
import numpy as np
from apifish import *
import scipy.io
from params import param
import time
import os
# import kronos

print "initialize"
# krono = kronos.krono()
files = os.listdir("../../data/fish/train-fix/")
samples = len(files)
img_path = "../../data/fish/train-fix/"
lbl_path = "../../data/fish/label-train-fix/"
img_queue = []
lbl_queue = []
for file in files:
    img_queue.append(img_path + file)
    lbl_queue.append(lbl_path + file +".txt")

training_epochs = 800000 + 1
display_step = 1

parameters = param()
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
dropout = 0.5
save_epoch = 1000
# beta = 0.001
beta = 0.17 # only y
# beta = 0.01 # softmax on y
cv_all_size = 7
cv_all_channels = 1
last_img_size = 7
batch_size = 1
channels_jpg = 1
mat_name_file = "_conv16_pool5_imgNet_chan_" + str(cv_all_channels)

best_cost = 1e99
best_acc = 0

filename_queue = tf.train.string_input_producer(img_queue, shuffle=False)

filename_queue_label = tf.train.string_input_producer(lbl_queue, shuffle=False)

image_reader = tf.WholeFileReader()
label_reader = tf.WholeFileReader()

_ , image_file = image_reader.read(filename_queue)
_ , label_file = label_reader.read(filename_queue_label)

ratio_jpg = 1
img_height = img_height/ratio_jpg
img_width = img_width/ratio_jpg
image = tf.image.decode_jpeg(image_file, channels=channels_jpg, ratio=ratio_jpg)
image.set_shape([img_height, img_width, channels_jpg])
record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
col1, col2, col3, col4, col5, col6, col7, col8 = tf.decode_csv(label_file, record_defaults=record_defaults)
label = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8])
# min_after_dequeue + 3 * batch_size
x, y = tf.train.shuffle_batch(
    [image, label], batch_size = batch_size, 
    capacity = 4000,
    min_after_dequeue = 600)

x = tf.cast(x, tf.float32)
y = tf.cast(y, tf.float32)
keep_prob = tf.placeholder(tf.float32)
  
W_conv1 = weight_variable([cv_all_size, cv_all_size, channels_jpg, cv_all_channels])
b_conv1 = bias_variable([cv_all_channels])
W_conv2 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels])
b_conv2 = bias_variable([cv_all_channels])

W_conv3 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels * 2])
b_conv3 = bias_variable([cv_all_channels * 2])
W_conv4 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 2, cv_all_channels * 2])
b_conv4 = bias_variable([cv_all_channels * 2])

W_conv5 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 2, cv_all_channels * 4])
b_conv5 = bias_variable([cv_all_channels * 4])
W_conv6 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 4, cv_all_channels * 4])
b_conv6 = bias_variable([cv_all_channels * 4])
W_conv7 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 4, cv_all_channels * 4])
b_conv7 = bias_variable([cv_all_channels * 4])
W_conv8 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 4, cv_all_channels * 4])
b_conv8 = bias_variable([cv_all_channels * 4])

W_conv9 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 4, cv_all_channels * 8])
b_conv9 = bias_variable([cv_all_channels * 8])
W_conv10 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 8])
b_conv10 = bias_variable([cv_all_channels * 8])
W_conv11 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 8])
b_conv11 = bias_variable([cv_all_channels * 8])
W_conv12 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 8])
b_conv12 = bias_variable([cv_all_channels * 8])

W_conv13 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 8])
b_conv13 = bias_variable([cv_all_channels * 8])
W_conv14 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 8])
b_conv14 = bias_variable([cv_all_channels * 8])
W_conv15 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 8])
b_conv15 = bias_variable([cv_all_channels * 8])
W_conv16 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 8])
b_conv16 = bias_variable([cv_all_channels * 8])

W_fc1 = weight_variable([last_img_size * last_img_size * cv_all_channels * 8, hidden])
b_fc1 = bias_variable([hidden])
W_fc2 = weight_variable([hidden, hidden])
b_fc2 = bias_variable([hidden])
W_fc3 = weight_variable([hidden, categories])
b_fc3 = bias_variable([categories])

x = tf.reshape(x, [-1, img_width, img_height,channels_jpg])

# conv
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)
h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)
h_pool8 = max_pool_2x2(h_conv8)

h_conv9 = tf.nn.relu(conv2d(h_pool8, W_conv9) + b_conv9)
h_conv10 = tf.nn.relu(conv2d(h_conv9, W_conv10) + b_conv10)
h_conv11 = tf.nn.relu(conv2d(h_conv10, W_conv11) + b_conv11)
h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)
h_pool12 = max_pool_2x2(h_conv12)

h_conv13 = tf.nn.relu(conv2d(h_pool12, W_conv13) + b_conv13)
h_conv14 = tf.nn.relu(conv2d(h_conv13, W_conv14) + b_conv14)
h_conv15 = tf.nn.relu(conv2d(h_conv14, W_conv15) + b_conv15)
h_conv16 = tf.nn.relu(conv2d(h_conv15, W_conv16) + b_conv16)
h_pool16 = max_pool_2x2(h_conv16)

h_pool_last_flat = tf.reshape(h_pool16, [-1, last_img_size * last_img_size  * cv_all_channels * 8])

# full conected
h_fc1 = tf.nn.relu(tf.matmul(h_pool_last_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
# pred = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
pred = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
print "x", x
print "y", y
print "pred", pred
print "h_conv1", h_conv1
# print "h_pool1", h_pool1
print "h_conv2", h_conv2
# print "h_pool2", h_pool2

# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-20), reduction_indices=[1]))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y) +
#     (beta * (tf.nn.l2_loss(W_fc1) + 
#         tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3))))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y) +
#     beta * tf.nn.l2_loss(W_fc1) + beta * tf.nn.l2_loss(W_fc2) + beta * tf.nn.l2_loss(W_fc3) +
#     beta * tf.nn.l2_loss(W_conv1) + beta * tf.nn.l2_loss(W_conv2) +
#     beta * tf.nn.l2_loss(W_conv3) + beta * tf.nn.l2_loss(W_conv4) +
#     beta * tf.nn.l2_loss(W_conv5) + beta * tf.nn.l2_loss(W_conv6) +
#     beta * tf.nn.l2_loss(W_conv7) + beta * tf.nn.l2_loss(W_conv8) +
#     beta * tf.nn.l2_loss(W_conv9) + beta * tf.nn.l2_loss(W_conv10) +
#     beta * tf.nn.l2_loss(W_conv11) + beta * tf.nn.l2_loss(W_conv12) +
#     beta * tf.nn.l2_loss(W_conv13) + beta * tf.nn.l2_loss(W_conv14) +
#     beta * tf.nn.l2_loss(W_conv15) + beta * tf.nn.l2_loss(W_conv16))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=tf.nn.softmax(y)))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) + tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(y,pred)),1))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_arg2 = tf.argmax(y, 1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    sleep = 5
    print "queue sleep in seg:", sleep
    time.sleep(sleep)
    print "end queue sleep"

    # Training cycle
    print "learning..."
    for epoch in xrange(training_epochs):
        _, c, acc = sess.run([optimizer, cost, accuracy], feed_dict={keep_prob: dropout})

        print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),"dropout: " + str(dropout) + " bad acc:", round(acc*100.0,2),"%"
            
        if epoch%save_epoch == 0:
            # print "Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(c),"dropout: 0.5 bad acc:", round(acc*100.0,2),"%"
            # print krono.elapsed()/save_epoch
            # krono.start()
            # alb = 0
            # bet = 0
            # dol = 0
            # lag = 0
            # nof = 0
            # other = 0
            # shark = 0
            # yft = 0

            # alb_total = 1e-20
            # bet_total = 1e-20
            # dol_total = 1e-20
            # lag_total = 1e-20
            # nof_total = 1e-20 
            # other_total = 1e-20
            # shark_total = 1e-20
            # yft_total = 1e-20
            # acc, y_arg , c_p = sess.run([accuracy,y_arg2,correct_prediction],feed_dict={keep_prob: 1.0})
            # for i in xrange(len(y_arg)):
            #     if y_arg[i] == 0:
            #         alb_total += 1
            #     elif y_arg[i] == 1:
            #         bet_total += 1
            #     elif y_arg[i] == 2:
            #         dol_total += 1
            #     elif y_arg[i] == 3:
            #         lag_total += 1
            #     elif y_arg[i] == 4:
            #         nof_total += 1
            #     elif y_arg[i] == 5:
            #         other_total += 1
            #     elif y_arg[i] == 6:
            #         shark_total += 1
            #     elif y_arg[i] == 7:
            #         yft_total += 1

            #     if c_p[i] == True:
            #         if y_arg[i] == 0:
            #             alb += 1
            #         elif y_arg[i] == 1:
            #             bet += 1
            #         elif y_arg[i] == 2:
            #             dol += 1
            #         elif y_arg[i] == 3:
            #             lag += 1
            #         elif y_arg[i] == 4:
            #             nof += 1
            #         elif y_arg[i] == 5:
            #             other += 1
            #         elif y_arg[i] == 6:
            #             shark += 1
            #         elif y_arg[i] == 7:
            #             yft += 1

            # print "###########################################################################"
            # print "                                         saving weights"
            # print "                                         accuracy:", round(acc*100.0,2),"%"
            # print "    ALB  ", round(alb*100.0/alb_total,2),"total good int:",alb, "of", int(alb_total)
            # print "    BET  ", round(bet*100.0/bet_total,2),"total good int:",bet, "of", int(bet_total)
            # print "    DOL  ", round(dol*100.0/dol_total,2),"total good int:",dol, "of", int(dol_total)
            # print "    LAG  ", round(lag*100.0/lag_total,2),"total good int:",lag, "of", int(lag_total)
            # print "    NoF  ", round(nof*100.0/nof_total,2),"total good int:",nof, "of", int(nof_total)
            # print "    OTHER", round(other*100.0/other_total,2),"total good int:",other, "of", int(other_total)
            # print "    SHARK", round(shark*100.0/shark_total,2),"total good int:",shark, "of", int(shark_total)
            # print "    YFT  ", round(yft*100.0/yft_total,2),"total good int:",yft, "of", int(yft_total)
            # print
            # print "    total batch", str(int(alb_total+bet_total+dol_total+lag_total+nof_total+other_total+shark_total+yft_total))
            # print "###########################################################################"
            features = {}
            features["W_conv1"] = W_conv1.eval()
            features["b_conv1"] = b_conv1.eval()
            features["W_conv2"] = W_conv2.eval()
            features["b_conv2"] = b_conv2.eval()
            features["W_conv3"] = W_conv3.eval()
            features["b_conv3"] = b_conv3.eval()
            features["W_conv4"] = W_conv4.eval()
            features["b_conv4"] = b_conv4.eval()
            features["W_conv5"] = W_conv5.eval()
            features["b_conv5"] = b_conv5.eval()
            features["W_conv6"] = W_conv6.eval()
            features["b_conv6"] = b_conv6.eval()
            features["W_conv7"] = W_conv7.eval()
            features["b_conv7"] = b_conv7.eval()
            features["W_conv8"] = W_conv8.eval()
            features["b_conv8"] = b_conv8.eval()
            features["W_conv9"] = W_conv9.eval()
            features["b_conv9"] = b_conv9.eval()
            features["W_conv10"] = W_conv10.eval()
            features["b_conv10"] = b_conv10.eval()
            features["W_conv11"] = W_conv11.eval()
            features["b_conv11"] = b_conv11.eval()
            features["W_conv12"] = W_conv12.eval()
            features["b_conv12"] = b_conv12.eval()
            features["W_conv13"] = W_conv13.eval()
            features["b_conv13"] = b_conv13.eval()
            features["W_conv14"] = W_conv14.eval()
            features["b_conv14"] = b_conv14.eval()
            features["W_conv15"] = W_conv15.eval()
            features["b_conv15"] = b_conv15.eval()
            features["W_conv16"] = W_conv16.eval()
            features["b_conv16"] = b_conv16.eval()
            features["W_fc1"] = W_fc1.eval()
            features["b_fc1"] = b_fc1.eval()
            features["W_fc2"] = W_fc2.eval()
            features["b_fc2"] = b_fc2.eval()
            features["W_fc3"] = W_fc3.eval()
            features["b_fc3"] = b_fc3.eval()
            scipy.io.savemat("resp" + str(mat_name_file), features, do_compression=True) 
        
    print "Optimization Finished!"

    features = {}
    features["W_conv1"] = W_conv1.eval()
    features["b_conv1"] = b_conv1.eval()
    features["W_conv2"] = W_conv2.eval()
    features["b_conv2"] = b_conv2.eval()
    features["W_conv3"] = W_conv3.eval()
    features["b_conv3"] = b_conv3.eval()
    features["W_conv4"] = W_conv4.eval()
    features["b_conv4"] = b_conv4.eval()
    features["W_conv5"] = W_conv5.eval()
    features["b_conv5"] = b_conv5.eval()
    features["W_conv6"] = W_conv6.eval()
    features["b_conv6"] = b_conv6.eval()
    features["W_conv7"] = W_conv7.eval()
    features["b_conv7"] = b_conv7.eval()
    features["W_conv8"] = W_conv8.eval()
    features["b_conv8"] = b_conv8.eval()
    features["W_conv9"] = W_conv9.eval()
    features["b_conv9"] = b_conv9.eval()
    features["W_conv10"] = W_conv10.eval()
    features["b_conv10"] = b_conv10.eval()
    features["W_conv11"] = W_conv11.eval()
    features["b_conv11"] = b_conv11.eval()
    features["W_conv12"] = W_conv12.eval()
    features["b_conv12"] = b_conv12.eval()
    features["W_conv13"] = W_conv13.eval()
    features["b_conv13"] = b_conv13.eval()
    features["W_conv14"] = W_conv14.eval()
    features["b_conv14"] = b_conv14.eval()
    features["W_conv15"] = W_conv15.eval()
    features["b_conv15"] = b_conv15.eval()
    features["W_conv16"] = W_conv16.eval()
    features["b_conv16"] = b_conv16.eval()
    features["W_fc1"] = W_fc1.eval()
    features["b_fc1"] = b_fc1.eval()
    features["W_fc2"] = W_fc2.eval()
    features["b_fc2"] = b_fc2.eval()
    features["W_fc3"] = W_fc3.eval()
    features["b_fc3"] = b_fc3.eval()

    coord.request_stop()
    coord.join(threads)
    sess.close()


print "epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_heigth img_width dropout"
print ("    %s      %s        %s       %s          %s            %s        %s        %s      %s      %s" 
    % (training_epochs, parameters["learning_rate"], parameters["cv1_size"], parameters["cv2_size"], parameters["cv1_channels"]
       , parameters["cv2_channels"], parameters["hidden"], parameters["img_height"], parameters["img_width"], parameters["dropout"]))

print "Cost", cost

print "saving last"
scipy.io.savemat("resp" + str(mat_name_file), features, do_compression=True)    
print "end"