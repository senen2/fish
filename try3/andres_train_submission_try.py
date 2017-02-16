'''
Model Evaluation

@author: botpi
'''
import tensorflow as tf
import numpy as np
from apifish import *
import scipy.io
from params import param
import time
import os

features = scipy.io.loadmat("resp_50_cost_conv5_diff_chan_1")
sub_file = "train_2.csv"
parameters = param()
files = os.listdir("../../data/fish/train-fix/")
# files = os.listdir("../../data/fish/test_stg1_fix/")
samples = len(files)
img_path = "../../data/fish/train-fix/"
lbl_path = "../../data/fish/label-train-fix/"
img_queue = []
lbl_queue = []
for file in files:
	img_queue.append(img_path + file)
	lbl_queue.append(lbl_path + file +".txt")

cv1_size = parameters["cv1_size"]
cv2_size = parameters["cv2_size"]
cv1_channels = parameters["cv1_channels"]
cv2_channels = parameters["cv2_channels"]
hidden = parameters["hidden"]
img_width = parameters["img_width"]
img_height = parameters["img_height"]
categories = parameters["categories"]
cv_all_size = 5
cv_all_channels = 1
last_img_size = 14
channels_jpg = 1

filename_queue = tf.train.string_input_producer(img_queue, shuffle=False)

filename_queue_label = tf.train.string_input_producer(lbl_queue, shuffle=False)

image_reader = tf.WholeFileReader()
label_reader = tf.WholeFileReader()

image_name , image_file = image_reader.read(filename_queue)
_ , label_file = label_reader.read(filename_queue_label)

ratio_jpg = 1
img_height = img_height/ratio_jpg
img_width = img_width/ratio_jpg

x = tf.image.decode_jpeg(image_file, channels=channels_jpg, ratio=ratio_jpg)
x.set_shape([img_height, img_width, channels_jpg])

record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
col1, col2, col3, col4, col5, col6, col7, col8 = tf.decode_csv(label_file, record_defaults=record_defaults)
y = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8])
y = tf.pack([y])

W_conv1 = weight_variable([cv_all_size, cv_all_size, channels_jpg, cv_all_channels])
b_conv1 = bias_variable([cv_all_channels])
W_conv2 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels * 2])
b_conv2 = bias_variable([cv_all_channels * 2])
W_conv3 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 2, cv_all_channels * 4])
b_conv3 = bias_variable([cv_all_channels * 4])
W_conv4 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 4, cv_all_channels * 8])
b_conv4 = bias_variable([cv_all_channels * 8])
W_conv5 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 16])
b_conv5 = bias_variable([cv_all_channels * 16])
# W_conv6 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 16, cv_all_channels * 32])
# b_conv6 = bias_variable([cv_all_channels * 32])
# W_conv7 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels])
# b_conv7 = bias_variable([cv_all_channels])
# W_conv8 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels])
# b_conv8 = bias_variable([cv_all_channels])
# W_conv9 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels])
# b_conv9 = bias_variable([cv_all_channels])
# W_conv10 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels])
# b_conv10 = bias_variable([cv_all_channels])

W_fc1 = weight_variable([last_img_size * last_img_size * cv_all_channels * 16, hidden])
b_fc1 = bias_variable([hidden])
W_fc2 = weight_variable([hidden, categories])
b_fc2 = bias_variable([categories])

x = tf.reshape(tf.cast(x, tf.float32), [-1,img_width,img_height,channels_jpg])

# conv
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)
# h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
# h_pool6 = max_pool_2x2(h_conv6)
# h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)
# h_pool7 = max_pool_2x2(h_conv7)
# h_conv8 = tf.nn.relu(conv2d(h_pool7, W_conv8) + b_conv8)
# h_pool8 = max_pool_2x2(h_conv8)
# h_conv9 = tf.nn.relu(conv2d(h_pool8, W_conv9) + b_conv9)
# h_pool9 = max_pool_2x2(h_conv9)
# h_conv10 = tf.nn.relu(conv2d(h_pool9, W_conv10) + b_conv10)
# h_pool10 = max_pool_2x2(h_conv10)

h_pool_last_flat = tf.reshape(h_pool5, [-1, last_img_size * last_img_size  * cv_all_channels * 16])

h_fc1 = tf.nn.relu(tf.matmul(h_pool_last_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, 1)
pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print pred
print "h_conv1", h_conv1
print "h_pool1", h_pool1
print "h_conv2", h_conv2
print "h_pool2", h_pool2
correct_prediction2 = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred_arg2 = tf.argmax(pred, 1)
# y_arg2 = tf.argmax(y, 1)
# print "y",y
# print "pred", pred

init = tf.global_variables_initializer()
acc_total = 0.0

with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	# sleep = 5
	# print "sleep:",sleep
	# time.sleep(sleep)
	r = []
	r.append(["image","ALB","BET","DOL","LAG","NoF","OTHER","SHARK","YFT","correct_prediction"])
	for step in xrange(samples):
		print "step:",step
		# prob,correct_prediction2,acc,pred_arg,y_arg,y_tes = sess.run([pred,correct_prediction,accuracy,pred_arg2,y_arg2,y],{ 
		#     W_conv1:features["W_conv1"],
		#     b_conv1:features["b_conv1"][0],
		#     W_conv2:features["W_conv2"],
		#     b_conv2:features["b_conv2"][0],
		#     W_fc1:features["W_fc1"],
		#     b_fc1:features["b_fc1"][0],
		#     W_fc2:features["W_fc2"],
		#     b_fc2:features["b_fc2"][0]                          
		#   })
		prob, img_name,pred_arg,correct_prediction = sess.run([pred,image_name,pred_arg2,correct_prediction2],{ 
		    W_conv1:features["W_conv1"],
		    b_conv1:features["b_conv1"][0],
		    W_conv2:features["W_conv2"],
		    b_conv2:features["b_conv2"][0],
		    W_conv3:features["W_conv3"],
		    b_conv3:features["b_conv3"][0],
		    W_conv4:features["W_conv4"],
		    b_conv4:features["b_conv4"][0],
		    W_conv5:features["W_conv5"],
		    b_conv5:features["b_conv5"][0],
		    # W_conv6:features["W_conv6"],
		    # b_conv6:features["b_conv6"][0],
		    # W_conv7:features["W_conv7"],
		    # b_conv7:features["b_conv7"][0],
		    W_fc1:features["W_fc1"],
		    b_fc1:features["b_fc1"][0],
		    W_fc2:features["W_fc2"],
		    b_fc2:features["b_fc2"][0]                          
		  })
		# print prob
		prob = prob[0]
		# print prob
		img = img_name[26:]
		print img
		# print str(img), prob[0],prob[1],prob[2],prob[3],prob[4],prob[5],prob[6],prob[7]
		rest_all = 0.0
		prob[0],prob[1],prob[2],prob[3],prob[4],prob[5],prob[6],prob[7] = rest_all,rest_all,rest_all,rest_all,rest_all,rest_all,rest_all,rest_all
		prob[pred_arg] = 0.99
		cp = 0
		if correct_prediction == True:
			cp = 1
		r.append([str(img), prob[0],prob[1],prob[2],prob[3],prob[4],prob[5],prob[6],prob[7],cp])
		# break
		# r.append([str(img), prob[0],prob[1],prob[2],prob[3],prob[4],prob[5],prob[6],prob[7]])
		# acc_total += acc
		# if step >= 10:
		# 	break
	# print "accuracy:", acc_total*100.0/samples,"total good int:",acc_total
	print "image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT"
	np.savetxt(sub_file, r, delimiter=',', fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s,%s")

# coord.join(threads)
# coord.request_stop()
sess.close()
print sub_file