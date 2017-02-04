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

features = scipy.io.loadmat("resp_100_cost")
parameters = param()
files = os.listdir("../../data/fish/train-fix/")
# files = os.listdir("../../data/fish/test_stg1_fix/")
samples = len(files)

cv1_size = parameters["cv1_size"]
cv2_size = parameters["cv2_size"]
cv1_channels = parameters["cv1_channels"]
cv2_channels = parameters["cv2_channels"]
hidden = parameters["hidden"]
img_width = parameters["img_width"]
img_height = parameters["img_height"]
categories = parameters["categories"]

filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("../../data/fish/train-fix/*.jpg"), shuffle=False)

# filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once("../../data/fish/test_stg1_fix/*.jpg"), shuffle=False)

filename_queue_label = tf.train.string_input_producer(
    tf.train.match_filenames_once("../../data/fish/label-train-fix/*.txt"), shuffle=False)

image_reader = tf.WholeFileReader()
label_reader = tf.WholeFileReader()

_ , image_file = image_reader.read(filename_queue)
_ , label_file = label_reader.read(filename_queue_label)

channels_jpg = 1
ratio_jpg = 1
img_height = img_height/ratio_jpg
img_width = img_width/ratio_jpg

x = tf.image.decode_jpeg(image_file, channels=channels_jpg, ratio=ratio_jpg)
x.set_shape([img_height, img_width, channels_jpg])

record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
col1, col2, col3, col4, col5, col6, col7, col8 = tf.decode_csv(label_file, record_defaults=record_defaults)
y = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8])
y = tf.pack([y])

W_conv1 = weight_variable([cv1_size, cv1_size, channels_jpg, cv1_channels])
b_conv1 = bias_variable([cv1_channels])
W_conv2 = weight_variable([cv2_size, cv2_size, cv1_channels, cv2_channels])
b_conv2 = bias_variable([cv2_channels])
W_fc1 = weight_variable([img_width/4 * img_height/4 * cv2_channels, hidden])
b_fc1 = bias_variable([hidden])
W_fc2 = weight_variable([hidden, categories])
b_fc2 = bias_variable([categories])

x_image = tf.reshape(tf.cast(x, tf.float32), [-1,img_width,img_height,channels_jpg])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, img_width/4 * img_height/4  * cv2_channels])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, 1)
pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print pred
print "h_conv1", h_conv1
print "h_pool1", h_pool1
print "h_conv2", h_conv2
print "h_pool2", h_pool2
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred_arg2 = tf.argmax(pred, 1)
y_arg2 = tf.argmax(y, 1)
print "y",y
print "pred", pred

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
	r.append(["image","ALB","BET","DOL","LAG","NoF","OTHER","SHARK","YFT"])
	for step in xrange(samples):
		print "step:",step
		prob,correct_prediction2,acc,pred_arg,y_arg,y_tes = sess.run([pred,correct_prediction,accuracy,pred_arg2,y_arg2,y],{ 
		    W_conv1:features["W_conv1"],
		    b_conv1:features["b_conv1"][0],
		    W_conv2:features["W_conv2"],
		    b_conv2:features["b_conv2"][0],
		    W_fc1:features["W_fc1"],
		    b_fc1:features["b_fc1"][0],
		    W_fc2:features["W_fc2"],
		    b_fc2:features["b_fc2"][0]                          
		  })
		# prob = sess.run([pred],{ 
		#     W_conv1:features["W_conv1"],
		#     b_conv1:features["b_conv1"][0],
		#     W_conv2:features["W_conv2"],
		#     b_conv2:features["b_conv2"][0],
		#     W_fc1:features["W_fc1"],
		#     b_fc1:features["b_fc1"][0],
		#     W_fc2:features["W_fc2"],
		#     b_fc2:features["b_fc2"][0]                          
		#   })
		# print prob
		prob = prob[0]
		# prob = prob[0]
		# print prob
		r.append([files[step], prob[0],prob[1],prob[2],prob[3],prob[4],prob[5],prob[6],prob[7] ])
		acc_total += acc
		# if step >= 10:
		# 	break
	print "accuracy:", acc_total*100.0/samples,"total good int:",acc_total
	sub_file = "submission_7_stg1.csv"
	print "image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT"
	np.savetxt(sub_file, r, delimiter=',', fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s")

# coord.join(threads)
# coord.request_stop()
sess.close()