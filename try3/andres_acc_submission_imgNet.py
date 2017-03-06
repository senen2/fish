'''
Model Evaluation

@author: botpi
'''
import tensorflow as tf
import numpy as np
from apifish import *
import scipy.io
from params import param
import os

features = scipy.io.loadmat("resp_conv16_pool5_imgNet_chan_1")
sub_file = "submission_imgNet_10_stg1.csv"
parameters = param()
# files = os.listdir("../../data/fish/train-fix/")
files = os.listdir("../../data/fish/test_stg1_fix/")
samples = len(files)

cv1_size = parameters["cv1_size"]
cv2_size = parameters["cv2_size"]
cv1_channels = parameters["cv1_channels"]
cv2_channels = parameters["cv2_channels"]
hidden = parameters["hidden"]
img_width = parameters["img_width"]
img_height = parameters["img_height"]
categories = parameters["categories"]
cv_all_size = 7
cv_all_channels = 1
last_img_size = 7
channels_jpg = 1

filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("../../data/fish/test_stg1_fix/*.jpg"), shuffle=False)

image_reader = tf.WholeFileReader()

image_name , image_file = image_reader.read(filename_queue)

ratio_jpg = 1
img_height = img_height/ratio_jpg
img_width = img_width/ratio_jpg

x = tf.image.decode_jpeg(image_file, channels=channels_jpg, ratio=ratio_jpg)
x.set_shape([img_height, img_width, channels_jpg])

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

x = tf.reshape(tf.cast(x, tf.float32), [-1,img_width,img_height,channels_jpg])

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

h_fc1 = tf.nn.relu(tf.matmul(h_pool_last_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, 1)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, 1)
pred = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
print pred
print "h_conv1", h_conv1
print "h_conv2", h_conv2
print "h_pool2", h_pool2
# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
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
	r = []
	r.append(["image","ALB","BET","DOL","LAG","NoF","OTHER","SHARK","YFT"])
	for step in xrange(samples):
		print "step:",step
		prob, img_name,pred_arg = sess.run([pred,image_name,pred_arg2],{ 
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
		    W_conv6:features["W_conv6"],
		    b_conv6:features["b_conv6"][0],
		    W_conv7:features["W_conv7"],
		    b_conv7:features["b_conv7"][0],
		    W_conv8:features["W_conv8"],
		    b_conv8:features["b_conv8"][0],
		    W_conv9:features["W_conv9"],
		    b_conv9:features["b_conv9"][0],
		    W_conv10:features["W_conv10"],
		    b_conv10:features["b_conv10"][0],
		    W_fc1:features["W_fc1"],
		    b_fc1:features["b_fc1"][0],
		    W_fc2:features["W_fc2"],
		    b_fc2:features["b_fc2"][0],
		    W_fc3:features["W_fc3"],
		    b_fc3:features["b_fc3"][0]
		  })
		prob = prob[0]
		img = img_name[30:]
		# rest_all = 0.0
		# prob[0],prob[1],prob[2],prob[3],prob[4],prob[5],prob[6],prob[7] = rest_all,rest_all,rest_all,rest_all,rest_all,rest_all,rest_all,rest_all
		# prob[pred_arg] = 0.99
		r.append([str(img), prob[0],prob[1],prob[2],prob[3],prob[4],prob[5],prob[6],prob[7]])
		# if step >= 10:
		# 	break
	print "image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT"
	np.savetxt(sub_file, r, delimiter=',', fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s")

sess.close()
print sub_file