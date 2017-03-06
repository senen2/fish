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
parameters = param()
files = os.listdir("../../data/fish/train-fix/")
samples = len(files)
img_path = "../../data/fish/train-fix/"
lbl_path = "../../data/fish/label-train-fix/"
img_queue = []
lbl_queue = []
for file in files:
	img_queue.append(img_path + file)
	lbl_queue.append(lbl_path + file +".txt")

# for i in xrange(len(files)):
# 	print "img_queue", img_queue[i], "lbl_queue", lbl_queue[i]

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

filename_queue = tf.train.string_input_producer(img_queue, num_epochs=1, shuffle=False)

filename_queue_label = tf.train.string_input_producer(lbl_queue, num_epochs=1, shuffle=False)

image_reader = tf.WholeFileReader()
label_reader = tf.WholeFileReader()

image_name , image_file = image_reader.read(filename_queue)
_ , label_file = label_reader.read(filename_queue_label)

ratio_jpg = 1
img_height = img_height/ratio_jpg
img_width = img_width/ratio_jpg

x = tf.image.decode_jpeg(image_file, channels=channels_jpg, ratio=ratio_jpg)
x.set_shape([img_height, img_width, channels_jpg])

x = tf.cast(x, tf.float32)
x = tf.nn.l2_normalize(x, dim=0)

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
# W_conv7 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 32, cv_all_channels * 64])
# b_conv7 = bias_variable([cv_all_channels * 64])
# W_conv8 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 64, cv_all_channels * 128])
# b_conv8 = bias_variable([cv_all_channels * 128])
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
pred2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print "pred",pred
print "y", y
print "h_conv1", h_conv1
print "h_pool1", h_pool1
print "h_conv2", h_conv2
print "h_pool2", h_pool2
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-20), reduction_indices=[1]))
cost_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred2,labels=tf.nn.softmax(y)))
# cost_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=tf.argmax(y, 1)))
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred_arg2 = tf.argmax(pred, 1)
y_arg2 = tf.argmax(y, 1)
auc2, update_op_auc2 = tf.contrib.metrics.streaming_auc(pred, y, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, curve='ROC', name=None)

print "y",y
print "pred", pred

init = tf.global_variables_initializer()
acc_total = 0.0

alb = 0
bet = 0
dol = 0
lag = 0
nof = 0
other = 0
shark = 0
yft = 0

alb2 = 1e-20
bet2 = 1e-20
dol2 = 1e-20
lag2 = 1e-20
nof2 = 1e-20
other2 = 1e-20
shark2 = 1e-20
yft2 = 1e-20

alb_total = 85
bet_total = 85
dol_total = 85
lag_total = 67
nof_total = 85
other_total = 85
shark_total = 85
yft_total = 85

total_cost2 = 0
total_cost3 = 0
test_acc = 0
with tf.Session() as sess:
	sess.run(tf.local_variables_initializer())
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	# sleep = 5
	# print "sleep:",sleep
	# time.sleep(sleep)
	# r = []
	# r.append(["image","ALB","BET","DOL","LAG","NoF","OTHER","SHARK","YFT"])
	for step in xrange(samples):
		print "step:",step + 1
		prob,correct_prediction2,acc,pred_arg,y_arg,y_tes,update_op_auc,auc,cost2,cost3,img_name = sess.run([pred,correct_prediction,accuracy,pred_arg2,y_arg2,y,update_op_auc2,auc2,cost,cost_1,image_name],{ 
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
		    # W_conv8:features["W_conv8"],
		    # b_conv8:features["b_conv8"][0],
		    W_fc1:features["W_fc1"],
		    b_fc1:features["b_fc1"][0],
		    W_fc2:features["W_fc2"],
		    b_fc2:features["b_fc2"][0]                          
		  })
		prob = prob[0]
		# prob = prob[0]
		# print prob
		y_arg = y_arg[0]
		pred_arg = pred_arg[0]
		correct_prediction2 = correct_prediction2[0]
		# print "cost2",cost2
		total_cost2 += cost2
		total_cost3 += cost3
		print "auc:", auc,"update_op_auc:",update_op_auc
		# print
		# print "False"
		# print "y_arg", y_arg
		# print "pred_arg", pred_arg
		# print "correct_prediction:", correct_prediction2
		# if pred_arg == y_arg:
		# 	test_acc += 1
		# 	print "img_name",img_name[30:],"pred_arg", pred_arg,"y_arg:", y_arg, "ok","y_tes", y_tes,"prob",prob
		# else:
		# 	print "img_name",img_name[30:],"pred_arg", pred_arg,"y_arg:", y_arg, "false","y_tes", y_tes,"prob",prob
		if correct_prediction2 == True:
			# print
			# print "True"
			# print "y_arg", y_arg
			# print "pred_arg", pred_arg
			if y_arg == 0:
				alb += 1
			elif y_arg == 1:
			    bet += 1
			elif y_arg == 2:
			    dol += 1
			elif y_arg == 3:
			    lag += 1
			elif y_arg == 4:
			    nof += 1
			elif y_arg == 5:
			    other += 1
			elif y_arg == 6:
			    shark += 1
			elif y_arg == 7:
			    yft += 1
		if y_arg == y_arg:
			# print
			# print "true test"
			# print "y_arg", y_arg
			# print "pred_arg", pred_arg
			if y_arg == 0:
				alb2 += 1
			elif y_arg == 1:
			    bet2 += 1
			elif y_arg == 2:
			    dol2 += 1
			elif y_arg == 3:
			    lag2 += 1
			elif y_arg == 4:
			    nof2 += 1
			elif y_arg == 5:
			    other2 += 1
			elif y_arg == 6:
			    shark2 += 1
			elif y_arg == 7:
			    yft2 += 1
		acc_total += acc
		# if step >= 30:
		# 	break
	auc = auc2.eval()
	total_s = alb2 + bet2 + dol2 + lag2 + nof2 + other2 + shark2 + yft2
	acc_label = (alb*100.0/alb_total + bet*100.0/bet_total + dol*100.0/dol_total + lag*100.0/lag_total +
				nof*100.0/nof_total + other*100.0/other_total + shark*100.0/shark_total + yft*100.0/yft_total) / 8.0
	print
	print "auc:", auc
	print "accuracy:", round(acc_total*100.0/total_s,2),"total good int:",acc_total
	print "acc by label:", round(acc_label,2)
	print "cost2:", total_cost2*1.0/total_s, "cost3:", total_cost3*1.0/total_s
	print
	print "ALB  ", round(alb*100.0/alb2,2),"total good int:",alb, "of", alb2
	print "BET  ", round(bet*100.0/bet2,2),"total good int:",bet, "of", bet2
	print "DOL  ", round(dol*100.0/dol2,2),"total good int:",dol, "of", dol2
	print "LAG  ", round(lag*100.0/lag2,2),"total good int:",lag, "of", lag2
	print "NoF  ", round(nof*100.0/nof2,2),"total good int:",nof, "of", nof2
	print "OTHER", round(other*100.0/other2,2),"total good int:",other, "of", other2
	print "SHARK", round(shark*100.0/shark2,2),"total good int:",shark, "of", shark2
	print "YFT  ", round(yft*100.0/yft2,2),"total good int:",yft, "of", yft2
	print

	# print
	# print "test"
	# print "ALB  ", round(alb2*100.0/alb_total,2),"total good int:",alb2, "of", alb_total
	# print "BET  ", round(bet2*100.0/bet_total,2),"total good int:",bet2, "of", bet_total
	# print "DOL  ", round(dol2*100.0/dol_total,2),"total good int:",dol2, "of", dol_total
	# print "LAG  ", round(lag2*100.0/lag_total,2),"total good int:",lag2, "of", lag_total
	# print "NoF  ", round(nof2*100.0/nof_total,2),"total good int:",nof2, "of", nof_total
	# print "OTHER", round(other2*100.0/other_total,2),"total good int:",other2, "of", other_total
	# print "SHARK", round(shark2*100.0/shark_total,2),"total good int:",shark2, "of", shark_total
	# print "YFT  ", round(yft2*100.0/yft_total,2),"total good int:",yft2, "of", yft_total

sess.close()