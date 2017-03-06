'''
Model Evaluation

@author: botpi
'''
import tensorflow as tf
import numpy as np
from apifish import *
import scipy.io
from params_ocr import param
import time
import os
from PIL import Image
import matplotlib.pyplot as plt

def fish_label(name):
    a = np.zeros(8)
    if name == 1:
        a[0] = 1
    elif name == 2:
        a[1] = 1
    elif name == 3:
        a[2] = 1
    elif name == 4:
        a[3] = 1
    elif name == 5:
        a[4] = 1
    elif name == 6:
        a[5] = 1
    elif name == 7:
        a[6] = 1
    else:
        a[7] = 1
        
    return a

nom_path = "../../data/fish/train/"
paths = [[nom_path + "ALB/",1],[nom_path + "BET/",2],[nom_path + "DOL/",3],
		[nom_path + "LAG/",4],[nom_path + "NoF/",5],[nom_path + "OTHER/",6],
		[nom_path + "SHARK/",7],[nom_path + "YFT/",8]]

high_img_w = 112
high_img_h = 112

samples = 0
for p in paths:
	path = p[0]
	label = p[1]
	files = os.listdir(path)
	samples += len(files)
	# for i in xtange(files):
	# 	img = Image.open(path + file, 'r')

# print "total_count",samples

features = scipy.io.loadmat("resp_ocr_conv4_112_chan_1")
parameters = param()
# files = os.listdir("../../data/fish/train-fix2/")
# samples = len(files)
# img_path = "../../data/fish/train-fix2/"
# lbl_path = "../../data/fish/label-train-fix2/"
# img_queue = []
# lbl_queue = []
# for file in files:
# 	img_queue.append(img_path + file)
# 	lbl_queue.append(lbl_path + file +".txt")

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
last_img_size = 28
channels_jpg = 1

# filename_queue = tf.train.string_input_producer(img_queue, num_epochs=1, shuffle=False)

# filename_queue_label = tf.train.string_input_producer(lbl_queue, num_epochs=1, shuffle=False)

# image_reader = tf.WholeFileReader()
# label_reader = tf.WholeFileReader()

# _ , image_file = image_reader.read(filename_queue)
# _ , label_file = label_reader.read(filename_queue_label)

# ratio_jpg = 1
# img_height = img_height/ratio_jpg
# img_width = img_width/ratio_jpg

# x = tf.image.decode_jpeg(image_file, channels=channels_jpg, ratio=ratio_jpg)
# x.set_shape([img_height, img_width, channels_jpg])

# record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
# col1, col2, col3, col4, col5, col6, col7, col8 = tf.decode_csv(label_file, record_defaults=record_defaults)
# y = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8])
# y = tf.pack([y])

# x = tf.placeholder(tf.float32, shape=[None, img_width, img_height, channels_jpg])
x = tf.placeholder(tf.float32, shape=[None, img_width, img_height, channels_jpg])
y = tf.placeholder(tf.float32, shape=[None, categories])

W_conv1 = weight_variable([cv_all_size, cv_all_size, channels_jpg, cv_all_channels])
b_conv1 = bias_variable([cv_all_channels])
W_conv2 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels * 2])
b_conv2 = bias_variable([cv_all_channels * 2])
# W_conv3 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 2, cv_all_channels * 4])
# b_conv3 = bias_variable([cv_all_channels * 4])
# W_conv4 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 4, cv_all_channels * 8])
# b_conv4 = bias_variable([cv_all_channels * 8])
# W_conv5 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 16])
# b_conv5 = bias_variable([cv_all_channels * 16])
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

W_fc1 = weight_variable([last_img_size * last_img_size * cv_all_channels * 2, hidden])
b_fc1 = bias_variable([hidden])
W_fc2 = weight_variable([hidden, categories])
b_fc2 = bias_variable([categories])

x = tf.reshape(tf.cast(x, tf.float32), [-1,img_width,img_height,channels_jpg])

# conv
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
# h_pool3 = max_pool_2x2(h_conv3)
# h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
# h_pool4 = max_pool_2x2(h_conv4)
# h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
# h_pool5 = max_pool_2x2(h_conv5)
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

h_pool_last_flat = tf.reshape(h_pool2, [-1, last_img_size * last_img_size  * cv_all_channels * 2])

h_fc1 = tf.nn.relu(tf.matmul(h_pool_last_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, 1)
pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
pred2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print pred
print "h_conv1", h_conv1
print "h_pool1", h_pool1
print "h_conv2", h_conv2
print "h_pool2", h_pool2
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-20), reduction_indices=[1]))
cost_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
# cost_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred_arg2 = tf.argmax(pred, 1)
y_arg2 = tf.argmax(y, 1)
# auc2, update_op_auc2 = tf.contrib.metrics.streaming_auc(predictions=pred, labels=y, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, curve='ROC', name=None)

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

alb_total = 1719
bet_total = 200
dol_total = 117
lag_total = 67
nof_total = 465
other_total = 299
shark_total = 176
yft_total = 734

total_cost2 = 0
total_cost3 = 0

high_img_w = 112
high_img_h = 112
box_size = 300
move_pixel = 30
count_max = 10
test = 0

step = 0
# max_step = 8
step_lbl = 0
max_step_lbl = 1
# step_img = 0
max_step_img = 30
with tf.Session() as sess:
	# sess.run(tf.local_variables_initializer())
	sess.run(init)
	# sess.run([pred],feed_dict={x:pix,y:row,
	# 	W_conv1:features["W_conv1"],
	#     b_conv1:features["b_conv1"][0],
	#     W_conv2:features["W_conv2"],
	#     b_conv2:features["b_conv2"][0],
	#     W_conv3:features["W_conv3"],
	#     b_conv3:features["b_conv3"][0],
	#     W_conv4:features["W_conv4"],
	#     b_conv4:features["b_conv4"][0],
	#     # W_conv5:features["W_conv5"],
	#     # b_conv5:features["b_conv5"][0],
	#     # W_conv6:features["W_conv6"],
	#     # b_conv6:features["b_conv6"][0],
	#     # W_conv7:features["W_conv7"],
	#     # b_conv7:features["b_conv7"][0],
	#     # W_conv8:features["W_conv8"],
	#     # b_conv8:features["b_conv8"][0],
	#     W_fc1:features["W_fc1"],
	#     b_fc1:features["b_fc1"][0],
	#     W_fc2:features["W_fc2"],
	#     b_fc2:features["b_fc2"][0]
	# 	})
	# coord = tf.train.Coordinator()
	# threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	for p in paths:
		# if step_lbl >= max_step_lbl:
		# 	# print "break:", step
		# 	break
		path = p[0]
		label = p[1]
		row = fish_label(label)
		files = os.listdir(path)
		step_img = 0
		# step_lbl += 1
		for file in files:
			if step_img >= max_step_img:
				# print "break:", step
				break
			img = Image.open(path + file, 'r')
			if channels_jpg == 1:
				img = img.convert("L")
			img_w, img_h = img.size
			step += 1
			step_img += 1
			print "step:",step
			# print "name:",file,"--path:" , path,"--label:", label ,"size" ,img.size
			by_img = 0
			best_prob = 0.0
			best_prob_full = np.zeros(8)
			best_prob_full[4] = 1.0
			best_y_arg = 4
			best_pred_arg = 4
			best_correct_prediction2 = False
			for _x in xrange(img_w):
				# if _x*move_pixel > count_max:
				# 	break
				if _x*move_pixel > (img_w-1) - box_size:
					break
				for _y in xrange(img_h):
					# if _y*move_pixel > count_max:
					# 	break
					if _y*move_pixel > (img_h-1) - box_size:
						break
					box = (_x*move_pixel,_y*move_pixel,(_x*move_pixel)+box_size,(_y*move_pixel)+box_size)
					temp = img.crop(box)
					temp.thumbnail((high_img_w, high_img_h), Image.ANTIALIAS)
					# plt.gray()
					# plt.imshow(np.array(temp))
					# plt.show()
					pix = np.array(temp).reshape(1,high_img_w, high_img_h,channels_jpg)
					row = row.reshape(1,categories)
					# test += 1
					# print pix.shape
					# print row.shape
					# prob, correct_prediction2 = sess.run([pred, correct_prediction],feed_dict={
					#     x:pix,
					# 	y:row})
					prob, correct_prediction2, pred_arg, y_arg, cost2, cost3 = sess.run([pred, correct_prediction, pred_arg2, y_arg2, cost, cost_1],feed_dict={
					    x:pix,
						y:row,
					    W_conv1:features["W_conv1"],
					    b_conv1:features["b_conv1"][0],
					    W_conv2:features["W_conv2"],
					    b_conv2:features["b_conv2"][0],
					    # W_conv3:features["W_conv3"],
					    # b_conv3:features["b_conv3"][0],
					    # W_conv4:features["W_conv4"],
					    # b_conv4:features["b_conv4"][0],
					    # W_conv5:features["W_conv5"],
					    # b_conv5:features["b_conv5"][0],
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
					y_arg_2 = y_arg[0]
					pred_arg_2 = pred_arg[0]
					correct_prediction_2 = correct_prediction2[0]
					for po in prob:
						if best_prob < po:
							# print po
							# print prob
							# print y_arg_2
							# print pred_arg_2
							# print correct_prediction_2
							best_prob_full = prob
							# best_prob = 0.0
							best_y_arg = y_arg_2
							best_pred_arg = pred_arg_2
							best_correct_prediction2 = correct_prediction_2
							total_cost2 += cost2
							total_cost3 += cost3
							best_img = np.array(temp)
			plt.gray()
			plt.imshow(best_img)
			plt.show()
			if best_correct_prediction2 == True:
			# if correct_prediction2 == True and by_img == 0:
				print best_prob_full
				print best_y_arg
				# print best_pred_arg
				# print best_correct_prediction2
				print file
				plt.gray()
				plt.imshow(best_img)
				plt.show()
				by_img = 1
				acc_total += 1
				if best_y_arg == 0:
					alb += 1
				elif best_y_arg == 1:
				    bet += 1
				elif best_y_arg == 2:
				    dol += 1
				elif best_y_arg == 3:
				    lag += 1
				elif best_y_arg == 4:
				    nof += 1
				elif best_y_arg == 5:
				    other += 1
				elif best_y_arg == 6:
				    shark += 1
				elif best_y_arg == 7:
				    yft += 1
			if y_arg == y_arg:
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

	# auc = auc2.eval()
	total_s = alb2 + bet2 + dol2 + lag2 + nof2 + other2 + shark2 + yft2
	acc_label = (alb*100.0/alb2 + bet*100.0/bet2 + dol*100.0/dol2 + lag*100.0/lag2 +
				nof*100.0/nof2 + other*100.0/other2 + shark*100.0/shark2 + yft*100.0/yft2) / 8.0
	print
	# print "test:", test
	print "accuracy:", round(acc_total*100.0/total_s,2),"total good int:",int(acc_total), "of",int(total_s)
	print "acc by label:", round(acc_label,2)
	print "cost2", total_cost2*1.0/total_s, "cost3", total_cost3*1.0/total_s
	
	print
	print "ALB  ", round(alb*100.0/alb2,2),"% total good int:",alb, "of", int(alb2)
	print "BET  ", round(bet*100.0/bet2,2),"% total good int:",bet, "of", int(bet2)
	print "DOL  ", round(dol*100.0/dol2,2),"% total good int:",dol, "of", int(dol2)
	print "LAG  ", round(lag*100.0/lag2,2),"% total good int:",lag, "of", int(lag2)
	print "NoF  ", round(nof*100.0/nof2,2),"% total good int:",nof, "of", int(nof2)
	print "OTHER", round(other*100.0/other2,2),"% total good int:",other, "of", int(other2)
	print "SHARK", round(shark*100.0/shark2,2),"% total good int:",shark, "of", int(shark2)
	print "YFT  ", round(yft*100.0/yft2,2),"% total good int:",yft, "of", int(yft2)
	print

# coord.request_stop()
# coord.join(threads)
sess.close()
