'''
Train
'''
import tensorflow as tf
import numpy as np
from model_eval import *
import scipy.io
from model_train import *
from params import param
from apifish import read_images_balanced

print "begin"
patient = 1
group = "train_%s_new" % patient
parameters = param(patient)
training_epochs = 2

images, labels, names = read_images(group)
#print images.shape, labels.shape, len(names)
#images, labels, names = read_images_balanced(group, 2)
print images.shape, labels.shape, len(names)
features, prob, acc, cost = train_tf(images, labels, parameters, training_epochs=training_epochs)

print "Accuracy:", "epochs", "learning rate", "cv1 size", "cv2 size", "cv1 channels", "cv2channels", "hidden", "img resize", "dropout"
print acc, training_epochs, parameters["learning_rate"], parameters["cv1_size"], parameters["cv2_size"], parameters["cv1_channels"], parameters["cv2_channels"], parameters["hidden"], parameters["img_resize"], parameters["dropout"]
print "AUC", auc(labels, prob), "Cost", cost, "patient", patient, "con todo"

scipy.io.savemat("resp_%s_new" % patient, features, do_compression=True)    
print "end"