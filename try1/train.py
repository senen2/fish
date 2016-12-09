'''
Train
'''
import tensorflow as tf
import numpy as np
from model_eval import *
import scipy.io
from model_train import *
from params import param
from apifish import read_images

print "begin"
dirdata = "../../data/fish/"
group = "train 64x36"
#group = "LAG 64x36"
training_epochs = 100

images, labels = read_images(dirdata + group)
parameters = param()
#print images.shape, labels.shape, len(names)
#images, labels, names = read_images_balanced(group, 2)
print images.shape, labels.shape
features, prob, acc, cost = train_tf(images, labels, parameters, training_epochs=training_epochs)

print "epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_heigth img_width dropout"
print ("    %s      %s        %s       %s          %s            %s        %s        %s      %s      %s" 
    % (training_epochs, parameters["learning_rate"], parameters["cv1_size"], parameters["cv2_size"], parameters["cv1_channels"]
       , parameters["cv2_channels"], parameters["hidden"], parameters["img_height"], parameters["img_width"], parameters["dropout"]))

print "Accuracy:", acc #, "Accuracy test:", test_acc
print "Cost", cost

scipy.io.savemat(dirdata + group + "_resp", features, do_compression=True)    
print "end"