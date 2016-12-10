'''
Train

best cost 1.16182 epoch 97
best acc 0.676463 epoch 99
epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_heigth img_width dropout
    100      0.0005        5       5          16            16        24        64      36      0.5
Accuracy: 0.676463
Cost 1.17652

best cost 0.208947 epoch 999
best acc 0.998676 epoch 952
epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_heigth img_width dropout
    1000      0.0005        5       5          16            16        24        64      36      0.5
Accuracy: 0.998147
Cost 0.208947
end
[Finished in 54555.1s]

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

images, labels, names = read_images(dirdata + group)
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