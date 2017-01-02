'''
Create submission

@author: botpi
'''
import tensorflow as tf
import numpy as np
from model_eval import *
import scipy.io
from apifish import *
from params import param

print "begin"
group = "train"
group = "test"
sub_file = "submission_1.csv"
    
r = []
r.append(["File", "Class"])

features = scipy.io.loadmat(datadiresp + group +"_resp")
#images, labels, names = read_images("%s %s_new" % (group, ii))
images, labels, names = read_images(dirdata + group)
parameters = param()

prob = eval_conv(images, parameters, features)

p = 0
for i in xrange(len(images)):
    r.append([names[i] + ".mat", prob[i][1] ])

print "image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT"
np.savetxt(sub_file, r, delimiter=',', fmt="%s,%s")

print "end"