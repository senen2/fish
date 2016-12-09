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
sub_file = "submission_conv_48.csv"
    
r = []
r.append(["File", "Class"])

for i in range(3):
    patient = i+1
    features = scipy.io.loadmat("resp_%s_new" % patient)
    #images, labels, names = read_images("%s %s_new" % (group, ii))
    images, labels, names = read_images("%s_%s_new" % (group, patient))
    parameters = param(patient)
    
    prob = eval_conv(images, parameters, features)
    
    p = 0
    for i in xrange(len(names)):
#         if prob[i][1] > 0.999:
#             r.append([names[i] + ".mat", 1])
#         elif prob[i][1] < 0.001:
#             r.append([names[i] + ".mat", 0])
#         else:
#             r.append([names[i] + ".mat", prob[i][1] ])

        r.append([names[i] + ".mat", prob[i][1] ])
        if prob[i][1]>0.5:
            p += 1

    print "positives", p, "totals", len(names)
    if group == "train":
        print "AUC", auc(labels, prob)

print "gran total", len(r)
np.savetxt(sub_file, r, delimiter=',', fmt="%s,%s")

print "end"