"""
Create mat file with image reduced for train
"""
import numpy as np
import os
import scipy.io, scipy.stats
import Image
from apifish import *

print "begin"
group = "train"
dirdata = "../../data/fish/"

d0 = directory() + group + "/"
t = []
for subdir in os.listdir(directory() + group):
    d1 = d0 + subdir + "/"
    if os.path.isdir(d1):
        y = fish_label(subdir)
        #print d1
        d2 = directory() + group + "_min/" + subdir + "/"
        for file in os.listdir(d1):
            if not os.path.isdir(d1 + file):
                #print file
                img = Image.open(d1 + file)
                img = img.resize((128,72),Image.ANTIALIAS)
                #img.save(d2 + file)
                d = {}
                d["x"] = (np.asarray(img, dtype="float16") - 128)/128
                d["y"] = y
                t.append(d)

d = {}
d["t"] = t
scipy.io.savemat(dirdata + group + " 128x72", d, do_compression=True)
print "end"    
            