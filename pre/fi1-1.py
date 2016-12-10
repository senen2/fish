"""
Create mat file with image reduced for test
"""
import numpy as np
import os
import scipy.io, scipy.stats
import Image
from apifish import *

print "begin"
group = "test_stg1"
dirdata = "../../data/fish/"

d1 = directory() + group + "/"
t = []
z = np.zeros(8)
for file in os.listdir(d1):
    if not os.path.isdir(d1 + file):
        #print file
        img = Image.open(d1 + file)
        img = img.resize((64,36),Image.ANTIALIAS)
        #img.save(d2 + file)
        d = {}
        d["x"] = (np.asarray(img, dtype="float16") - 128)/128
        d["y"] = z
        d["name"] = file
        t.append(d)

d = {}
d["t"] = t
scipy.io.savemat(dirdata + group + " 64x36", d, do_compression=True)    
print "end"            