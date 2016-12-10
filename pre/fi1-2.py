"""
Create mat file with image reduced for tiny tiny sample
"""
import numpy as np
import os
import scipy.io, scipy.stats
import Image
from apifish import *

print "begin"
group = "train"
subdir = "LAG"
dirdata = "../../data/fish/"

d1 = directory() + group + "/" + subdir + "/"
t = []
z = fish_label(subdir)
for file in os.listdir(d1):
    if not os.path.isdir(d1 + file):
        #print file
        img = Image.open(d1 + file)
        img = img.resize((128,72),Image.ANTIALIAS)
        #img.save(d2 + file)
        d = {}
        d["x"] = (np.asarray(img, dtype="float16") - 128)/128
        d["y"] = z
        d["name"] = file        
        t.append(d)

d = {}
d["t"] = t
scipy.io.savemat(dirdata + subdir + " 128x72", d, do_compression=True)    
print "end"            