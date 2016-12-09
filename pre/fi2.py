"""
Check image -> array -> image
"""
import numpy as np
import os
import Image
import tensorflow as tf
from scipy.misc import toimage
from apifish import *

"""
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )
"""

group = "train"

d0 = directory() + group + "/"
for subdir in os.listdir(directory() + group):
    d1 = d0 + subdir + "/"
    if os.path.isdir(d1):
        print d1
        d2 = directory() + group + "_min/" + subdir + "/"
        for file in os.listdir(d1):
            print file
            img = Image.open(d1 + file)
            #data = np.asarray(img, dtype="int16")
            #img = img.resize((128,72),Image.ANTIALIAS)
            data = (np.asarray(img, dtype="float16") - 128)/128
            print data.shape
            print data
            toimage(data).show()
            z()
