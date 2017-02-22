'''
Created on Nov 16, 2016

@author: botpi
'''

def param():
    return { 
             "cv1_size": 5
           , "cv2_size": 5
           , "cv1_channels": 32
           , "cv2_channels": 64
           , "hidden": 1024
           , "img_width": 224
           , "img_height": 224
           # , "hidden": 32
           # , "img_width": 32
           # , "img_height": 32
           , "categories": 8
           , "learning_rate": 1e-4
           , "dropout": 0.5
          }
