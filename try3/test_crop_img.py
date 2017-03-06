from PIL import Image
import os
# import numpy as np

nom_path = "../../data/fish/train/"
paths = [[nom_path + "ALB/",1],[nom_path + "BET/",2],[nom_path + "DOL/",3],
		[nom_path + "LAG/",4],[nom_path + "NoF/",5],[nom_path + "OTHER/",6],
		[nom_path + "SHARK/",7],[nom_path + "YFT/",8]]

high_img_w = 112
high_img_h = 112
box_size = 160
move_pixel = 5
count_max = 10

samples = 0
test = 0
step = 0
cat = 0
for p in paths:
	path = p[0]
	label = p[1]
	files = os.listdir(path)
	samples += len(files)
	cat += 1
	for file in files:
		img = Image.open(path + file, 'r')
		img = img.convert("L")
		# img.show()
		img_w, img_h = img.size
		step += 1
		# print step
		for x in xrange(img_w):
			if x*move_pixel > count_max:
				# print "break x"
				break
			if x*move_pixel > img_w - box_size:
				break
			for y in xrange(img_h):
				if y*move_pixel > count_max:
					# print "break y"
					break
				if y*move_pixel > img_h - box_size:
					break
				box = (x*move_pixel,y*move_pixel,(x*move_pixel)+box_size,(y*move_pixel)+box_size)
				temp = img.crop(box)
				temp.thumbnail((high_img_w, high_img_h), Image.ANTIALIAS)
				test += 1
		# temp.show()
		# img.show()
		break
	break

print "cat:",cat,"images:", step,"how many by img:",test
# 3104 - 3105 img change box_size = 100
# test[3104].show()
# test[0].show()
# img.show()