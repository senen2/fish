from PIL import Image
import os
import numpy as np
# test_stg1_fix
nom_path = "../../data/fish/test_stg1/"
path_to_img = "../../data/fish/test_stg1_fix/"
high_img_w = 224
high_img_h = 224
# high_img_w = 896
# high_img_h = 896
print "making files"
total_count2 = 0
files = os.listdir(nom_path)
for file in files:
	print file, total_count2*100.0/len(files)
	total_count2 += 1

	img = Image.open(nom_path + file, 'r')
	img_w, img_h = img.size
	if img_w > high_img_w:
		img.thumbnail((high_img_w, high_img_h), Image.ANTIALIAS)
		img_w, img_h = img.size

	background = Image.new('RGB', (high_img_w, high_img_h), (0, 0, 0))
	bg_w, bg_h = background.size
	offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
	background.paste(img, offset)
	background.save(path_to_img + file, "JPEG")

print
print "img_w:", high_img_w,"img_h",high_img_h ,"total_count:",total_count2
print "end"