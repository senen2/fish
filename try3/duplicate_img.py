from PIL import Image
import os
import hashlib
# import numpy as np

nom_path = "../../data/fish/train/"
# path_to_img = "../../data/fish/train-fix/"
# path_to_label = "../../data/fish/label-train-fix/"
paths = [[nom_path + "ALB",1],[nom_path + "BET",2],[nom_path + "DOL",3],
		[nom_path + "LAG",4],[nom_path + "NoF",5],[nom_path + "OTHER",6],
		[nom_path + "SHARK",7],[nom_path + "YFT",8]]

total = 0
alb_total = 0
bet_total = 0
dol_total = 0
lag_total = 0
nof_total = 0
other_total = 0
shark_total = 0
yft_total = 0
for p in paths:
	path = p[0]
	label = p[1]
	files = os.listdir(path)
	total += len(files)
	if label == 1:
		alb_total += len(files)
	elif label == 2:
		bet_total += len(files)
	elif label == 3:
		dol_total += len(files)
	elif label == 4:
		lag_total += len(files)
	elif label == 5:
		nof_total += len(files)
	elif label == 6:
		other_total += len(files)
	elif label == 7:
		shark_total += len(files)
	elif label == 8:
		yft_total += len(files)
		
print "ALB  ", alb_total
print "BET  ", bet_total
print "DOL  ", dol_total
print "LAG  ", lag_total
print "NoF  ", nof_total
print "OTHER", other_total
print "SHARK", shark_total
print "YFT  ", yft_total
print "total", total

count = 0
dup = 0
names = []
for p in paths:
	path = p[0]
	label = p[1]
	files = os.listdir(path)
	for file in files:
		print file, count*100.0/total, "dup", dup
		for p2 in paths:
			path2 = p2[0]
			label2 = p2[1]
			if label == label2:
				continue
			files2 = os.listdir(path2)
			for file2 in files2:
				img1 = open(path + "/" + file).read()
				img2 = open(path2 + "/" + file2).read()
				hash_img1 = hashlib.md5(img1).hexdigest()
				hash_img2 = hashlib.md5(img2).hexdigest()
				if hash_img1 == hash_img2:
					dup += 1
					names.append([file,file2])
		count += 1
print "dup", dup
print "check dup" , len(names)
print names