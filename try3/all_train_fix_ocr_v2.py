from PIL import Image,ImageOps
import os
import numpy as np

def fish_label(name):
    a = np.zeros(8)
    if name == 1:
        a[0] = 1
    elif name == 2:
        a[1] = 1
    elif name == 3:
        a[2] = 1
    elif name == 4:
        a[3] = 1
    elif name == 5:
        a[4] = 1
    elif name == 6:
        a[5] = 1
    elif name == 7:
        a[6] = 1
    else:
        a[7] = 1
        
    return a

nom_path = "../../data/fish/train-type2/"
path_to_img_train = "../../data/fish/train-img-type2/"
path_to_label_train = "../../data/fish/train-lbl-type2/"
path_to_img_test = "../../data/fish/test-img-type2/"
path_to_label_test = "../../data/fish/test-lbl-type2/"
paths = [[nom_path + "ALB/",1],[nom_path + "BET/",2],[nom_path + "DOL/",3],
		[nom_path + "LAG/",4],[nom_path + "NoF/",5],[nom_path + "OTHER/",6],
		[nom_path + "SHARK/",7],[nom_path + "YFT/",8]]

print "deleting train img..."
files1 = os.listdir(path_to_img_train)
temp = len(files1)
t = 0
for file in files1:
	os.remove(path_to_img_train + file)
	print "del img", t*1.0/temp
	t += 1

print "deleting test img..."
files1 = os.listdir(path_to_img_test)
temp = len(files1)
t = 0
for file in files1:
	os.remove(path_to_img_test + file)
	print "del img", t*1.0/temp
	t += 1

print "deleting train labels..."
files1 = os.listdir(path_to_label_train)
temp = len(files1)
t = 0
for file in files1:
	os.remove(path_to_label_train + file)
	print "del label", t*1.0/temp
	t += 1

print "deleting test labels..."
files1 = os.listdir(path_to_label_test)
temp = len(files1)
t = 0
for file in files1:
	os.remove(path_to_label_test + file)
	print "del label", t*1.0/temp
	t += 1

# input_raw()
# high_img_w = 448
# high_img_h = 448
# high_img_w = 1280
# high_img_h = 1280
# high_img_w = 224
# high_img_h = 224
high_img_w = 112
high_img_h = 112
# high_img_w = 896
# high_img_h = 896
# high_img_w = 32
# high_img_h = 32
c_w = 0
c_h = 0
max_count = 3000
test_split = 0.2
test_list = {}
total_count = 0
c = 0
train_count = 0
test_count = 0
print "find high W and H"
for p in paths:
	path = p[0]
	label = p[1]
	count = 0
	files = os.listdir(path)
	total_count += len(files)
	to_test = int(len(files) * test_split)
	test_list[str(label)] = to_test

print "img_w:", high_img_w,"img_h",high_img_h ,"total_count:",total_count
print
print "making files"
total_count2 = 0
for p in paths:
	path = p[0]
	label = p[1] 
	count = 0
	print "dir", c,"of", len(paths)
	c += 1
	files = os.listdir(path)
	total_files = len(files)
	for file in files:
		if count >= total_files - test_list[str(label)]:
			path_to_img = path_to_img_test
			path_to_label = path_to_label_test
		else:
			path_to_img = path_to_img_train
			path_to_label = path_to_label_train
		print file, round(total_count2*100.0/total_count,2)
		total_count2 += 1

		row = fish_label(label)
		lbl_text = str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(row[4]) + "," + str(row[5]) + "," + str(row[6]) + "," + str(row[7])
		name, ext = file.split(".")

		for t in xrange(8):
			img = Image.open(path + file, 'r')
			# print t
			if t == 1:
				img = ImageOps.mirror(img)
			if t == 2:
				img = ImageOps.flip(img)
			if t == 3:
				img = ImageOps.mirror(img)
				img = ImageOps.flip(img)
			if t == 4:
				img = img.rotate(90)
			if t == 5:
				img = img.rotate(90)
				img = ImageOps.mirror(img)
			if t == 6:
				img = img.rotate(270)
			if t == 7:
				img = img.rotate(270)
				img = ImageOps.mirror(img)
			## making the size require aspec ratio
			img_w, img_h = img.size
			if img_w > high_img_w or img_h > high_img_h:
				img.thumbnail((high_img_w, high_img_h), Image.ANTIALIAS)
				img_w, img_h = img.size

			# ## paste original in black background
			# background = Image.new('RGB', (high_img_w, high_img_h), (0, 0, 0))
			# bg_w, bg_h = background.size
			# offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
			# background.paste(img, offset)
			# background.save(path_to_img + name + "_" + str(label) + "_" + str(t) + "." + ext, "JPEG")
			img.save(path_to_img + name + "_" + str(label) + "_" + str(t) + "." + ext, "JPEG")
			f = open(path_to_label + name + "_" + str(label) + "_" + str(t) + "." + ext + '.txt', 'w')
			f.write(lbl_text)
			f.close()
			if count >= total_files - test_list[str(label)]:
				test_count += 1
			else:
				train_count += 1

		count += 1

print
print "img_w:", high_img_w,"img_h",high_img_h ,"total_count:",total_count, "train count:",train_count,"test count:",test_count
print "test dict:",test_list
print "end"