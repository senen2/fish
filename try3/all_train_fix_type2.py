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
path_to_img = "../../data/fish/train-img-type2/"
path_to_label = "../../data/fish/train-lbl-type2/"
paths = [[nom_path + "ALB/",1],[nom_path + "BET/",2],[nom_path + "DOL/",3],
		[nom_path + "LAG/",4],[nom_path + "NoF/",5],[nom_path + "OTHER/",6],
		[nom_path + "SHARK/",7],[nom_path + "YFT/",8]]

print "deleting img..."
files1 = os.listdir(path_to_img)
temp = len(files1)
t = 0
for file in files1:
	os.remove(path_to_img + file)
	print "del img", t*1.0/temp
	t += 1
print "deleting labels..."
files1 = os.listdir(path_to_label)
temp = len(files1)
t = 0
for file in files1:
	os.remove(path_to_label + file)
	print "del label", t*1.0/temp
	t += 1
	
# input_raw()
# high_img_w = 448
# high_img_h = 448
# high_img_w = 1280
# high_img_h = 1280
high_img_w = 224
high_img_h = 224
# high_img_w = 896
# high_img_h = 896
# high_img_w = 32
# high_img_h = 32
c_w = 0
c_h = 0
fix_img_w = 30
fix_img_h =	30
max_count = 3000
move_pixel = 30
# max_count = 200
total_count = 0
c = 0
other_c = 0
print "find high W and H"
for p in paths:
	path = p[0]
	label = p[1]
	count = 0
	files = os.listdir(path)
	for file in files:
		if count == max_count :
			break
		count += 1
		total_count +=1


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
	for file in files:
		if count == max_count :
			break
		print file, round(total_count2*100.0/total_count,2)
		total_count2 += 1

		row = fish_label(label)
		lbl_text = str(row[0]) + "," + str(row[1]) + "," + str(row[2]) + "," + str(row[3]) + "," + str(row[4]) + "," + str(row[5]) + "," + str(row[6]) + "," + str(row[7])
		name, ext = file.split(".")
		# f = open(path_to_label + file + '.txt', 'w')
		# f.write(lbl_text)
		# f.close()

		for t in xrange(4):
			img = Image.open(path + file, 'r')
			if t == 1:
				img = ImageOps.mirror(img)
			if t == 2:
				img = ImageOps.flip(img)
			if t == 3:
				img = ImageOps.mirror(img)
				img = ImageOps.flip(img)
			## making the size require aspec ratio
			for i in xrange(1,5):
				# print i
				# img = Image.open(path + file, 'r')
				img_w, img_h = img.size
				if img_w > fix_img_w:
					img.thumbnail((fix_img_w * i, fix_img_h * i), Image.ANTIALIAS)
					img_w, img_h = img.size
				# img.show()

				for x in xrange(high_img_w):
					if x == max_count:
						# print "x",x
						break
					if x * move_pixel > high_img_w - img_w:
						break
					for y in xrange(high_img_h):
						if y == max_count:
							# print "y",y
							break
						if y * move_pixel > high_img_h - img_h:
							break
						## paste original in black background
						background = Image.new('RGB', (high_img_w, high_img_h), (0, 0, 0))
						bg_w, bg_h = background.size
						offset = (x * move_pixel, y * move_pixel)
						background.paste(img, offset)
						# background.show()
						background.save(path_to_img + name + "_" + str(label) + "_" + str(t) + "_" + str(i) + "_" + str(x) + "_" + str(y) + "." + ext, "JPEG")
						f = open(path_to_label + name + "_" + str(label) + "_" + str(t) + "_" + str(i) + "_" + str(x) + "_" + str(y) + "." + ext + '.txt', 'w')
						f.write(lbl_text)
						f.close()
						other_c += 1

		count += 1

print
print "img_w:", high_img_w,"img_h",high_img_h ,"total_count:",total_count, "other_c:",other_c
print "end"