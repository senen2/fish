from PIL import Image
import os

# cwd = os.getcwd()
path = "../../data/fish/train/ALB/"
path_to = "../../data/fish/train-fix/"
files = os.listdir(path)
high_img_w = 0
high_img_h = 0
count = 0
for file in files:
	print file, count*100.0/len(files) 
	img = Image.open(path + file, 'r')
	img_w, img_h = img.size

	background = Image.new('RGB', (1540, 1000), (255, 255, 255))
	bg_w, bg_h = background.size
	offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
	background.paste(img, offset)
	# background.thumbnail(background.size, Image.ANTIALIAS)
	background.save(path_to + file + '.jpg', "JPEG")
	count += 1