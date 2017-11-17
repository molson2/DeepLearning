from PIL import Image
import re
import urllib
from bs4 import BeautifulSoup
import time
import glob
import os
import numpy as np


def downsample_image(path, out_dir):
    im = Image.open(path)
    im_small = im.resize((299, 299))
    im_small.save(out_dir + path)


def get_images(html_path, base_name):
    with open(html_path, 'r') as f:
        txt = f.read()

    img_srcs = re.findall(r'<img .*?>', txt)
    i = 0
    for img_src in img_srcs:
        time.sleep(2)
        try:
            img_soup = BeautifulSoup(img_src, 'lxml')
            url = img_soup.find('img')['src']
            urllib.urlretrieve(url, base_name + str(i) + '.jpg')
            i += 1
        except:
            print 'bad img'
    return i


# download jpgs'
get_images('nike.html', 'nike')
get_images('addidas.html', 'addidas')

# downsample #
all_jpgs = glob.glob('*.jpg')
all_jpgs = [jpg for jpg in all_jpgs if os.path.getsize(jpg) > 2000]
for jpg in all_jpgs:
    downsample_image(jpg, '../shoes_processed/')
