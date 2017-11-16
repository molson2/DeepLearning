from PIL import Image
import re
import urllib
from bs4 import BeautifulSoup
import time


def downsample_image(name):
    im = Image.open(name + '.jpg')
    im_small = im.resize((299, 299))
    im_small.save(name + '_small.jpg')


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


html_path = 'nike.html'
base_name = 'nike'
