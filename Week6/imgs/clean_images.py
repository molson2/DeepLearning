from PIL import Image


def downsample_image(name):
    im = Image.open(name + '.jpg')
    im_small = im.resize((299, 299))
    im_small.save(name + '_small.jpg')


downsample_image('felix')
downsample_image('bebe')
downsample_image('logan')
downsample_image('donny')
downsample_image('hotdog')
