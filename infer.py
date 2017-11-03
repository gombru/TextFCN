import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imresize,imsave, toimage


import caffe

caffe.set_mode_gpu()

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('test.jpg') #.resize((512, 512), Image.ANTIALIAS)

if (im.size.__len__() == 2):
    im_gray = im
    im = Image.new("RGB", im_gray.size)
    im.paste(im_gray)

in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('voc-fcn8s-atonce/deploy.prototxt', '../../data/fcn_training/coco-104000/train_iter_104000.caffemodel', caffe.TEST)

# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

# run net and take argmax for prediction
net.forward()


hmap_0 = net.blobs['score_conv'].data[0][0,:,:]
hmap_1 = net.blobs['score_conv'].data[0][1,:,:]
hmap_0 = np.exp(hmap_0)
hmap_1 = np.exp(hmap_1)
hmap_softmax = hmap_1 / (hmap_0 + hmap_1)

# Save SoftMax heatmap
hmap_softmax_2save = (255.0 * hmap_softmax).astype(np.uint8)
hmap_softmax_2save = Image.fromarray(hmap_softmax_2save)
hmap_softmax_2save.save('heatmap.jpg')

# Save SoftMax color heatmap
fig = plt.figure(frameon=False)
fig.set_size_inches(5.12,5.12)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(hmap_softmax, aspect='auto', cmap="jet")
fig.savefig('heatmap-color.jpg')

#Save normalized heatmap
# heatmap = net.blobs['score_conv'].data[0][1,:,:]
# if heatmap.min() < 0:
#     heatmap = heatmap - heatmap.min()
# heatmap = (255.0 / heatmap.max() * (heatmap)).astype(np.uint8)
# heatmap = Image.fromarray(heatmap)
# heatmap.save('heatmap-normalized' + '.png')

print 'Done'
