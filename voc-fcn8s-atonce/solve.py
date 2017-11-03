import sys
sys.path.append('/home/imatge/caffe-master/python')
sys.path.append('/home/imatge/caffe-master/code')
sys.path.append('../')
sys.path.append('../../caffe/python')
import caffe
import surgery, score
import os
from pylab import *
import setproctitle


setproctitle.setproctitle(os.path.basename(os.getcwd()))

# Load weights
weights = '../../../data/models/VGG_ILSVRC_16_layers.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../../../data/coco-text/val-onlyLegibleText-500im.txt', dtype=str)
# val = np.loadtxt('../../../data/icdar-resized/val.txt', dtype=str)


# change batch-size
# print(solver.net.blobs['data'].data.shape)
# solver.net.blobs['data'].reshape(1, 3, 720, 1280)

#load snapshot
# solver.restore('../../../data/fcn_training/snapshot/train_iter_4000.solverstate')

# init vars to train and  store results
size_intervals = 7000 #4000 No of iterations between each validation and plot
num_intervals = 40  #25 No of times to validate and plot
total_iterations = size_intervals * num_intervals # 25*4000 = 100.000 total iterations

# set plots data
train_loss = zeros(num_intervals)
val_loss = zeros(num_intervals)
val_acc = zeros(num_intervals)
val_iu = zeros(num_intervals)
it_axes = (arange(num_intervals) * size_intervals) + size_intervals

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss (b) - val loss (r)')
ax2.set_ylabel('val accuracy (y) - val iu (g)')
ax2.set_autoscaley_on(False)
ax2.set_ylim([0, 1])

for it in range(num_intervals):

    solver.step(size_intervals)
    # solver.net.forward()

    # Test with validation set every 'size_intervals' iterations
    [loss, acc, iu] = score.seg_tests(solver, False, val, layer='score')
    val_acc[it] = acc
    val_iu[it] = iu
    val_loss[it] = loss
    train_loss[it] = solver.net.blobs['loss_conv'].data

    # Plot results
    if it > 0:
        ax1.lines.pop(1)
        ax1.lines.pop(0)
        ax2.lines.pop(1)
        ax2.lines.pop(0)

    ax1.plot(it_axes[0:it+1], train_loss[0:it+1], 'b') #Training loss averaged last 20 iterations
    ax1.plot(it_axes[0:it+1], val_loss[0:it+1], 'r')    #Average validation loss
    ax2.plot(it_axes[0:it+1], val_acc[0:it+1], 'y') #Average validation accuracy (mean accuracy of text and background)
    ax2.plot(it_axes[0:it+1], val_iu[0:it+1], 'g')  #Average intersecction over union of score-groundtruth masks
    plt.ion()
    plt.show()
    plt.pause(0.001)
    title = '../../../data/fcn_training/evaluation/training-' + str(solver.iter) + '.png' #Save graph to disk every "size intervals"
    savefig(title, bbox_inches='tight')


