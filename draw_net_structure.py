import sys
sys.path.append('/home/imatge/caffe-master/python')
import caffe
import caffe.draw
from caffe.proto import caffe_pb2

from google.protobuf import text_format

#Parameter configuration

# network file structure
input_net_proto_file ='../fcn/voc-fcn8s-atonce/train.prototxt'
# output image file
output_image_file ='net_pic.jpg'
# Arrangement of # network structure: LR, TB, RL etc.
rankdir ='LR'

#Read network

net = caffe_pb2.NetParameter()
text_format.Merge(open(input_net_proto_file).read(), net)

#Draw network
print('Drawing')
caffe.draw.draw_net_to_file (net, output_image_file, rankdir)
print('done... ')