import numpy as np
import sys
from PIL import Image
from sklearn.metrics import confusion_matrix
import math
from os import listdir
from os.path import isfile, join


args = sys.argv
PRETRAINED = ""
MODEL_FILE = ""
mean = ""
testDir = ""
labelstxt = ""


files = [f for f in listdir(".") if isfile(join(".", f))]
for t in files:
    if t.endswith(".caffemodel"):
        PRETRAINED = t
    if t.endswith("deploy.prototxt"):
        MODEL_FILE = t
    if t.endswith(".binaryproto"):
        meanproto = t
    if t.endswith("labels.txt"):
        labelstxt = t
print (listdir("."))
testFile = "/home/cnt/students/ML/Daten/CoinPortraitData/test/constantine_ii/constantine_ii23.png"
groundTruth = "constantine_ii"

# Make sure that caffe is on the python path: 
CAFFE_ROOT = '/home/cnt/caffe-master/python'        # CHANGE THIS LINE TO YOUR Caffe PATH
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe
import os

# Generating list of all labels
labelsO = open(labelstxt,'r').read().split('\n')
labels = []
for lab in labelsO:
    labels.append(lab.replace(" ","_"))
#labels.replace(" ","_")
if '' in labels:
    del labels[(labels.index(''))]

# convert .binaryproto to .npy
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( meanproto , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( "mean.npy" , out )


# Use GPU or CPU
#caffe.set_mode_cpu()
caffe.set_mode_gpu()


def processImage(image):
    # loading neural net with trained weights
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)    
    in_shape = net.blobs['data'].data.shape    
    # reshaping data layer to process #images in one batch (bs) (with 3 colors (RGB) and size 244x244)
    net.blobs['data'].reshape(1, 3, 224, 224)
    net.reshape()
    # Set the shape of the input for the transformer
    transformer = caffe.io.Transformer({'data': in_shape})
    # set mean from given mean file (previous converted from the .binaryproto)
    transformer.set_mean('data', np.load('mean.npy').mean(1).mean(1))
    # order of color channels
    transformer.set_transpose('data', (2,0,1))
    # RGB to BGR
    transformer.set_channel_swap('data', (2,1,0))
    img = Image.open(image)
    # scale all images to 224x224 (VGG16) or 227x227 (AlexNet)
    img = img.resize((224,224), Image.ANTIALIAS)
    #img = img.resize((227,227), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    # Transform the image depending of data layer definition and transformer settings from above
    transformed_image = transformer.preprocess('data', img)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass of the images through the network
    out = net.forward()
    maxClass = labels[out['softmax'][0].argmax()]    
    print("For picture " + str(testFile) + " the net predicts: " + str(maxClass) + " and ground truth is: " + groundTruth)



processImage(testFile)
