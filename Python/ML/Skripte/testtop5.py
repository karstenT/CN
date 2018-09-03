import numpy as np
import sys
from PIL import Image
from sklearn.metrics import confusion_matrix
import math
from os import listdir
from os.path import isfile, join


import timing
########## Set Batch Size here: ###########
batchSize = 250
###########################################


args = sys.argv
PRETRAINED = ""
MODEL_FILE = ""
mean = ""
testDir = ""
labelstxt = ""

# Set paths of files with weights (PRETRAINED), architecture (MODEL_FILE), average pixel values from training images (meanproto) and labels (labelstxt)
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

# Set path to dircectory with testset (Portraits - /test/ or OCRE Classes - /test2/)

testDir = "/test/"
#testDir = "/test2/"


# Make sure that caffe is on the python path: 
CAFFE_ROOT = '/home/antje/caffe/python' # CHANGE THIS LINE TO YOUR Caffe PATH
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe
import os

# Generating a list of all labels
labelsO = open(labelstxt,'r').read().split('\n')
labels = []
for lab in labelsO:
	labels.append(lab.replace(" ","_"))
if '' in labels:
	del labels[(labels.index(''))]

# convert .binaryproto to .npy (necessary to work with caffe)
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( meanproto , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( "mean.npy" , out )


# Use GPU or CPU
caffe.set_mode_cpu()
#caffe.set_mode_gpu()


# List all test files (.png) in subfolders of dir
def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:		                                                                                        
                if 'png' in str(file):
			r.append(subdir + "/" + file)                                                                         
    return r 



# generate a dictionary with path to images and their ground trouth classId
def genTestList(fileList):
	fileClassDict = {}
	for files in fileList:
		for classes in labels:
			fileclass = classes.replace(" ","_")
			if fileclass in files:
				classId = (labels.index(classes)+1)
				pathAndClass = {files:classId}
				fileClassDict.update(pathAndClass) 	
	return(fileClassDict)                         

# Calculate the number of batches to process given a size and a list of all files to process	
def calcBatchNumbers(batchSize, fileList):
	lenfL = len(fileList) 
	number = int(math.ceil(lenfL/batchSize))
	print("The Net has to process " + str(number+1) + " Batches, one Batch contains " + str(batchSize) + " images")
	return(number)
	

# Returns a list of all images that have to be processed in this batch	
def batchFileList(batchSize, batchNumber, testImages):
	processedImages = batchSize*batchNumber
	newImagesEnd = processedImages + batchSize
	if newImagesEnd > len(testImages):
		filesToProcess = testImages.keys()[processedImages:len(testImages)]
	else:
		filesToProcess = testImages.keys()[processedImages:newImagesEnd]
	return (filesToProcess)

	
# Actual processing of images in batches with caffe and given model	
def processBatches(batchSize,filesToProcess,trueList,predList):
	inimages = filesToProcess
	gTList = {}
	for inpath in inimages:
		pathWords = inpath.split('/')
		lenpathWords = len(pathWords)
		gTList[inpath] = pathWords[lenpathWords-2]
	input_images=[]
	bs = len(inimages)

	# loading neural net with trained weights
	net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)	
	in_shape = net.blobs['data'].data.shape	
	# reshaping data layer to process #images in one batch (bs) (with 3 colors (RGB) and size 244x244)
	# use 227x227 for AlexNet
	net.blobs['data'].reshape(bs, 3, 227, 227)
	net.reshape()
	# Set the shape of the input for the transformer
	transformer = caffe.io.Transformer({'data': in_shape})
	# set mean from given mean file (previous converted from the .binaryproto)
	transformer.set_mean('data', np.load('mean.npy').mean(1).mean(1))
	# order of color channels
	transformer.set_transpose('data', (2,0,1))
	# RGB to BGR
	transformer.set_channel_swap('data', (2,1,0))


	for i, f in enumerate(inimages):
   		img = Image.open(f)
    		# scale all images to 224x224 (VGG16) or 227x227 (AlexNet)
    		img = img.resize((224,224), Image.ANTIALIAS)
		#img = img.resize((227,227), Image.ANTIALIAS)
    		img = np.array(img).astype(np.float32)
		# Transform the image depending of data layer definition and transformer settings from above
    		transformed_image = transformer.preprocess('data', img)
    		# put the image into i-th place in batch
    		net.blobs['data'].data[i,:,:,:] = transformed_image

	# Forward pass of the images through the network
	out = net.forward()
	
	# Return ground truth and prediction from the net
	for i in range(0,bs):
		top5Indices = []
		listOut = out['softmax'][i]
		maxClass = out['softmax'][i].argmax()

		top5Indices = np.argpartition(listOut,-5)[-5:]
		top5Labels = []
		for x in top5Indices:
			top5Labels.append(labels[x])
		groundTruth = gTList[inimages[i]]
		trueList.append(groundTruth)
		if groundTruth in top5Labels:
			predList.append(groundTruth)
		else:
			predList.append(labels[maxClass])
		print("For picture " + str(i) + " the net predicts: " + str(top5Labels) + " and ground truth is: " + groundTruth)
	return trueList,predList



# Initialize
trueList = []
predList = []
fileList = list_files(testDir) 	# list of all test images in folder /images/test/
testImages = genTestList(fileList)
batchNum = calcBatchNumbers(batchSize, fileList)
for i in range(0,batchNum+1):	
	print("i: " + str(i))
	filesToProcess = batchFileList(batchSize, i, testImages)
	trueList, predList = processBatches(batchSize,filesToProcess,trueList,predList)

# Save all predictions to a confusion matrix in csv to read in again and calculate evaluation metrics
np.savetxt("confMat.csv", (confusion_matrix(trueList, predList,labels)), delimiter=",")
























