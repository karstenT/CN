from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import itertools
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sn
import pandas as pd

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


mat = np.loadtxt(open("confMat.csv", "rb"), delimiter=",").astype("int")

labelsO = open('labels.txt','r').read().split('\n')
labels = []
dicError = {}
for lab in labelsO:
	labels.append(lab.replace(" ","_"))
	dicError[lab.replace(" ","_")] = 0
#labels.replace(" ","_")
if '' in labels:
	del labels[(labels.index(''))]

i=0
pred = []
true = []
a = 0

ges = 0


for x in mat:	#reihe	
	b = 0
	j=0
	for y in x:     #spalte
		if a!=b:
			prevNumber = dicError[labels[j]]
			dicError[labels[j]]+=y	
		b+=1
		ges += y
		for n in range(0,y):
			true.append(labels[i])
			pred.append(labels[j])
		j+=1
	i+=1
	a+=1

ntop1 = 0
for key, value in dicError.iteritems():
	ntop1+=value

#print(classification_report(true, pred, target_names=labels))
#print("+++++++++++++++++++++++++ accuracy: ++++++++++++++++++++++++++")
#print("Accuracy: " + str(accuracy_score(true, pred)))

#top1_100 = ntop1*100
#top1Error = top1_100/float(ges)

#print("Top-1 Error: {:f} ".format(top1Error))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Paired):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)



    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# Compute confusion matrix
cnf_matrix = confusion_matrix(true, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
nonNormFig = plt.figure(figsize=(12,12))
plt.grid(True)
plt.grid(linestyle='-', linewidth='0.5', color='white')
plot_confusion_matrix(cnf_matrix, classes=labels,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
normFig = plt.figure(figsize=(12,12))
plt.grid(True)
plt.grid(linestyle='-', linewidth='0.5', color='white')
plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

nonNormFig.savefig('nonNormConf.png')
normFig.savefig('normConf.png')
plt.show()

