from sklearn.metrics import classification_report
import numpy
from sklearn.metrics import accuracy_score


mat = numpy.loadtxt(open("confMat.csv", "rb"), delimiter=",").astype("int")

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

print(classification_report(true, pred, target_names=labels))
print("+++++++++++++++++++++++++ accuracy: ++++++++++++++++++++++++++")
print("Accuracy: " + str(accuracy_score(true, pred)))

top1_100 = ntop1*100
top1Error = top1_100/float(ges)

print("Top-1 Error: {:f} ".format(top1Error))

