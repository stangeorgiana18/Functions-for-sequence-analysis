import urllib3
import re
import requests
import numpy as np
from collections import OrderedDict
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sys import call_tracing

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve




# read fasta sequences and put them in a list

def readFasta(fasta_path):
    sequences = []

    # open the file to read the sequences
    with open(fasta_path, "r") as fasta:

        currentSequence = ""

        for line in fasta:
            line = line.strip()
            if line.startswith(">"):
                if currentSequence != "":
                    sequences.append(currentSequence)
                currentSequence = line + "\n"
            else: 
                # add a line to the read sequence
                currentSequence = currentSequence + line

        if currentSequence != "":
            sequences.append(currentSequence)

    return sequences

# function to parse the seq of aminoacids and the Uniprot Accesion code

def parse_seq(seq_fasta):

    lines = seq_fasta.split("\n")
    header = lines[0] # extract the first line = sequence name + uniprot code
    wholeSequence = "".join(lines[1:]) # get the whole protein seq without the antet/header line

    # [1] indicates the second element of this list which is the uniprot code
    uniProt_AC = header.split("|")[1]


    return uniProt_AC, wholeSequence

# return the frequency of aminoacids in a list

def freq_aa(seq):
    orderedAminoacidsCount = OrderedDict()
    for orderedAminoacids in "ACDEFGHIKLMNPQRSTVWY":
        orderedAminoacidsCount[orderedAminoacids] = 0
    
    for orderedAminoacids in seq:
        if orderedAminoacids in orderedAminoacidsCount:
            orderedAminoacidsCount[orderedAminoacids] += 1
  
    freqAminoacids = [] # list that stores the no. of each aminoacid
    for value in orderedAminoacidsCount.values():
        freqAminoacids.append(value)
    
    # return the list
    return freqAminoacids



fasta_path = "/Users/georgianastan/Desktop/proteinSeqAnalysis/catalytic_activity_proteins.fasta.txt"
proteins_list = readFasta(fasta_path)
# print(proteins_list)

frequences = []
for protein in proteins_list:
    uniProt_AC, seq = parse_seq(protein)
    frequences.append(freq_aa(seq))

# print(frequences[0:10])

catalyticActivityProteins = "/Users/georgianastan/Desktop/proteinSeqAnalysis/catalytic_activity_proteins.fasta.txt"
NonCatalyticActivityProteins = "/Users/georgianastan/Desktop/proteinSeqAnalysis/non_catalytic_activity_proteins.fasta.txt"

# insert protein by types in a respective list
caSequences = readFasta(catalyticActivityProteins)
noncaSequences = readFasta(NonCatalyticActivityProteins)

# model to classify proteins based on their amino acid sequences and evaluating its performance using standard metrics

# each row in data corresponds to a protein, each column to a feature
# target contains the labels - 0 and 1 for proteins

data = []
target = []

# consider only 300 proteins of each type

for n in caSequences[0:300]:
    uniProt_AC, seq = parse_seq(n)
    data.append(freq_aa((seq)))
    target.append(1) # protein type

for n in noncaSequences[0:300]:
    uniProt_AC, seq = parse_seq(n)
    data.append(freq_aa((seq)))
    target.append(0)

# split the data into training and testing sets using train_test_split
# and train a linear SVM model (svm.SVC) on the training data

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.4, random_state = 0) 
clf = svm.SVC(kernel = 'linear', C=1, probability = True).fit(X_train, y_train)

# cross validation evaluation metrics 
accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

# ROC curve = Receiver Operating Characteristic curve = graphic representation of 
# the learning model -> TPR true positive (sensitivity) / FPR false positive RATIO

# calculate the probabilities for the data sets 
y_prob = clf.predict_proba(X_test)[:, 1]

# calculate fpr, tpr and tresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# note: the closer the curve to the left upper corner, the better/higher the performance

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')

# add a diagonal line as a reference for the model performance
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# x and y axes limits of the graph 
# xlim is from 0 to 1 (false negative rate cannot be > 1)
# ylim is also from 0 to 1 (the value set is 1.05 to get space above the graph, to ensure it's visible)
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# display the roc curve
# every point on the roc curve = a different value for the classification treshold

plt.show()