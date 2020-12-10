# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:38:20 2018

@author: Mayuri Joshi
Milestone3 Assignment
Split your dataset into training and testing sets
Train your classifiers, using the training set partition
Apply your (trained) classifiers on the test set
Measure each classifier’s performance using at least 3 of the metrics we covered in this course (one of them has to be the ROC-based one). At one point, you’ll need to create a confusion matrix.
Document your results and your conclusions, along with any relevant comments about your work

"""

# import package
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read in MammographicCleanData dataset 
#Mamm = pd.read_csv('C:\\Users\\mayur_000\\Anaconda3\\Scripts\\Milestone3\\MammographicCleanData.csv',delimiter=",")
Mamm = pd.read_csv('MammographicCleanData.csv',delimiter=",")

# print the first20 rows of data from the dataframe
Mamm.head(20)

#Check distribution of each feature
from pandas.tools.plotting import scatter_matrix

import seaborn as sns
sns.set()
plt.xlabel("Reading of BI-RADS")
plt.ylabel("Totalnumber of patients")
plt.title("Histogram1 of Bi-RADS")
plt.legend()

plt.hist(Mamm.loc[:, "BI-RADSNormal"],rwidth=0.95,color='yellow')
plt.show()

# Check the distribution of the "Age" column
# Corece to numeric and impute medians for Age column
Mamm.loc[:, "Age"] = pd.to_numeric(Mamm.loc[:, "Age"], errors='coerce')
HasNan = np.isnan(Mamm.loc[:,"Age"]) 
Mamm.loc[HasNan, "Age"] = np.nanmedian(Mamm.loc[:,"Age"])
sns.set()
plt.xlabel("Age of patients")
plt.ylabel("Totalnumber of patients")
plt.title("Histogram2 of Age column")
plt.legend()

plt.hist(Mamm.loc[:, "Age"],rwidth=0.95,color='green')
plt.show()


#Check the distribution of the "Severity" column
MammSeverity= Mamm.loc[:,'Severity']

plt.xlabel("Reading of Severity")
plt.ylabel("Totalnumber of patients")
plt.title("Histogram1 of Severity")
plt.legend()
plt.hist(Mamm.loc[:, "Severity"],rwidth=0.95,color='blue')
plt.show()

# Plot all the numeric columns against each other
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import seaborn as sns
#scatter_matrix(Mamm)
plt.scatter(Mamm['Age'],Mamm['BI-RADSNormal'])
plt.show()
#############
#_ = scatter_matrix(Mamm, c=Mamm.loc[:,"Severity"], figsize=[8,8], s=1000)

plt.scatter(Mamm['DensityNormal'],Mamm['BI-RADSNormal'])
plt.show()


#From above scatter graph, we can confidently say that there is no linear corelation between Density and BIRADS and Age and BIRADS
#Predict the severity of the mammography to perform biopsy
#This is a classification problem
#Severity: benign=0 or malignant=1 (binominal)
#Severity as tageget variable
# define X and y

feature_cols = ['BI-RADSNormal', 'Age', 'DensityNormal','Shape_lobular','Shape_oval','Shape_round','Margin_microlobulated','Margin_spiculated']
# X is a matrix, hence we use [] to access the features we want in feature_cols
X = Mamm[feature_cols]

# y is a vector, hence we use dot to access 'label'
y = Mamm.Severity
# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split

# Spliting data in 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Training sample having dimesions===")
print(X_train.shape)
print("Testing sample having dimensions=====")
print(X_test.shape)

# Fit and Test Logistic regression model#Logistic regression model
# instantiate model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# fit model
logreg.fit(X_train, y_train)

# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)
print(y_pred_class)

# import sklearn KNNclassifier module
from sklearn.neighbors import KNeighborsClassifier 
# instantiate model
knn= KNeighborsClassifier()
# fit model
knn.fit(X_train,y_train)
#make predictions
y_knn_pred_class= knn.predict(X_test)
print(y_knn_pred_class)

#import sklearn SVC module 
from sklearn.svm import SVC
print ('\n\nSupport Vector Machine classifier\n')
# instantiate model
svm = SVC()
# fit model
svm.fit(X_train, y_train)
# make predictions on test data set
y_svm_pred_class = svm.predict(X_test)
print (y_svm_pred_class)

### Fit and apply Descion tree classifier#Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
#instentiate the model
dtc = DecisionTreeClassifier()
#fit model
dtc.fit(X_train,y_train)
# Make predictions on test data set
y_dtc_pred_class= dtc.predict(X_test)
print(y_dtc_pred_class)

#Calculate classification accuracy
from sklearn import metrics
# Classification accuracy of logistic regression
print(metrics.accuracy_score(y_test,y_pred_class))

# Classification accuracy of KNN
print(metrics.accuracy_score(y_test,y_knn_pred_class))

# Classification accuracy of Decision tree
print(metrics.accuracy_score(y_test,y_dtc_pred_class))

# Classification accuracy of support vector Matrix
print(metrics.accuracy_score(y_test,y_svm_pred_class))

Acc_score = [0.80,0.53,0.50,0.50]
Classifier=['DescionTree','LogisticReg','KNN','SVM']
plt.plot(Classifier,Acc_score)
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Accuracy score for Breast cancer classifier')
plt.xlabel('Classifier ')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
from IPython.display import display
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "orange")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def compute_cnf(classifier,x_test,y_test):
    cnf_matrix = confusion_matrix(classifier.predict(x_test),y_test)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Benign','Malignant'],
                      title='Confusion matrix, without normalization')

    plt.show()

    
from sklearn.metrics import f1_score as fscorer

def f1_score(classifier,x_test,y_test):
    return fscorer(classifier.predict(X_test),y_test)


#Calculate confusion matrix for logistic regression
from sklearn.metrics import *
confusion = confusion_matrix(y_test, y_pred_class)
print(confusion)

print("True positives are") 
TP = confusion[1,1]
print(TP)
print("True Negatives are") 
TN = confusion[0,0]
print(TN)
print("False positives are") 
FP = confusion[0,1]
print(FP)
print("False negatives are") 
FN = confusion[1,0]
print(FN)

print(accuracy_score(y_test, y_pred_class))

# calculate error rate
print(1 - accuracy_score(y_test, y_pred_class))

#recall 
print(recall_score(y_test, y_pred_class))

#precision
print(precision_score(y_test, y_pred_class))

#f1_score
print(f1_score(y_test, y_pred_class))

classifier = LogisticRegression()
classifier.fit(X_train,y_train)
print (classifier.score(X_test,y_test))

# Compute and plot confusion matrix
compute_cnf(classifier,X_test,y_test)

from sklearn.metrics import *


confusion = confusion_matrix(y_test, y_dtc_pred_class)
print(confusion)

print("True positives are") 
TP = confusion[1,1]
print(TP)
print("True Negatives are") 
TN = confusion[0,0]
print(TN)
print("False positives are") 
FP = confusion[0,1]
print(FP)
print("False negatives are") 
FN = confusion[1,0]
print(FN)

# calculate error rate
print(1 - accuracy_score(y_test, y_dtc_pred_class))

#recall 
print(recall_score(y_test, y_dtc_pred_class))

#precision
print(precision_score(y_test, y_dtc_pred_class))

#f1_score
print(f1_score(y_test, y_dtc_pred_class))

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
print (classifier.score(X_test,y_test))

# Compute and plot confusion matrix
compute_cnf(classifier,X_test,y_test)

#Calculate confusion matrix for K nearest mean
from sklearn.metrics import *
confusion = confusion_matrix(y_test, y_knn_pred_class)
print(confusion)

print("True positives are")
TP = confusion[1,1]
print(TP)
print("True Negatives are") 
TN = confusion[0,0]
print(TN)
print("False positives are") 
FP = confusion[0,1]
print(FP)
print("False negatives are") 
FN = confusion[1,0]
print(FN)

classifier = KNeighborsClassifier()
classifier.fit(X_train,y_train)
print (classifier.score(X_test,y_test))

# Compute and plot confusion matrix
compute_cnf(classifier,X_test,y_test)


#Calculate confusion matrix for support vector Matrix
from sklearn.metrics import *
confusion = confusion_matrix(y_test, y_svm_pred_class)
print(confusion)

print("True positives are")
TP = confusion[1,1]
print(TP)
print("True Negatives are") 
TN = confusion[0,0]
print(TN)
print("False positives are") 
FP = confusion[0,1]
print(FP)
print("False negatives are") 
FN = confusion[1,0]


classifier = SVC()
classifier.fit(X_train,y_train)
print (classifier.score(X_test,y_test))

# Compute and plot confusion matrix
compute_cnf(classifier,X_test,y_test)

#Calculate recall or sensitivity score
from sklearn import metrics
# Classification accuracy of logistic regression
print(metrics.recall_score(y_test,y_pred_class))

# Classification accuracy of KNN
print(metrics.recall_score(y_test,y_knn_pred_class))

# Classification accuracy of Decision tree
print(metrics.recall_score(y_test,y_dtc_pred_class))

# Classification accuracy of support vector Matrix
print(metrics.recall_score(y_test,y_svm_pred_class))

recall_score = [0.79,0.75,0.70,0.64]
Classifier=['LogisticReg','KNN','SVM','DescionTree']
plt.plot(Classifier,Acc_score)
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Sensitivity of Breast cancer classifier')
plt.xlabel('Classifier ')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()


 #IMPORTANT: first argument is true values, second argument is predicted probabilities

# we pass y_test and y_pred_prob
# we do not use y_pred_class, because it will give incorrect results without generating an error
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate

# store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for LogisticRegression classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

# store the predicted probabilities for class Knn
knn_y_pred_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, knn_y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Knn classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

y_pred_prob = dtc.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Knn classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()


##ROC Analysis for all Classifiers
# Parameters for the AUC Plot
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC1 = 'red' # Line Color 
LC2 = 'blue' # Line Color 
LC3 = 'orange' # Line Color 
LC4 = 'black' # Line Color 
LC5 = 'cyan' # Line Color 
LC6 = 'green' # Line Color 

def getAUCScore(testdata,preddata,classifiername):
    fpr, tpr, th = roc_curve(testdata, preddata) # False Positive Rate, True Positive Rate, probability thresholds
    AUC = auc(fpr, tpr)
    print ("For Classifier ",classifiername,":")
    print ("\nTP rates:", np.round(tpr, 2))
    print ("\nFP rates:", np.round(fpr, 2))
    print ("\nProbability thresholds:", np.round(th, 2),'\n\n')
    return fpr, tpr, th, AUC

fpr_logreg, tpr_logreg, th_logreg, AUC_logreg = getAUCScore(y_test, y_pred_class, "Logistic Regression")
fpr_dt, tpr_dt, th_dt, AUC_dt = getAUCScore(y_test, y_dtc_pred_class, "Decision Tree")
fpr_knn, tpr_knn, th_knn, AUC_knn = getAUCScore(y_test, y_knn_pred_class, "k Nearest Neighbors")
fpr_sv, tpr_sv, th_sv, AUC_sv = getAUCScore(y_test, y_svm_pred_class, "Support vector")




#Plot the Results of ROC Analysis/AUC Score
plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr_logreg, tpr_logreg, color=LC1,lw=LW, label='ROC curve LogReg (area = %0.2f)' % AUC_logreg)
plt.plot(fpr_dt, tpr_dt, color=LC2,lw=LW, label='ROC curve DT (area = %0.2f)' % AUC_dt)
plt.plot(fpr_knn, tpr_knn, color=LC4,lw=LW, label='ROC curve knn (area = %0.2f)' % AUC_knn)
plt.plot(fpr_sv, tpr_sv, color=LC5,lw=LW, label='ROC curve SV (area = %0.2f)' % AUC_sv)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for SVM
plt.legend(loc=LL)
plt.show()
####################


#Conclusion
#In this problem false negatives (FN) should not acceptable. 
#If patient is actually having malignant but it is predicted as non-cancerous, is most dangerous so our aim is to improve sensitivity. In other words we need to focus on sensitivity Sensitivity means- When actual value is positive, how often it is predicted correctly? Also known as Recall rate.
#What is best fit predictive model? So in the graph above we can clearly see, logistic regression has the highest value of sensitivity. so it is best fit predictive model