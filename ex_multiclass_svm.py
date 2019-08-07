import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib

from sklearn.utils import shuffle
from sklearn.svm import SVC, SVR
from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

train = shuffle(pd.read_csv("data_training.csv"))
test = shuffle(pd.read_csv("data_testing.csv"))

print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in test set:",test.isnull().values.any(), "\n")

#Frequency distribution of classes"
train_outcome = pd.crosstab(index=train["label"],  # Make a crosstab
                              columns="count")      # Name the count column

# Visualizing Outcome Distribution
temp = train["label"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })

#df.plot(kind='pie',labels='labels',values='values', title='Activity Ditribution',subplots= "True")

labels = df['labels']
sizes = df['values']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','cyan','lightpink']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()


# Seperating Predictors and Outcome values from train and test sets
X_train = pd.DataFrame(train.drop(['label','filename'],axis=1))
Y_train_label = train.label.values.astype(object)
X_test = pd.DataFrame(test.drop(['label','filename'],axis=1))
Y_test_label = test.label.values.astype(object)

# Dimension of Train and Test set
print("Dimension of Train set",X_train.shape)
print("Dimension of Test set",X_test.shape,"\n")

# Transforming non numerical labels into numerical labels
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

# encoding train labels
encoder.fit(Y_train_label)
Y_train = encoder.transform(Y_train_label)

# encoding test labels
encoder.fit(Y_test_label)
Y_test = encoder.transform(Y_test_label)

#Total Number of Continous and Categorical features in the training set
num_cols = X_train._get_numeric_data().columns
print("Number of numeric features:",num_cols.size)
#list(set(X_train.columns) - set(num_cols))


names_of_predictors = list(X_train.columns.values)



#Libraries to Build Ensemble Model : Random Forest Classifier
# Create the parameter grid based on the results of random search
params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [0.1,1, 10, 100, 1000]},
                    {'kernel': ['linear'],
                     'tol':[0.01,0.06,0.8,0.1,0.3,0.6,1,3],
                     'C': [0.1,1, 10, 100, 1000]}]

svm_model = GridSearchCV(SVC(), params_grid, cv=5)
svm_model.fit(X_train, Y_train)


# View the accuracy score
print('Best score for training data:', svm_model.best_score_,"\n")

# View the best parameters for the model found using grid search
print('Best C:',svm_model.best_estimator_.C,"\n")
print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

final_model = svm_model.best_estimator_
Y_pred = final_model.predict(X_test)
joblib.dump(final_model, 'model_ku.pkl')

Y_pred_label = list(encoder.inverse_transform(Y_pred))
print(Y_pred)
# Making the Confusion Matrix
print(confusion_matrix(Y_test_label,Y_pred_label))
print("\n")
print(classification_report(Y_test_label,Y_pred_label))

print("Training set score for SVM: %f" % final_model.score(X_train , Y_train))
print("Testing  set score for SVM: %f" % final_model.score(X_test  , Y_test ))

mcm = multilabel_confusion_matrix(Y_test, Y_pred)
print(mcm)
ak=[]
for i in mcm:
    tP = (i[0])
    fP = (i[1])
    fN = (i[2])
    tN = (i[3])
    akurasi_klass = (tP + tN) / (tP+tN+fP+fN)
    ak.append(akurasi_klass)
    print("Akurasi Kelas ",i," ", akurasi_klass*100)

print("Akurasi Multiclass Confusion Matrix ",sum(ak)/4)
