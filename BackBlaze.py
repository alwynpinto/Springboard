# -*- coding: utf-8 -*-
"""
BackBlaze HDD Failure Dataset
"""
# Import all dependencies
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from collections import Counter
from numpy import where
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Import data into DF
filenames = sorted(glob('Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\AllData\*.csv'))

df_2019=[]
for filename in filenames:
    df = pd.read_csv(filename, index_col=None, header=0, nrows = 1000)
    df_2019.append(df)

frame = pd.concat(df_2019, axis=0, ignore_index=True)
frame.head() , frame.shape
frame.tail() , frame.shape

frame_5 = frame[['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'smart_5_normalized', 'smart_5_raw', 'smart_187_normalized', 'smart_187_raw', 'smart_188_normalized', 'smart_188_raw', 'smart_197_normalized', 'smart_197_raw', 'smart_198_normalized', 'smart_198_raw' ]]
frame_5.head() , frame_5.shape
frame_5.tail() , frame_5.shape

frame_5norm = frame_5[['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'smart_5_normalized', 'smart_187_normalized', 'smart_188_normalized', 'smart_197_normalized', 'smart_198_normalized']]
frame_5norm.head(), frame_5norm.shape
frame_5norm.tail(), frame_5norm.shape

frame_5raw = frame_5[['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw' ]]
frame_5raw.head(), frame_5raw.shape
frame_5raw.tail(), frame_5raw.shape

'''
Backblaze's analysis of nearly 40,000 drives showed five SMART metrics that correlate strongly with impending disk drive failure:

SMART 5 - Reallocated_Sector_Count.
SMART 187 - Reported_Uncorrectable_Errors.
SMART 188 - Command_Timeout.
SMART 197 - Current_Pending_Sector_Count.
SMART 198 - Offline_Uncorrectable

Not considering others because different manufacturers' SMART values mean very different. If there is data available for what each SMART means and is standard across all the HDD, we can use these for analysis.

http://www.cropel.com/library/smart-attribute-list.aspx
https://www.hdsentinel.com/smart/index.php

'''

# Export data for further consideration
frame.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame.csv') # contains all column data
frame_5.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5.csv') # contains 5 columns mentioned by BackBlaze
frame_5norm.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5norm.csv') # contains only 5 normalized by BackBlaze
frame_5raw.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5raw.csv') # contains only 5 normalized by BackBlaze


############# Import data into DataFrame from Exported clean CSV
df = pd.read_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5raw.csv', parse_dates=True, index_col='date').fillna(0).drop('Unnamed: 0', axis=1)

# Fill up missing capacity
missing_capacity = (df[df['capacity_bytes'] != -1].drop_duplicates('model').set_index('model')['capacity_bytes'])
df['capacity_bytes'] = df['model'].map(missing_capacity)
df['capacity_tb'] = round(df['capacity_bytes']/1099511627776,2)

# Export cleaned data to CSV for archiving.
df.to_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5raw_clean.csv')


######*******************************#####################*********


# Import cleaned data for analysis
df = pd.read_csv(r'Z:\PersonalFolders\Alwyn Documents\Springboard\Projects\BackBlazeHDDfailure\frame_5raw_clean.csv', parse_dates=True, index_col='date')
df.shape
df_copy = df.copy()

# EDA of frame_5norm / df



# Train Test Split
X = df.reset_index().drop(['serial_number', 'model', 'capacity_bytes','capacity_tb','date', 'failure'],axis=1)
y = df.reset_index().failure
Counter(y)


x_train , x_test , y_train, y_test = train_test_split(X,y, test_size=0.3, stratify = y, random_state=42)
x_train.shape , y_train.shape, x_test.shape, y_test.shape
Counter(y_train)
columnnames = [x_train.columns]

# Preprocessing. Min Max Scaler
mm_scaler = preprocessing.MinMaxScaler()
x_train_scaled = mm_scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=columnnames)
x_train_scaled.shape

x_test_scaled = mm_scaler.fit_transform(x_test)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=columnnames)
x_train_scaled
x_test_scaled.shape








# Testing all classifiers
# Source: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
'''
Ran optimization for over sampling and under sampling percentages and hte results were:
os = [0.1,0.2,0.3]
us = [0.5,0.6,0.7]

and 0.1 and 0.5 gave a score of 0.839
'''


# SMOTE for balancing the data
oversample = SMOTE(sampling_strategy = 0.1)
undersample = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', oversample), ('u', undersample)]
pipeline = Pipeline(steps=steps)
x_train_scaled_s, y_train_s = pipeline.fit_resample(x_train_scaled, y_train)
Counter(y_train_s)[1]/(Counter(y_train_s)[0]+Counter(y_train_s)[1])

'''
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(model_dtc, x_train_scaled_s, y_train_s, scoring='accuracy', cv=cv, n_jobs=1)
print('Mean ROC AUC: %.3f' % (mean(scores)))
scores        
'''

# Create a function to test all classifiers, feeding xtrain, ytrain, xtest, ytest
# Source: https://www.kaggle.com/paultimothymooney/predicting-breast-cancer-from-nuclear-shape
# Source2:  https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/


def best_BC_model(a,b,c,d):
    print('\nComparing Multiple Classifiers: \n')
    print('K-Fold Cross-Validation Accuracy: \n')
    names = []
    models = []
    resultsAccuracy = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('XGB', XGBClassifier()))
    for name, model in models:
        model.fit(a, b)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
        accuracy_results = cross_val_score(model, a,b, cv=cv, scoring='accuracy')
        resultsAccuracy.append(accuracy_results)
        names.append(name)
        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(), accuracy_results.std())
        print(accuracyMessage) 
        y_pred_model = model.predict(c)
        accuracy_score_model = accuracy_score(y_test , y_pred_model)
        print(accuracy_score_model)
        print(confusion_matrix(y_test , y_pred_model))
        print(classification_report(y_test, y_pred_model))

# Feeding the 
best_BC_model(x_train_scaled_s , y_train_s, x_test_scaled, y_test)
Counter(y_test)
Counter(y_train_s)

