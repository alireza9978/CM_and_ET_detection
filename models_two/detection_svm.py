# SVM
import numpy
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
from pathlib import Path
from sklearn import metrics


def train_test_splitter(df):
    anomalies = df[df.label == 1]
    healthy_data = df[df.label == 0].sample(n=2 * anomalies.shape[0])
    new_data = pd.concat([anomalies, healthy_data])
    labels = new_data['label'].values
    new_data = new_data.drop(columns=['label', 'Unnamed: 0'])
    return train_test_split(
        new_data, labels, test_size=0.33, random_state=42, stratify=labels)


df = pd.read_csv(Path('H:/Projects/Datasets/irish_ann_total.csv'), delimiter=',')
X_train, X_test, y_train, y_test = train_test_splitter(df)
clf = svm.SVC(C=1.0, kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
