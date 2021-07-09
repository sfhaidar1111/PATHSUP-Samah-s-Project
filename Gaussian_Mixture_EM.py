# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:12:32 2021

@author: Samah Haidar
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import matplotlib.pyplot as pyplot
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
import pickle
from collections import Counter
import math
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import sklearn as sk
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

## Gaussian Mixture with Expectation Maximization (EM) Clustering

dataset = datasets.load_breast_cancer()
X_data = dataset.data
y_target = dataset.target
StandardizedData = StandardScaler().fit_transform(X_data)
model = GaussianMixture(n_components= 2, covariance_type="full")
k_pred = model.fit_predict(StandardizedData)
df = pd.DataFrame({'prediction': k_pred, 'ground-truth': y_target})
ct = pd.crosstab(df['prediction'], df['ground-truth'])
print(ct)

y_pred = np.zeros((569,))
y_pred[np.where(k_pred== 0)]= 1
y_pred[np.where(k_pred== 1)]= 0



print("Confusion matrix: \n", confusion_matrix(y_target, y_pred))
print("Accuracy score: \n", accuracy_score(y_target, y_pred))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(StandardizedData[:,0], StandardizedData[:,1], c=y_target, cmap='jet', edgecolor='None', alpha=0.35)
ax1.set_title('Actual labels')
ax2.scatter(StandardizedData[:,0], StandardizedData[:,1], c=y_pred, cmap='jet', edgecolor='None', alpha=0.35)
ax2.set_title('GmEMModel clustering results')
