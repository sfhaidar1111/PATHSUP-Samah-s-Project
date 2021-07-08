# PATHSUP-Samah-s-Project
##DBSCAN
dataset = datasets.load_breast_cancer()
X_data = dataset.data
y_target = dataset.target
StandardizedData = StandardScaler().fit_transform(X_data)
model= DBSCAN(eps= 0.2, min_samples= 6, metric='euclidean')
k_pred = model.fit_predict(StandardizedData)
df = pd.DataFrame({'prediction': k_pred, 'ground-truth': y_target})
ct = pd.crosstab(df['prediction'], df['ground-truth'])
print(ct)

y_pred = np.zeros((569,))
y_pred[np.where(y_target==-1)]= 1




print("Confusion matrix: \n", confusion_matrix(y_target, y_pred))
print("Accuracy score: \n", accuracy_score(y_target, y_pred))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(StandardizedData[:,0], StandardizedData[:,1], c=y_target, cmap='jet', edgecolor='None', alpha=0.35)
ax1.set_title('Actual labels')
ax2.scatter(StandardizedData[:,0], StandardizedData[:,1], c=y_pred, cmap='jet', edgecolor='None', alpha=0.35)
ax2.set_title('DScanModel clustering results')

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
y_pred[np.where(y_target== 0)]= 0
y_pred[np.where(y_target== 1)]= 1

print("Confusion matrix: \n", confusion_matrix(y_target, y_pred))
print("Accuracy score: \n", accuracy_score(y_target, y_pred))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(StandardizedData[:,0], StandardizedData[:,1], c=y_target, cmap='jet', edgecolor='None', alpha=0.35)
ax1.set_title('Actual labels')
ax2.scatter(StandardizedData[:,0], StandardizedData[:,1], c=y_pred, cmap='jet', edgecolor='None', alpha=0.35)
ax2.set_title('GmModel clustering results')


