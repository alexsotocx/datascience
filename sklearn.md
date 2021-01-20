# SKlearn

Create models lib in python <https://scikit-learn.org/stable/>

## Linear models

Relationship between 1 to 1 continous variables

```py
from sklearn.linear_model import LinearRegression
# Create a Linear regression object
lm = LinearRegression()
# Define predictor and target variable
x = df[['predictor']]
y = df['price']
# Fit the model -> obtain slope and intercept
lm.fit(x, y)

# Predict
predicted_y = lm.predict(x)
# Intercept
lm.intercept_
# slope
lm.coef_
```

## Multiple linear regression

Relationship between 1 continous variable and Two or more predictor variables.

```py
# extract variables you need
Z = df[['var_1', 'var_2', 'var_x']]
lm = LinearRegression()
lm.fit(Z, df['predicted_var'])
```

## Polynomial Regression with more than One dimension

```py
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=2, include_bias=False)
pr.fit_transform([1,2], include_bias=false)
```

## Preprocessing data

### Normalizing data scaler

```py
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df[['var_1', 'var_2']])
df_scaled = scaler.transform(df[['var_1', 'var_2']])
```

## Pipelines

```py
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

steps = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree=2)),more steps, ('mode', LinearRegression())]
pipeline = Pipeline(steps)
pipeline.fit(df[['var1', 'var2', 'var']], y)
yhat = pipeline.predict(X[['var1', 'var2', 'var']])
```


## Evaluating error

### MSE

Error between predicted and actual value squared divided by the number of samples

```py
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['var'], t_dpredicted_var)
```

### R^2

Coefficient of determination, how closed is the data to the regression line

R^2 = (1 - (MSE of regression Line / MSE of the average of the data))

if close to 1 is good if close to 0 is bad
Explains how much percentage of the variation of the predicted value is explanied by this simple linear module

```py
lm = LinearRegression()
lm.fit(df['var_1'], df['var2'])
R2 = lm.score(df['var_1'], df['var2'])
```

## Training and splitting set

Normally you split your data to train the model and other to test it.  A large subset of the sample is dedicated to train the model and the rest is used to validate it.

```py
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3(the percentage to split), random_state=0 (random seed))
```

## Cross validation

Divide the sample in K groups known as a fold, and use each fold as training and validation. Then get the average result

```py
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
scores = cross_val_score(lr, x_data, y_data, cv=partitions)
yhat = cross_val_predict(lr, x_data, y_data, cv=partitions)
```

## Ridge Regression

Define alpha values to reduce the coefficients of the Polynomial regression.

To select the correct alpha we generate several model, first we split the sample to train and to predict and then calculate the R^2 of each model and select the best of those

```py
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X, y)
Yhat = ridge_model.predict(X)
```


## Knearest neighboor

Classification based on the closest K points based and compare to the new one, and see how they were classified, and then select the average of those close K points
Classify based to simarility

1. Pick a value for K
2. Calculate the distance of all unknown cases from all cases.
3. Select all nearest k points
4. Select the classification based on the most popular respnse.

```py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

k = 4
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

```

### Evaluation metrics

```py
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
```

#### Jaccard Index

y -> Actual values
yhat -> Predicted values

J(y, yhat) = |y n yhat| / |y| + |yhat| - |y n yhat|

The bigger the value is close to 1 the better

#### Confussion matrix - F1 score

* Confussion matrix row shows the true label.
* Confussion matrix columns shows the predicted label.
* Precision = TP / (TP + FP)
* Recall = TP / (TP + FN)
* F1-score = 2 * (prc * rec) / (prc + rec) the closer to 1 the better

```py
from sklearn.metrics import classification_report, confusion_matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
print (classification_report(y_test, yhat))
```

#### Log loss

Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.

```py
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)
```

### Logicstic regression

Logistic Regression is a variation of Linear Regression, useful when the observed dependent variable, y, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables.

Logistic regression fits a special s-shaped curve by taking the linear regression and transforming the numeric estimate into a probability

Probability of a class = e ^ (theta * X) / 1 + e ^ (theta * X)

```py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat
yhat_prob = LR.predict_proba(X_test)
yhat_prob
```

## Decision trees

Used to clasify attributes into 1 outcome

* Entropy: Measures randomness in the data = `- p(A)log(p(A)) - p(B)log(p(B))` where p(x) -> is the proportion or the probability of getting x in the sample
* Information gain = Entropy before splitting - weighted entropy after split
* Nodes are considered pure if all the items fall into the same category.

```py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)
drugTree.predict(y_test)

```

### Evaluation

```py
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)
print (classification_report(y_test, yhat))
```

### How to build?

* Select the more attribute predictiviness, less impurity, low Entropy
* Calculate the entropy and information gain,
* Split the data, and repeat

## K Mean cluster

Iterative process:

1. Select k random points
2. calculate the distance to each point of the sample
3. select the distance to the minimum K point and integrate to cluster.
4. Move the K centroids calculating the center of the points forming the cluster.
5. Go back to 2 until the centroids are balanced.

What is the objective of k-means?

* To form clusters in such a way that similar samples go into a cluster, and dissimilar samples fall into different clusters.

* To minimize the “intra cluster” distances and
maximize the “inter-cluster” distances.

* To divide the data into non-overlapping clusters without any cluster-internal structure

### Implementation

* init: Initialization method of the centroids. Value will be: "k-means++", k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
* n_clusters: The number of clusters to form as well as the number of centroids to generate.
Value will be: 4 (since we have 4 centers)
* n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
Value will be: 12

```py
from sklearn.cluster import KMeans
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(X)
```

## Hierachical clustering

The Agglomerative Clustering class will require two inputs:

* n_clusters: The number of clusters to form as well as the number of centroids to generate.
Value will be: 4
* linkage: Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
Value will be: 'complete'.
Note: It is recommended you try everything with 'average' as well

```py
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
agglom.fit(X1,y1)

dist_matrix = distance_matrix(X1,X1)
print(dist_matrix)

Z = hierarchy.linkage(dist_matrix, 'complete')
dendro = hierarchy.dendrogram(Z)
```

### How to calculate the distance

```py
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
```

### Clustering using sklearn

```py
dist_matrix = distance_matrix(feature_mtx,feature_mtx)
print(dist_matrix)
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
agglom.labels_
```

## DBSCAN - Density based clustering

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. This technique is one of the most common clustering algorithms which works based on density of object. The whole idea is that if a particular point belongs to a cluster, it should be near to lots of other points in that cluster.

It works based on two parameters: Epsilon and Minimum Points
Epsilon determine a specified radius that if includes enough number of points within, we call it dense area
minimumSamples determine the minimum number of data points we want in a neighborhood to define a cluster.

Note: Remember the data need to be standarized

```py
from sklearn.cluster import DBSCAN
epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_
labels
```
