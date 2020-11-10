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
