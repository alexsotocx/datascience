## Import plotting lib

* https://seaborn.pydata.org/introduction.html
* https://matplotlib.org/tutorials/introductory/pyplot.html

```py
import seaborn as sns
import matplotlib.pyplot as plt
```

## Draw boxplot

```py
sns.boxplot(x='var_name', y='other_Var', data=df)
```

### Scatterplot

```py
plt.scatter(x_independent_var_predictor, y_predicted_var_to_predict)

plt.tittle('string')
plt.xlable('string')
plt.ylabel('string')
```

### Regression plot

Used to see correlation between two variables and show the regression between them

```py
sns.regplot(x='x_var', y='variable', data=df)
plt.ylim(0,)
```

### Residual plot

See if the regression is correct visually
<https://seaborn.pydata.org/generated/seaborn.residplot.html>

```py
sns.residplot(df['predictor'], df['price'])
```

### Distribution plots

Counts how the predicted values vs the actual values behave
<https://seaborn.pydata.org/generated/seaborn.displot.html#seaborn.displot>

```py

ax1 = sns.displot(df['price'], hist=False (Histogram), color="r", label="Actual values")
# Yhat is the prediction
sns.displot(Yhat, hist=False, color='b', label='Fitted values', ax=ax1)
```

### Plotting magic in notebooks

```py
%matplotlib inline -> renders in the same windows
%matplotlib notebook -> allow modifications of the generated plot
```

### Plotting data with Pandas

```py
df.plot(kind='line')
df['india'].plot(kind='hist')
```

### Line plots

Used when the variable to plot is continue.

```py
df.loc['var1', 'var2'].plot(kind='line')
plt.title('title')
plt.xlabel('title')
plt.ylabel('title')

plt.show()
```
