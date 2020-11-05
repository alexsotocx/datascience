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

Used to see correlation between two variables

```py
sns.regplot(x='x_var', y='variable', data=df)
plt.ylim(0,)
```


### Heatmap
