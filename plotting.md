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

* <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html?highlight=plot#pandas.DataFrame.plot>

```py
df.loc['var1', 'var2'].plot(kind='line')
plt.title('title')
plt.xlabel('title')
plt.ylabel('title')

plt.show()
```

### Area plots

Used to represent columated totals using numbers or percentages over time

```py
df.plot(kind='area',  alpha=0.25, # 0-1, default value a= 0.5
             stacked=False,
             figsize=(20, 10),)
plt.show()
```

### Bar charts

```py
df.plot(kind='bar')
plt.show()

# Annotate arrow
plt.annotate('',                      # s: str. will leave it blank for no text
             xy=(32, 70),             # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),         # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',         # will use the coordinate system of the object being annotated
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
            )

# Annotate Text
plt.annotate('2008 - 2011 Financial Crisis', # text to display
             xy=(28, 30),                    # start the text at at point (year 2008 , pop 30)
             rotation=72.5,                  # based on trial and error to match the arrow
             va='bottom',                    # want the text to be vertically 'bottom' aligned
             ha='left',                      # want the text to be horizontally 'left' algned.
            )
```

### Histomgrams

```py
# 'bin_edges' is a list of bin intervals
count, bin_edges = np.histogram(df['2013'])

df['2013'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)

```

### Pie chart

SHow proportion using circle shape


```py
# 'bin_edges' is a list of bin intervals
df['var'].plot(kind='pie')

```

### Multiple Graphs

```py
fig = plt.figure() # create figure

ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

# Subplot 1: Box plot
df_CI.plot(kind='box', color='blue', vert=False, figsize=(20, 6), ax=ax0) # add to subplot 1
ax0.set_title('Box Plots of Immigrants from China and India (1980 - 2013)')
ax0.set_xlabel('Number of Immigrants')
ax0.set_ylabel('Countries')

# Subplot 2: Line plot
df_CI.plot(kind='line', figsize=(20, 6), ax=ax1) # add to subplot 2
ax1.set_title ('Line Plots of Immigrants from China and India (1980 - 2013)')
ax1.set_ylabel('Number of Immigrants')
ax1.set_xlabel('Years')

plt.show()
```

### Waffle chart

* Display progress toward goals. Contribution of each country

```py
def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):

    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)

    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print (df_dsn.index.values[i] + ': ' + str(tiles))

    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1

            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index

    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'

        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )

```

### World cloud

Shows the frequency of different words in textual data. The more the word appear the bolder, and bigger will appear.
