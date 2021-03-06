# Pandas

https://pandas.pydata.org/pandas-docs/stable/reference/index.html#api

## Import pandas

```py
import pandas as pd
```

## Read csv

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

```py
# imported pandas as pd
df = pd.read_csv('url or local path', names=[])
# names is to put the name of the columns
```

## Show descriptive stats

```py
df.describe(include='all')
```

## Replace values

```py
df.replace('value_to_search', 'value_to_replace', inplace = True)
df['column_name'].replace('value_to_search', 'value_to_replace', inplace = True)
```

## Count categorical variables

```py
df['column_name'].value_counts()
```

## Get types

```py
df.dtypes
```

## Convert column of type

```py
df['column'] = df['column'].astype('type')
# type could be float int string
```

## Rename column

```py
df.rename(columns={'old_name': 'new_name'}, inplace=True)
```

## Calculate correlation

Calculate corr between all numeric variables

```py
df.corr()
```


## Read from excel fules

```py
!pip install xlrd
df = pd.read_excel('URL', sheetname='sheetname', skiprows=[], skip_footer=number)

```


### Transform json

```py
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize
dataframe = json_normalize(venues)
dataframe.head()
```


### Pandas filter

```py
df = df[df['Borough'] != 'Not assigned']
```

### Pandas Unique - freq

```py
df['Postal Code'].unique
df['Postal Code'].value_counts
```

### Drop column

<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html>

```py
df.drop(['B', 'C'], axis=1)
```
