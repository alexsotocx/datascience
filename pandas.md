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