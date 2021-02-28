# datascience_eda 

![](https://github.com/lephanthuymai/datascience_eda/workflows/build/badge.svg) 
[![codecov](https://codecov.io/gh/UBC-MDS/datascience_eda/branch/main/graph/badge.svg?token=FL08APHGDS)](https://codecov.io/gh/UBC-MDS/datascience_eda)
![Release](https://github.com/lephanthuymai/datascience_eda/workflows/Release/badge.svg) 
[![Documentation Status](https://readthedocs.org/projects/datascience_eda/badge/?version=latest)](https://datascience_eda.readthedocs.io/en/latest/?badge=latest)

This package includes functions assisting data scientists with some common tasks during the exploratory data analysis stage of a data science project. This library will help the data scientist to do preliminary analysis on common column types like numeric columns, categorical columns and text columns and also conduct several experimental clustering on the dataset.

Our functions are tailored based on our own experience, there are also similar packages published on [PyPi](https://pypi.org/search/?q=eda&page=1), a few good ones worth mentioning:
* [eda-viz](https://github.com/ajaymaity/eda-viz)
* [eda_and_beyond](https://github.com/FredaXin/eda_and_beyond)
* [Quick-EDA](https://github.com/sid-the-coder/QuickDA)
* [easy-data-analysis](https://github.com/jschnab/easy-data-analysis)

## Installation

```bash
$ pip install -i https://test.pypi.org/simple/ datascience_eda
```

## Main Functions

- `explore_numeric_columns`: conducts common exploratory analysis on columns with numeric type: it generates a heatmap showing correlation coefficients (using `pearson`, `kendall` or `spearman` correlation on choice), histograms and SPLOM plots for all numeric columns or a list of columns specified by the user. This returns a list of plot objects so that the user can save and use them later on.

- `explore_categorical_columns`: performs exploratory analysis on categorical features. It creates and returns a dataframe containing column names, corresponding unique categories, counts of null values, percentages of null values and most frequent categories. It also generates and visualize countplots of a list of categorical columns of choice.

- `explore_text_columns`: performs exploratory data analysis of text features. It prints the summary statistics of character length and word count. It also plots the word cloud, distributions of character lengths, word count and polarity and subjectivity scores. Bar charts of top n stopwords and top n words other than stopwords, top n bigrams, sentiments, name entities and part of speech tags will be visualized as well. This returns a list of plot objects.

- `explore_clustering`: fits K-Means and DBSCAN clustering algorithms on the dataset and visualizes Elbow and Silhouette Score plots. It returns a dictionary with each key being name of the clustering algorithm and the value being a list of plots generated by the models


## Dependencies

List of depencies can be found under at: https://github.com/UBC-MDS/datascience_eda/blob/main/pyproject.toml

## Usage

```
import pandas as pd
import datascience_eda as eda

df = pd.read_csv("datafile.csv")

eda.explore_numeric_columns(df)
eda.explore_categorical_columns(df)
eda.explore_text_columns(df, ["categorical_column1", "categorical_column2"])
eda.explore_clustering(df)

```

## Documentation

The official documentation is hosted on Read the Docs: https://datascience_eda.readthedocs.io/en/latest/

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/UBC-MDS/datascience_eda/graphs/contributors).

### Credits

This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
