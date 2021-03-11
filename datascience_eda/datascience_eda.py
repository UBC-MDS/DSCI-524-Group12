from nltk.corpus import stopwords
from IPython.display import Markdown, display
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import en_core_web_md
from collections import Counter
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import numpy as np
import altair as alt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import nltk

nltk.download("stopwords")


# region support functions


def get_numeric_columns(df):
    """get all numeric columns' names

    Parameters
    ----------
    df : [pandas.DataFrame]
        the dataset

    Returns
    -------
    list
        list of numeric column names
    """
    numeric_cols = df.select_dtypes("number").columns.tolist()
    return numeric_cols


def _verify_distance_metric(m):
    """check if a distance metric is valid

    Parameters
    ----------
    m : string
        metric, should be among [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]

    Raise
    -------
    Exception
        throw an exception if the metric is invalid
    """
    if m not in ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]:
        raise Exception(f"Invalid distance metric: {m}")


def _verify_numeric_cols(df, num_cols):
    """check if numeric columns are valid

    Parameters
    ----------
    df : pandas.DataFrame
        the data set
    num_cols : list
        list of numeric column names

    Raises
    ------
    Exception
        if any of the column name is invalid/not a numeric column
    """
    all_num_cols = get_numeric_columns(df)
    # check if the column names are valid
    for c in num_cols:
        if not (c in all_num_cols):
            raise Exception(f"Invalid numeric column name: {c}")


def get_clustering_default_hyperparameters():
    """create a dictionary listing default hyperparameters for K-Means and DBSCAN clustering

    Returns
    -------
    dict
        a dictionary with key = clustering algorithm name, value = a dictionary of hyperameters' name and value

    Examples
    -------
    >>> hyper_dict = get_clustering_default_hyperparameters()
    >>> hyper_dict["KMeans"]["n_clusters"] = range(2, 10)
    >>> hyper_dict["DBSCAN"]["eps"] = [0.3]
    >>> hyper_dict["DBSCAN"]["min_samples"] = [3]
    >>> hyper_dict["DBSCAN"]["distance_metric"] = "cosine"
    >>> explore_clustering(X, hyperparameter_dict=hyper_dict)
    """
    clustering_default_hyperparameters = {
        "KMeans": {"n_clusters": range(2, 6)},
        "DBSCAN": {
            "eps": [0.5],
            "min_samples": [5],
            "distance_metric": "euclidean",
        },
    }
    return clustering_default_hyperparameters


def plot_pca_clusters(data, labels, random_state=None):
    """Carries out dimensionality reduction on the data for visualization, apdated from Lecture 2

    Parameters
    ----------
    data : pandas.DataFrame
        the dataset
    labels : list
        list of labels predicted
    random_state : int, optional
        a number determines random number generation for centroid initialization, by default None

    Returns
    -------
    matplotlib.axes.AxesSubPlot
        the PCA plot
    """
    pca = PCA(n_components=2, random_state=random_state)
    principal_comp = pca.fit_transform(data)
    pca_df = pd.DataFrame(
        data=principal_comp, columns=["pca1", "pca2"], index=data.index
    )
    pca_df["cluster"] = labels
    # fig = plt.figure(figsize=(6, 4))
    fig = sns.scatterplot(
        x="pca1", y="pca2", hue="cluster", data=pca_df, palette="tab10"
    )
    plt.show()
    plt.close()

    return fig


# endregion

# region clustering functions


def explore_KMeans_clustering(
    df,
    num_cols=None,
    n_clusters=range(3, 5),
    include_silhouette=True,
    include_PCA=True,
    random_state=None,
):
    """create, fit and plot KMeans clustering on the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset, should be transformed with StandardScaler
    num_cols : list, optional
        list of numeric column names, in case of None, get all numeric columns
    metric : str, optional
        metric, by default "euclidean"
    n_clusters : list, optional
        list of n_clusters hyperparams, by default range(2, 9)
    include_silhouette : bool, optional
        whether Silhouette plots should be generated, by default True
    include_PCA : bool, optional
        whether PCA plots should be generated, by default True
    random_state : int, optional
        a number determines random number generation for centroid initialization, by default None

    Returns
    -------
    dict
        a dictionary with key=type of plot, value=list of plots

    Examples
    -------
    >>> original_df = pd.read_csv("/data/menu.csv")
    >>> numeric_features = eda.get_numeric_columns(original_df)
    >>> numeric_transformer = make_pipeline(SimpleImputer(), StandardScaler())
    >>> preprocessor = make_column_transformer(
    >>>     (numeric_transformer, numeric_features)
    >>> )
    >>> df = pd.DataFrame(
    >>>     data=preprocessor.fit_transform(original_df), columns=numeric_features
    >>> )
    >>> explore_KMeans_clusterting(df)
    """
    if num_cols is None:
        num_cols = get_numeric_columns(df)
    else:
        _verify_numeric_cols(df, num_cols)
    x = df[num_cols]
    results = {}
    if 1 in n_clusters:
        raise Exception("n_cluster cannot be 1")

    print("------------------------")
    print("K-MEANS CLUSTERING")
    print("------------------------")

    if len(n_clusters) > 1:
        print("Generating KElbow plot for KMeans.")
        # visualize using KElbowVisualizer
        kmeans = KMeans(random_state=random_state)

        plt.clf()
        fig, ax = plt.subplots()
        elbow_visualizer = KElbowVisualizer(kmeans, k=n_clusters, ax=ax)
        elbow_visualizer.fit(x)  # Fit the data to the visualizer
        elbow_visualizer.show()
        plt.close()
        elbow_visualizer.k = elbow_visualizer.elbow_value_  # fix printing issue
        results["KElbow"] = fig
    else:
        results["KElbow"] = None

    # visualize using SilhouetteVisualizer
    print("Generating Silhouette & PCA plots")
    silhouette_plots = []
    pca_plots = []
    for k in n_clusters:
        print(f"Number of clusters: {k}")

        kmeans = KMeans(k, random_state=random_state)

        if include_silhouette:
            fig, ax = plt.subplots()
            s_visualizer = SilhouetteVisualizer(kmeans, colors="yellowbrick", ax=ax)
            s_visualizer.fit(x)  # Fit the data to the visualizer
            s_visualizer.show()
            silhouette_plots.append(fig)
            # plt.clf()
            plt.close()

        else:
            silhouette_plots.append(None)

        # PCA plots
        if include_PCA:
            labels = kmeans.fit_predict(x)
            pca_fig = plot_pca_clusters(x, labels, random_state=random_state)
            pca_plots.append(pca_fig)
        else:
            pca_plots.append(None)

    results["Silhouette"] = silhouette_plots
    results["PCA"] = pca_plots

    return results


def explore_DBSCAN_clustering(
    df,
    num_cols=None,
    metric="euclidean",
    eps=[0.5],
    min_samples=[5],
    include_silhouette=True,
    include_PCA=True,
    random_state=None,
):
    """fit and plot DBSCAN clustering algorithms

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset, should be transformed with StandardScaler
    num_cols : list, optional
        list of numeric column names, in case of None, get all numeric columns
    metric : str, optional
        metric, by default "euclidean"
    eps : list, optional
        list of eps hyperparams, by default [0.5]
    min_samples: list, optional
        list of min_samples hyperparams, by default [5]
    include_silhouette : bool, optional
        whether Silhouette plots should be generated, by default True
    include_PCA : bool, optional
        whether PCA plots should be generated, by default True
    random_state : int, optional
        a number determines random number generation for centroid initialization, by default None

    Returns
    -------
    Tuple
        list
            a list of n_clusters values returned by DBSCAN models
        dict
            a dictionary with key=type of plot, value=list of plots

    Examples
    -------
    >>> original_df = pd.read_csv("/data/menu.csv")
    >>> numeric_features = eda.get_numeric_columns(original_df)
    >>> numeric_transformer = make_pipeline(SimpleImputer(), StandardScaler())
    >>> preprocessor = make_column_transformer(
    >>>     (numeric_transformer, numeric_features)
    >>> )
    >>> df = pd.DataFrame(
    >>>     data=preprocessor.fit_transform(original_df), columns=numeric_features
    >>> )
    >>> n_clusters, dbscan_plots = explore_DBSCAN_clusterting(df)
    """
    if num_cols is None:
        num_cols = get_numeric_columns(df)
    else:
        _verify_numeric_cols(df, num_cols)

    x = df[num_cols]

    results = {}
    n_clusters = []

    s_plots = []
    pca_plots = []

    print("------------------------")
    print("DBSCAN CLUSTERING")
    print("------------------------")

    for e in eps:
        for ms in min_samples:
            dbscan = DBSCAN(eps=e, min_samples=ms, metric=metric)
            dbscan.fit(x)
            k = len(set(dbscan.labels_)) - 1  # exclduing -1 labels
            n_clusters.append(k)
            print(f"eps={e}, min_samples={ms}, n_cluster={k}")
            if include_silhouette and k > 0:
                # generat Silhouette plot
                dbscan.n_clusters = k
                dbscan.predict = lambda x: dbscan.labels_
                fig, ax = plt.subplots()
                s_visualizer = SilhouetteVisualizer(dbscan, colors="yellowbrick", ax=ax)
                s_visualizer.fit(x)
                s_visualizer.show()
                s_plots.append(fig)
                # plt.clf()
                plt.close()
            else:
                s_plots.append(None)

            if include_PCA:
                # genrate PCA plot
                p_lot = plot_pca_clusters(x, dbscan.labels_, random_state=random_state)
                pca_plots.append(p_lot)
            else:
                pca_plots.append(None)

    results["Silhouette"] = s_plots
    results["PCA"] = pca_plots

    return n_clusters, results


def explore_clustering(
    df,
    numeric_cols=None,
    hyperparameter_dict=get_clustering_default_hyperparameters(),
    random_state=None,
):
    """fit and plot K-Means, DBScan clustering algorithm on the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset (X), should already be transformed with StandardScaler
    numeric_cols : list, optional
        a list of numeric columns used for clustering, by default None, will be assigned with all numeric columns
    hyperparameter_dict : dict, optional
        the hyperparameters to be used in the clustering algorithms, by default use
        {
            "KMeans": {"n_clusters": range(2, 6)},
            "DBSCAN": {
                "eps": [0.5],
                "min_samples": [5],
                "distance_metric": "euclidean",
            }
        }
    random_state : int, optional
        a number determines random number generation for centroid initialization, by default None

    Returns
    -------
    dict
        a dictionary with each key = a clustering model name, value = list of plots generated by that model

    Examples
    -------
    >>> original_df = pd.read_csv("data/menu.csv")
    >>> numeric_features = eda.get_numeric_columns(original_df)
    >>> numeric_transformer = make_pipeline(SimpleImputer(), StandardScaler())
    >>> preprocessor = make_column_transformer(
    >>>     (numeric_transformer, numeric_features)
    >>> )
    >>> df = pd.DataFrame(
    >>>     data=preprocessor.fit_transform(original_df), columns=numeric_features
    >>> )
    >>> explore_clustering(df)
    """

    # region validate parameters, throw exception upon invalid ones

    if not (type(df) == pd.DataFrame):
        raise TypeError("df must be a DataFrame.")

    all_num_cols = get_numeric_columns(df)
    if numeric_cols is None:
        numeric_cols = all_num_cols
    else:
        # check if the column names are valid
        _verify_numeric_cols(df, numeric_cols)

    if not (type(hyperparameter_dict) == dict):
        raise TypeError("hyperparameter_dict must be a dict.")

    if not ("KMeans" in hyperparameter_dict):
        raise Exception("Expecting Kmeans hyperparams.")

    if not ("DBSCAN" in hyperparameter_dict):
        raise Exception("Expecting DBSCAN hyperparams.")

    kmeans_params = hyperparameter_dict["KMeans"]
    if not ("n_clusters" in kmeans_params):
        raise Exception("Expecting n_clusters in KMeans' hyperparams.")

    dbscan_params = hyperparameter_dict["DBSCAN"]
    if not ("eps" in dbscan_params):
        raise Exception("Expecting eps in DBSCAN's hyperparams.")

    if not ("min_samples" in dbscan_params):
        raise Exception("Expecting min_samples in DBSCAN's hyperparams.")

    if not ("distance_metric" in dbscan_params):
        raise Exception("Expecting distance_metric as a hyperparameter")

    metric = dbscan_params["distance_metric"]
    _verify_distance_metric(metric)

    # endregion

    print("***********************")
    print("EXPLORE CLUSTERING")
    print("***********************")

    kmeans_plots = explore_KMeans_clustering(
        df,
        num_cols=numeric_cols,
        n_clusters=kmeans_params["n_clusters"],
        random_state=random_state,
    )

    dbscan_plots = explore_DBSCAN_clustering(
        df,
        num_cols=numeric_cols,
        metric=metric,
        eps=dbscan_params["eps"],
        min_samples=dbscan_params["min_samples"],
        include_PCA=True,
        include_silhouette=True,
    )

    result = {}  # a dictionary to store charts generated by clustering models
    result["KMeans"] = kmeans_plots
    result["DBSCAN"] = dbscan_plots

    print("***********************")
    print("FINISHED CLUSTERING")
    print("***********************")

    return result


def printmd(string):
    """Displays the markdown representation of the
    string passed to it

    Parameters
    ----------
    string : str
        the string to be displayed using markdown syntax

    Returns
    -------
    None

    Examples
    -------
    >>> printmd("### I am Batman")
    """
    display(Markdown(string))


def explore_text_columns(df, text_col=None):
    """Performs EDA of text features.
    - prints the summary statistics of character length
    - plots the distribution of character length
    - prints the summary statistics of word count
    - plots the distribution of word count
    - plots the word cloud
    - plots bar chart of top n stopwords
    - plots bar chart of top n words other than stopwords
    - plots bar chart of top n bigrams
    - plots the distribution of polarity and subjectivity scores
    - plots bar charts of sentiments, name entities and part of speech tags

    Parameters
    ----------
    df : pandas.DataFrame
        the dataset (X)
    text_col : list
        name of text column(s) as list of string(s)

    Returns
    -------
    list
        A list of plot objects created by this function

    Examples
    -------
    >>> explore_text_columns(X)
    """
    result = []

    # exception if df is not a pandas dataframe
    if type(df) != pd.core.frame.DataFrame:
        raise Exception("df is not a Pandas Dataframe")

    # identify text columns if not specified by user
    if text_col is None:
        text_col = []
        non_num = df.columns[df.dtypes == ("object" or "string")]
        for col in non_num:
            if df[col].unique().shape[0] / df.shape[0] > 0.75:
                if df[col].str.split().apply(len).median() > 5:
                    text_col.append(col)

        # exception if text column cannot be identified
        if not text_col:
            raise Exception(
                "Could not identify any text column. Please pass the text column(s) when calling the function"
            )
        else:
            print("Identified the following as text columns:", text_col)
            result.append(text_col)

    # exception if text_col is not passed as a list
    elif type(text_col) is not list:
        raise Exception("text_col is not a list. Pass the text column(s) as a list")

    # exception if a column passes in text_col is not in df
    else:
        for col in text_col:
            if col not in df.columns.values:
                raise Exception(f"{col} is not a column in the dataframe")

    # print average, minimum, maximum and median character length of text
    # show the shortest and longest text (number of characters)
    print("\n")
    for col in text_col:

        printmd('## Exploratory Data Analysis of "' + col + '" column:<br>')

        printmd("### Character Length:<br>")

        mean_char_length = df[col].str.len().mean()
        median_char_length = df[col].str.len().median()
        longest_char_length = df[col].str.len().max()
        longest_text = df[col][df[col].str.len() == longest_char_length].unique()
        shortest_char_length = df[col].str.len().min()
        shortest_text = df[col][df[col].str.len() == shortest_char_length].unique()

        printmd(f"- The average character length of text is {mean_char_length:.2f}")
        printmd(f"- The median character length of text is {median_char_length:.0f}")

        printmd(f"- The longest text(s) has {longest_char_length:.0f} characters:\n")

        for text in longest_text:
            printmd('"' + text + '"')

        printmd(f"- The shortest text(s) has {shortest_char_length:.0f} characters:\n")

        for text in shortest_text:
            printmd('"' + text + '"<br><br>')

        result.append(
            [
                round(mean_char_length, 2),
                median_char_length,
                longest_char_length,
                longest_text[0],
                shortest_char_length,
                shortest_text[0],
            ]
        )

        # plot a histogram of the length of text (number of characters)
        printmd(f'#### Histogram of number of characters in "{col}":')
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({"figure.figsize": (12, 8)})
        char_length_plot = sns.histplot(data=df[col].str.len())
        plt.xlabel("Number of characters in " + '"' + col + '"')
        result.append(char_length_plot)
        plt.show()
        plt.close()

        # print average, minimum, maximum and median number of words
        # show text with most number of words
        printmd("### Word Count:<br>")

        mean_word_count = df[col].str.split().apply(len).mean()
        median_word_count = df[col].str.split().apply(len).median()
        highest_word_count = df[col].str.split().apply(len).max()
        text_most_words = df[col][
            df[col].str.split().apply(len) == highest_word_count
        ].unique()

        printmd(f'- The average number of words in "{col}": {mean_word_count:.2f}')
        printmd(f'- The median number of words in "{col}": {median_word_count:.0f}')

        printmd(
            f'- The text(s) in "{col}" with most words ({highest_word_count:.0f} words):\n'
        )

        for text in text_most_words:
            printmd('"' + text + '"')

        result.append(
            [
                round(mean_word_count, 2),
                median_word_count,
                highest_word_count,
                text_most_words,
            ]
        )

        # plot a histogram of the number of words
        printmd(f'#### Histogram of number of words in "{col}":')
        word_count_plot = sns.histplot(data=df[col].str.split().apply(len))
        plt.xlabel("Number of words in " + '"' + col + '"')
        result.append(word_count_plot)
        plt.show()
        plt.close()
        printmd("<br>")

        # plot word cloud of text
        printmd("### Word Cloud:<br>")
        wordcloud = WordCloud(random_state=1).generate(" ".join(df[col]))
        plt.rcParams.update({"figure.figsize": (12, 8)})
        wordcloud_plot = plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        result.append(wordcloud_plot)
        plt.show()
        plt.close()

        printmd("<br>")

        # plot a bar chart of the top stopwords
        stop = set(stopwords.words("english"))
        all_words = df[col].str.split()
        all_words = all_words.values.tolist()
        corpus = [word for i in all_words for word in i]
        corpus = pd.DataFrame(
            pd.DataFrame(corpus, columns=["counts"]).counts.value_counts()
        ).reset_index()
        corpus.columns = ["words", "counts"]

        all_stopwords = corpus.merge(
            pd.DataFrame(stop, columns=["words"]), on="words", how="right"
        )

        printmd("### Bar Chart of the top stopwords:<br>")
        stopwords_plot = sns.barplot(
            y="words",
            x="counts",
            data=all_stopwords.sort_values(by="counts", ascending=False).head(10),
        )
        plt.ylabel("Stop Words")
        plt.xlabel("Count")
        result.append(stopwords_plot)
        plt.show()
        plt.close()

        # plot a bar chart of words other than stopwords
        left_joined = corpus.merge(
            pd.DataFrame(stop, columns=["words"]),
            on="words",
            how="left",
            indicator=True,
        )
        non_stopwords = left_joined[left_joined["_merge"] == "left_only"]
        top_non_stopwords = non_stopwords.sort_values(
            by="counts", ascending=False
        ).head(10)

        printmd("### Bar Chart of the top non-stopwords:<br>")
        non_stopwords_plot = sns.barplot(y="words", x="counts", data=top_non_stopwords)
        plt.ylabel("Non Stop Words")
        plt.xlabel("Count")
        result.append(non_stopwords_plot)
        plt.show()
        plt.close()

        # plot a bar chart of top bigrams
        vec = CountVectorizer(ngram_range=(2, 2)).fit(df[col])
        bag_of_words = vec.transform(df[col])
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [
            (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
        ]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top_bi_grams = words_freq[:10]
        x, y = map(list, zip(*top_bi_grams))

        printmd("### Bar Chart of the top Bi-grams:<br>")
        bi_gram_plot = sns.barplot(x=y, y=x)
        plt.ylabel("Bi-grams")
        plt.xlabel("Count")
        result.append(bi_gram_plot)
        plt.show()
        plt.close()

        # plot the distribution of polarity scores
        polarity_scores = df[col].apply(lambda x: TextBlob(x).sentiment.polarity)

        printmd("### Distribution of Polarity scores:<br>")
        plarity_scores_plot = sns.histplot(data=polarity_scores, bins=15)
        plt.xlabel("Polarity scores in " + '"' + col + '"')
        result.append(plarity_scores_plot)
        plt.show()
        plt.close()

        # plot a bar chart of sentiments: Positive, Negative and Neutral
        polarity = polarity_scores.apply(
            lambda x: "Negative" if x < 0 else ("Neutral" if x == 0 else "Positive")
        )

        printmd("### Bar chart of Sentiments:<br>")
        sentiment_plot = sns.countplot(
            x="sms",
            data=pd.DataFrame(polarity),
            order=["Negative", "Neutral", "Positive"],
        )
        plt.ylabel("Count")
        plt.xlabel("Sentiments")
        result.append(sentiment_plot)
        plt.show()
        plt.close()

        # plot the distribution of subjectivity scores
        subjectivity_scores = df[col].apply(
            lambda x: TextBlob(x).sentiment.subjectivity
        )

        printmd("### Distribution of Subjectivity scores:<br>")
        subjectivity_plot = sns.histplot(data=subjectivity_scores, bins=15)
        plt.xlabel("Subjectivity scores in " + '"' + col + '"')
        result.append(subjectivity_plot)
        plt.show()
        plt.close()

        # plot a bar chart of named entities
        nlp = en_core_web_md.load()
        ent = df[col].apply(lambda x: [X.label_ for X in nlp(x).ents])
        ent = [x for sub in ent for x in sub]
        ent_counter = Counter(ent)

        ent_count_df = pd.DataFrame.from_dict(ent_counter, orient="index").reset_index()
        ent_count_df.columns = ["Entity", "Count"]
        ent_count_df = ent_count_df.sort_values(by="Count", ascending=False)

        printmd("### Bar Chart of Named Entities:<br>")
        entity_plot = sns.barplot(y="Entity", x="Count", data=ent_count_df)
        plt.ylabel("Entity")
        plt.xlabel("Count")
        result.append(entity_plot)
        plt.show()
        plt.close()

        # plot a bar chart of most common tokens per entity
        tokens = ["PERSON", "GPE", "ORG"]
        c = 0
        entity_token = [None] * len(tokens)
        for token in tokens:

            token_list = df[col].apply(
                lambda x: [X.text for X in nlp(x).ents if X.label_ == token]
            )
            token_list = [i for x in token_list for i in x]
            token_counter = Counter(token_list)

            token_count_df = pd.DataFrame.from_dict(
                token_counter, orient="index"
            ).reset_index()
            token_count_df.columns = [token, "Count"]
            token_count_df = token_count_df.sort_values(by="Count", ascending=False)

            printmd("### Bar Chart of the token- " + '"' + token + '"' + ":<br>")
            entity_token[c] = sns.barplot(
                y=token, x="Count", data=token_count_df.head(10)
            )
            plt.ylabel(token)
            plt.xlabel("Count")
            result.append(entity_token[c])
            plt.show()
            plt.close()
            c = c + 1

        # plot a bar chart of Part-of-speech tags
        tags = df[col].apply(lambda x: [tags.pos_ for tags in nlp(x)])
        tags = [x for sub in tags for x in sub]
        tag_counter = Counter(tags)

        tag_count_df = pd.DataFrame.from_dict(tag_counter, orient="index").reset_index()
        tag_count_df.columns = ["Tags", "Count"]
        tag_count_df = tag_count_df.sort_values(by="Count", ascending=False)

        printmd("### Bar Chart of Part of Speech Tags:<br>")
        pos_plot = sns.barplot(y="Tags", x="Count", data=tag_count_df)
        plt.ylabel("Part of Speech Tags")
        plt.xlabel("Count")
        result.append(pos_plot)
        plt.show()
        plt.close()

    return result


def explore_numeric_columns(
    data, hist_cols=None, pairplot_cols=None, corr_method="pearson"
):
    """This function will create common exploratory analysis visualizations on numeric columns in the dataset which is provided to it.

    The visualizations that will be created are:

    1. Histograms for all numeric columns or for columns specified in optional paramter `hist_cols`
    2. Scatterplot Matrix (SPLOM) for all numeric columns or for columns specified in optional paramter `hist_cols`
    3. Heatmap showing correlation coefficient (pearson, kendall or spearman) between all numeric columns

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe for which exploratory analysis is to be carried out
    hist_cols : list, optional
        If passed, it will limit histograms to a subset of columns
    pairplot_cols : list, optional
        If passed, it will limit pairplots to a subset of columns
    corr_method : str, optional
        Chooses the metric for correlation. Default value is 'pearson'. Possible values:
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation

    Returns
    -------
    list
        A list of plot objects created by this function

    Examples
    -------
    >>> explore_numeric_columns(df)
    """

    # Generate plots

    if not (type(data) == pd.DataFrame):
        raise TypeError("data must be passed as a DataFrame.")

    if type(hist_cols) != list and hist_cols is not None:
        raise TypeError("hist_cols must be passed as a list.")

    if type(pairplot_cols) != list and pairplot_cols is not None:
        raise TypeError("pairplot_cols must be passed as a list.")

    if type(corr_method) != str and corr_method is not None:
        raise TypeError("corr_method must be passed as a str.")

    cols = get_numeric_columns(data)

    # Generate plots

    plots = {}  # Dictionary with results

    # Create histograms
    histograms = []
    cols = data.select_dtypes(include=np.number).columns.tolist()
    if hist_cols is None:
        for col in cols:
            chart = (
                alt.Chart(data)
                .encode(alt.X(col), alt.Y("count()"))
                .mark_bar()
                .properties(title="Histogram for " + col)
            )
            plt.figure()
            print(chart)
            histograms.append(chart)
    else:
        _verify_numeric_cols(data, hist_cols)
        for col in hist_cols:
            chart = (
                alt.Chart(data)
                .encode(alt.X(col), alt.Y("count()"))
                .mark_bar()
                .properties(title="Histogram for " + col)
            )
            plt.figure()
            print(chart)
            histograms.append(chart)

    plots["hist"] = histograms

    # Create pairplots
    if pairplot_cols is None:
        chart = sns.pairplot(data)
        chart.fig.suptitle("Pairplot between numeric features", y=1.08)
        plt.figure()
        print(chart)
        plots["pairplot"] = chart
    else:
        _verify_numeric_cols(
            data, pairplot_cols
        )  # Check that each column passed is numeric
        chart = sns.pairplot(data, vars=pairplot_cols)
        chart.fig.suptitle("Pairplot between numeric features", y=1.08)
        plt.figure()
        print(chart)
        plots["pairplot"] = chart

    # Show heatmap with correlation coefficient
    if corr_method not in [None, "pearson", "spearman", "kendall"]:
        raise ValueError(
            f"Value for input 'corr_method' should be either None, 'pearson', 'spearman' or 'kendall'. '{corr_method}' was provided."
        )
    chart = sns.heatmap(data.corr(method=corr_method), cmap="coolwarm", center=0)
    plt.title("Heatmap showing correlation between numeric features")
    plt.figure()
    print(chart)
    plots["corr"] = chart

    return plots


def explore_categorical_columns(df, categorical_cols):
    """Performs EDA of categorical features.
    - Creates a dataframe containing column names and corresponding details about unique values, null values and most frequent category in the column
    - Plots countplots for given categorical columns
    Parameters
    ----------
    df : pandas.DataFrame
        the dataset (X)
    categorical_col : list
        name of categorical column(s)
    Returns
    -------
    cat_df: pandas.DataFrame
        A Dataframe with column names, corresponding unique categories, count of null values, percentage of null values and most frequent categories

    cat_plots: list
        A list having countplots of given categorical columns
    Examples
    -------
    >>> explore_categorical_columns(X, ['col1', 'col2'])
    """
    # exception if df is not a pandas dataframe
    if type(df) != pd.core.frame.DataFrame:
        raise Exception("df is not a Pandas Dataframe")

    # exception if categorical_cols is not passed as a list
    if type(categorical_cols) is not list:
        raise Exception(
            "categorical_cols is not a list. Pass the categorical column(s) as a list"
        )

    # exception if categorical_cols is not in columns of dataframe
    for col in categorical_cols:
        if col not in df.columns.values:
            raise Exception(f"{col} is not a column in the dataframe")

    cat_df = pd.DataFrame(
        columns=["column_name", "unique_items", "no_of_nulls", "percentage_missing"]
    )
    temp = pd.DataFrame()

    # Creating Dataframe
    for col in categorical_cols:
        temp["column_name"] = [col]
        temp["unique_items"] = [df[col].unique()]
        temp["no_of_nulls"] = df[col].isnull().sum()
        temp["percentage_missing"] = (df[col].isnull().sum() / len(df)).round(3) * 100
        cat_df = cat_df.append(temp)

    # Plotting
    # printmd('### Univariate Categorical Analysis')
    sns.set(style="darkgrid")
    sns.set(rc={"figure.figsize": (22, 5)})
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    cat_plots = []

    for col in categorical_cols:
        # printmd(f'##### Column Name: {col}')
        ax = sns.countplot(x=col, data=df, order=df[col].value_counts().index)
        ax.set_xlabel(col, fontsize=15)
        ax.set_ylabel("Count", fontsize=15)
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate(
                "{:.1f}%".format(100.0 * y / df.shape[0]),
                (x.mean(), y),
                ha="center",
                va="bottom",
            )
        #        plt.xticks(rotation=20)
        cat_plots.append(ax)
        plt.close()
    # printmd('### Dataframe containing unique items in each column and corresponding percentage as well as number of nulls')
    cat_df.set_index("column_name", inplace=True)
    cat_df.sort_values(by="percentage_missing", inplace=True, ascending=False)
    return pd.DataFrame(cat_df), cat_plots
