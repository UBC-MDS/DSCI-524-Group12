from datascience_eda import __version__
from datascience_eda import datascience_eda

import pytest
import pandas as pd


@pytest.fixture
def create_test_dataset():
    """create a test dataset

    Returns
    -------
    [pandas.DataFrame]
        a data set used for testing all functions
    """
    return pd.read_csv("data/menu.csv")


def test_version():
    assert __version__ == "0.1.0"


def test_explore_clustering():
    raise NotImplementedError()


def test_explore_KMeans_clustering():
    df = create_test_dataset()
    n_clusters_key = "n_clusters"
    n_clusters = (3, 5)
    hyperparams = {n_clusters_key: n_clusetrs}

    n_combs = len(n_clusters)

    plots = explore_KMeans_clustering(
        df, hyperparams, include_silhoutte=True, include_PCA=True
    )

    assert (
        len(plots) == 3
    ), "Expecting Elbow Method Plot, Silhouette Plots and PCA Plots."

    # verify Elbow plot
    elbow_plot = plots[0]
    n_rows = df.shape[0]
    assert type(elbow_plot) == "yellowbrick.cluster.elbow.KElbowVisualizer"
    assert elbow_plot.elbow_value_ in n_clusters, "Invalid value for K."

    # verify Sihoutte Plots
    silhouette_plots = plots[1]
    assert (
        len(silhoutte_plots) == n_combs
    ), "Expecting one Silhouette plot for each value of n_clusters"

    for i in range(len(silhouette_plots)):
        s_plot = silhouette_plots[i]
        assert (
            type(s_plot) == "yellowbrick.cluster.silhouette.SilhouetteVisualizer"
        ), "Invalid Silhouette plot."

        assert (
            s_plot.ax.title.get_text()
            == f"Silhouette Plot of KMeans Clustering for {n_rows} Samples in {n_clusters[i]} Centers"
        )

    # verify PCA Plots
    pca_plots = plots[2]
    assert (
        len(pca_plots) == n_combs
    ), "Expecting one PCA plot for each value of n_clusters"
    for i in range(len(pca_plots)):
        p_plot = pca_plots[i]
        assert type(p_plot) == "matplotlib.figure.Figure"


def test_explore_DBSCAN_clustering():
    raise NotImplementedError()