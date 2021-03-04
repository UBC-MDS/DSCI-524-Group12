from datascience_eda import __version__
from datascience_eda import datascience_eda as eda

import pytest
from pytest import raises
import pandas as pd

import os, sys, inspect

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.testing.compare import compare_images

matplotlib.use("Agg")  # use non-ui backend to close the plots

plt.ioff()  # disable interactive mode

import matplotlib.figure as mf

@pytest.fixture
def text_df():
    """create a test dataset for text features
    Returns
    -------
    [pandas.DataFrame]
        a data set for testing text features
    """
    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )

    df = pd.read_csv(currentdir + "/data/spam.csv", encoding="latin-1")
    df = df.rename(columns={"v1": "target", "v2": "sms"})

    return df

def test_explore_text_columns(text_df):
    """test explore_text_columns function
    Parameters
    ----------
    df : pandas.DataFrame
        test data
    """
    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )

    with raises(Exception):
        eda.explore_text_columns(['test'])

    with raises(Exception):
        eda.explore_text_columns(text_df.drop['sms'])

    with raises(Exception):
        eda.explore_text_columns(text_df, 'sms')

    with raises(Exception):
        eda.explore_text_columns(text_df, ['some_col_name'])

    result = eda.explore_text_columns(text_df)

    assert(result[0]==['sms'])

    assert(result[1]==[80.12, 61, 910, text_df['sms'][1084], 2, text_df['sms'][1924]])
    
    test_loc_hist_char = currentdir + "/test_plots/hist_char_length.png"
    result[2].figure.savefig(test_loc_hist_char)
    ref_loc_hist_char = currentdir + "/reference_plots/hist_char_length.png"

    assert((compare_images(test_loc_hist_char, ref_loc_hist_char, 0)) == None)

    assert(result[3]==[15.49, 12, 171, text_df['sms'][1084]])

    test_loc_hist_word = currentdir + "/test_plots/hist_word_count.png"
    result[4].figure.savefig(test_loc_hist_word)
    ref_loc_hist_word = currentdir + "/reference_plots/hist_word_count.png"

    assert((compare_images(test_loc_hist_word, ref_loc_hist_word, 0)) == None)

    test_loc_word_cloud = currentdir + "/test_plots/word_cloud.png"
    result[5].figure.savefig(test_loc_word_cloud)
    ref_loc_word_cloud = currentdir + "/reference_plots/word_cloud.png"

    assert((compare_images(test_loc_word_cloud, ref_loc_word_cloud, 0)) == None)

    test_loc_stopword = currentdir + "/test_plots/stopword.png"
    result[6].figure.savefig(test_loc_stopword)
    ref_loc_stopword = currentdir + "/reference_plots/stopword.png"

    assert((compare_images(test_loc_stopword, ref_loc_stopword, 0)) == None)

    test_loc_non_stopword = currentdir + "/test_plots/non_stopword.png"
    result[7].figure.savefig(test_loc_non_stopword)
    ref_loc_non_stopword = currentdir + "/reference_plots/non_stopword.png"

    assert((compare_images(test_loc_non_stopword, ref_loc_non_stopword, 0)) == None)

    test_loc_bi_gram = currentdir + "/test_plots/bi_gram.png"
    result[8].figure.savefig(test_loc_bi_gram)
    ref_loc_bi_gram = currentdir + "/reference_plots/bi_gram.png"

    assert((compare_images(test_loc_bi_gram, ref_loc_bi_gram, 0)) == None)

    test_loc_polarity_scores = currentdir + "/test_plots/polarity_scores.png"
    result[9].figure.savefig(test_loc_polarity_scores)
    ref_loc_polarity_scores = currentdir + "/reference_plots/polarity_scores.png"

    assert((compare_images(test_loc_polarity_scores, ref_loc_polarity_scores, 0)) == None)

    test_loc_sentiment = currentdir + "/test_plots/sentiment.png"
    result[10].figure.savefig(test_loc_sentiment)
    ref_loc_sentiment = currentdir + "/reference_plots/sentiment.png"

    assert((compare_images(test_loc_sentiment, ref_loc_sentiment, 0)) == None)

    test_loc_subjectivity = currentdir + "/test_plots/subjectivity.png"
    result[11].figure.savefig(test_loc_subjectivity)
    ref_loc_subjectivity = currentdir + "/reference_plots/subjectivity.png"

    assert((compare_images(test_loc_subjectivity, ref_loc_subjectivity, 0)) == None)

    test_loc_entity = currentdir + "/test_plots/entity.png"
    result[12].figure.savefig(test_loc_entity)
    ref_loc_entity = currentdir + "/reference_plots/entity.png"

    assert((compare_images(test_loc_entity, ref_loc_entity, 0)) == None)

    test_loc_entity_token_1 = currentdir + "/test_plots/entity_token_1.png"
    result[13].figure.savefig(test_loc_entity_token_1)
    ref_loc_entity_token_1 = currentdir + "/reference_plots/entity_token_1.png"

    assert((compare_images(test_loc_entity_token_1, ref_loc_entity_token_1, 0)) == None)

    test_loc_entity_token_2 = currentdir + "/test_plots/entity_token_2.png"
    result[14].figure.savefig(test_loc_entity_token_2)
    ref_loc_entity_token_2 = currentdir + "/reference_plots/entity_token_2.png"

    assert((compare_images(test_loc_entity_token_2, ref_loc_entity_token_2, 0)) == None)

    test_loc_entity_token_3 = currentdir + "/test_plots/entity_token_3.png"
    result[15].figure.savefig(test_loc_entity_token_3)
    ref_loc_entity_token_3 = currentdir + "/reference_plots/entity_token_3.png"

    assert((compare_images(test_loc_entity_token_3, ref_loc_entity_token_3, 0)) == None)

    test_loc_pos_plot = currentdir + "/test_plots/pos_plot.png"
    result[16].figure.savefig(test_loc_pos_plot)
    ref_loc_pos_plot = currentdir + "/reference_plots/pos_plot.png"

    assert((compare_images(test_loc_pos_plot, ref_loc_pos_plot, 0)) == None)
