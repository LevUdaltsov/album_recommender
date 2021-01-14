""" 
    This is a script that contains functions that handle data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(data_loc):

    data = pd.read_csv(data_loc, error_bad_lines=False)
    documents = data[['artist', 'album', 'review', 'score']]
    documents = documents.dropna(subset=['review']).reset_index(drop=True)

    return documents


def tfidf_generator(documents, num_features):

    tfidf = TfidfVectorizer(max_features=num_features,
                            lowercase=True,
                            analyzer='word',
                            stop_words='english',
                            ngram_range=(1,1))
    
    data_tfidf = tfidf.fit_transform(list(documents['review'].values))
    tfidf_tokens = tfidf.get_feature_names()

    return data_tfidf, tfidf_tokens


def create_df(documents, data_tfidf, tfidf_tokens):

    return pd.DataFrame(data=data_tfidf.toarray(),
                            index=documents['album'],
                            columns=tfidf_tokens)


def get_artist_albums(dataframe, artists):

    return dataframe[dataframe['artist'].isin(artists)]
    

def plot_encoded_albums(encoded_albums, album_df, labels=True):

    album_df['artist'] = album_df['artist'].astype('category')
    sample_doc_encoded = np.asarray([encoded_albums[item] for item in album_df.index])

    fig = plt.figure(1, figsize=(7, 7))

    ax = fig.add_subplot(1, 1, 1)

    for artist in album_df['artist'].unique():
        artist_df = album_df[album_df['artist'] == artist]
        artist_encoded = np.asarray([encoded_albums[item] for item in artist_df.index])

        if labels:
            for i, txt in enumerate(list(artist_df['album'].values)[:3]):
                ax.text(artist_encoded[i, 0], artist_encoded[i, 1]-0.01, txt, fontsize=6,
                ha='center', va='center')
        

        scatter= ax.scatter(artist_encoded[:, 0], artist_encoded[:, 1], label=artist)
    ax.legend(title="Artist")

    
    ax.dist = 12

def get_recommendation(documents, data, albums):
    good_reviews = documents[documents['score'] > 8.0]
    print(len(good_reviews))
    good_reviews_data = [data[i] for i in good_reviews.index]
    albums_idx = list(documents[documents.album.isin(albums)].index)
    list_of_points = [data[index] for index in albums_idx]
    mean_point = np.mean(list_of_points, axis=0)
    recommendation = good_reviews.iloc[closest_point(mean_point, good_reviews_data)]
    print(recommendation)
    albums.append(recommendation[1])
    plt.cla()
    plot_encoded_albums(data, documents[documents['album'].isin(albums)], labels=True)
    return recommendation

    


def closest_point(point, data):
    data = np.asarray(data)
    dist_2 = np.sum((data - point)**2, axis=1)
    return np.argmin(dist_2)