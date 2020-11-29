"""
This is a module that populates the database with album models and review

"""


import pandas as pd
import os
import requests
from datetime import datetime
from sqlalchemy import create_engine
from bs4 import BeautifulSoup

from django.conf import settings

from rec_system.models import Artist, Album


def upload_data_to_db():

    base_dir = settings.BASE_DIR
    reviews_csv = os.path.join(base_dir, 'rec_system/media/pitchfork_reviews.csv')
    review_data = pd.read_csv(reviews_csv)
    review_data.rename(columns={'title': 'name'}, inplace=True)

    review_data = review_data[['name', 'pub_date', 'artist', 'content', 'url', 'score', 'best_new_music']]
    review_data['pub_date'] = [datetime.strptime(item, '%Y-%m-%d') for item in review_data['pub_date'].values]
    review_data['score'] = [int(score) for score in review_data['score'].values]
    review_data['best_new_music'] = review_data['best_new_music'].astype(bool)
    
    for _, review in review_data.iterrows():

        response = requests.get(review['url'])
        pageinfo= BeautifulSoup(response.content,
        "html.parser")
        imgurl = pageinfo.find('img').attrs['src']   
        
        artist = Artist(name=review['artist'])
        album = Album(name=review['name'],
                    pub_date=review['pub_date'],
                    artist=artist,
                    content=review['content'],
                    score=review['score'],
                    url=review['url'],
                    imgurl=imgurl,
                    best_new_music=review['best_new_music'])
        
        artist.save()
        album.save()


    

    

        




    
    
    

