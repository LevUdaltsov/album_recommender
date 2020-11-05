from django.shortcuts import render

from rec_system.models import Album
from .scripts.create_db import upload_data_to_db


def show_album(request, album_id):
    upload_data_to_db()    
    albums = Album.objects.all()

    for album in albums:
        print(album)
    context = {'album':album.name,
            'artist':album.artist}
    
    return render(request, 'rec_system/page_1.html', context)