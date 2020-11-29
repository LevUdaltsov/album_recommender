import random

from django.shortcuts import render

from rec_system.models import Album
from .scripts.create_db import upload_data_to_db


def show_album(request, album_id):
    upload_data_to_db()
    len_albums = len(Album.objects.all())
    random_albums = random.sample(range(1, len_albums), 40)
    albums = Album.objects.filter(pk__in=random_albums)
    context = {}

    for album in albums:
        print(album)

    context['albums'] = list(albums)

    return render(request, 'rec_system/page_1.html', context)


# def show_album_list(request):
#     albums = Album.objects.all()


