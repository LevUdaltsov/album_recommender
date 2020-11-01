from django.shortcuts import render

from .models import Song

# Create your views here.

def show_song(request, song_id):

    song = Song.objects.get(pk=song_id)

    context = {'song':song.name,
            'album':song.album,
            'artist':song.artist}
    
    return render(request, 'rec_system/page_1.html', context)