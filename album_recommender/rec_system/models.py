from django.db import models
from django.utils import timezone

# Create your models here.


class Artist(models.Model):
    """
    Artist model that stores data about artists.
    """

    name = models.CharField(max_length=255, verbose_name='Name')

    def __str__(self):
        return self.name

class Album(models.Model):
    """
    Album model that stores data about albums.

    """

    name = models.CharField(max_length=255, verbose_name='Name')
    release_date = models.DateTimeField(default=timezone.now)
    artist = models.ForeignKey(Artist, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Song(models.Model):
    """
    Song model that stores data about songs.

    """
    name = models.CharField(max_length=255, verbose_name='Name')
    album = models.ForeignKey(Album, on_delete=models.CASCADE)
    artist = models.ForeignKey(Artist, on_delete=models.CASCADE)
    length =  models.IntegerField(verbose_name='Name')
    description = models.TextField(default='')


