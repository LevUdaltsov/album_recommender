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
    pub_date = models.DateField(default=None)
    artist = models.ForeignKey(Artist, on_delete=models.CASCADE)
    content = models.TextField(default='')

    url = models.CharField(max_length=255, default=None)
    score = models.IntegerField(default=None)
    best_new_music = models.BooleanField(max_length=255, default=False)

    def __str__(self):
        return self.name
