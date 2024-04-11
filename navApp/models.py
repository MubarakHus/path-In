from django.db import models

# Create your models here.
class mapImage(models.Model):
    title = models.CharField(max_length=100, null=True)
    path = models.ImageField(null=True)
    floor = models.IntegerField(null=True)
class graph(models.Model):
    vertices = models.IntegerField()
    graph = models.CharField(max_length=10000)
    floor = models.IntegerField(primary_key=True, default='0')
class Points(models.Model):
    id = models.IntegerField(primary_key=True)
    pointX = models.FloatField(null=True)
    pointY = models.FloatField(null=True)
    floor = models.IntegerField(null=True)
    alt = models.CharField(max_length=100, null=True)
    title = models.CharField(max_length=100, null=True)
class Lines(models.Model):
    startID = models.IntegerField(null=True)
    endID = models.IntegerField(null=True)
    Length = models.FloatField(null=True)
    floor = models.IntegerField(null=True)


