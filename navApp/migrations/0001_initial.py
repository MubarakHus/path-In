# Generated by Django 5.0.3 on 2024-03-19 13:26

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Lines',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start_ID', models.IntegerField()),
                ('end_ID', models.IntegerField()),
                ('length', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='mapImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=100)),
                ('path', models.ImageField(upload_to='')),
                ('floor', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Points',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Xcoordinate', models.FloatField()),
                ('Ycoordinate', models.FloatField()),
            ],
        ),
    ]
