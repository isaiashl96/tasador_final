# Generated by Django 2.1.2 on 2018-11-02 14:38

import django.contrib.gis.db.models.fields
import django.contrib.gis.geos.point
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('tasador_app', '0002_auto_20181102_0935'),
    ]

    operations = [
        migrations.AlterField(
            model_name='inmueble',
            name='location',
            field=django.contrib.gis.db.models.fields.PointField(default=django.contrib.gis.geos.point.Point(-77.0282, -12.0432, srid=4326), srid=4326),
        ),
    ]
