from django.contrib import admin
from django.contrib.gis.db import models
from .models import Departamento, Casa, Terreno
#from leaflet.admin import LeafletGeoAdmin
from mapwidgets.widgets import GooglePointFieldWidget

class InmuebleAdmin(admin.ModelAdmin):
    list_display = ('area', 'ubicacion')
    formfield_overrides = {
        models.PointField: {"widget": GooglePointFieldWidget}
    }

admin.site.register(Departamento, InmuebleAdmin)
admin.site.register(Casa)
admin.site.register(Terreno)