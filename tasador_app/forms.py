from django import forms
from . import models
from mapwidgets.widgets import GooglePointFieldWidget


class createDepartamento(forms.ModelForm):
    class Meta:
        model = models.Departamento
        fields = ["distrito", "area", "dormitorios", "baños", "ubicacion"]

        widgets = {
            'ubicacion': GooglePointFieldWidget,
        }

class createCasa(forms.ModelForm):
    class Meta:
        model = models.Casa
        fields = ["distrito", "area", "dormitorios", "baños", "ubicacion"]

        widgets = {
            'ubicacion': GooglePointFieldWidget,
        }


class createTerreno(forms.ModelForm):
    class Meta:
        model = models.Terreno
        fields = ["distrito", "area", "ubicacion"]

        widgets = {
            'ubicacion': GooglePointFieldWidget,
        }
