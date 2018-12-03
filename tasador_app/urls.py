from django.urls import path
from . import views

app_name = 'tasador_app'

urlpatterns = [
    path('', views.home, name='home'),
    path('create_departamento/', views.crear_departamento, name='crear_depa'),
    path('create_casa/', views.crear_casa, name='crear_casa'),
    path('create_terreno/', views.crear_terreno, name='crear_terreno'),
    path('result_departamento/', views.tasador_result_departamento, name='result_depa'),
    path('result_casa/', views.tasador_result_casa, name='result_casa'),
    path('result_terreno/', views.tasador_result_terreno, name='result_terreno')
    ]