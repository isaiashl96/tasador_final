import time
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from tasador_app.scripts.regression import make_regression, make_regression2
from . import forms
from .models import Departamento, Casa, Terreno

def home(request):
    return render(request, 'tasador_app/home.html')

#def tasador_list(request):
#    lista_inmuebles = Departamento.objects.all().order_by('area')
#    return render(request, 'tasador_app/tasador_list.html', {'lista_inmuebles': lista_inmuebles})

def tasador_result_departamento(request):
    dict_distritos = {'Ancon':0, 'Ate':0, 'Barranco':0, 'Bellavista':0,'Breña':0, 'Callao':0, 'Carabayllo':0,
                      'Cercado de Lima':0, 'Chaclacayo':0, 'Chorrillos':0, 'Chosica':0, 'Comas':0, 'El Agustino':0,
                      'Jesus Maria':0, 'La Molina':0, 'La Perla':0, 'La Punta':0, 'La Victoria':0, 'Lince':0,
                      'Los Olivos':0, 'Lurin':0, 'Magdalena del Mar':0, 'Miraflores':0, 'Pachacamac': 0, 'Pueblo Libre':0,
                      'Puente Piedra':0, 'Punta Hermosa':0, 'Rimac':0, 'San Bartolo':0, 'San Borja':0, 'San Isidro':0,
                      'San Juan de Lurigancho':0, 'San Juan de Miraflores':0, 'San Luis':0, 'San Martin de Porres':0,
                      'San Miguel':0, 'Santa Anita':0, 'Santa Maria del Mar':0, 'Santa Rosa':0,'Santiago de Surco':0,
                      'Surquillo':0, 'Ventanilla':0, 'Villa El Salvador':0}

    #print(dict_distritos)
    inmueble = Departamento.objects.last()

    dict_distritos[inmueble.distrito] = 1
    valores_distritos = list(dict_distritos.values())

    lat, long = inmueble.ubicacion.y, inmueble.ubicacion.x

    inicio = time.time()
    precio_reg = make_regression('departamentos',inmueble.area, inmueble.dormitorios, inmueble.baños, valores_distritos, lat, long)
    print(inicio-time.time())

    precio_reg_soles = round(precio_reg * 3.46,2)
    return render(request, 'tasador_app/tasador_result_depa.html', {'inmueble': inmueble, 'precio_reg': precio_reg,
                                                               'precio_reg_soles':precio_reg_soles})

def tasador_result_casa(request):
    dict_distritos = {'Ancon':0, 'Ate':0, 'Barranco':0, 'Bellavista':0,'Breña':0, 'Callao':0, 'Carabayllo':0,
                      'Cercado de Lima':0, 'Chaclacayo':0, 'Chorrillos':0, 'Chosica':0, 'Cieneguilla':0,'Comas':0,
                      'El Agustino':0, 'Independencia':0, 'Jesus Maria':0, 'La Molina':0, 'La Perla':0, 'La Punta':0,
                      'La Victoria':0, 'Lince':0, 'Los Olivos':0, 'Lurin':0, 'Magdalena del Mar':0, 'Miraflores':0,
                      'Pachacamac': 0, 'Pucusana':0, 'Pueblo Libre':0, 'Puente Piedra':0, 'Punta Hermosa':0, 'Punta Negra':0,'Rimac':0,
                      'San Bartolo':0, 'San Borja':0, 'San Isidro':0, 'San Juan de Lurigancho':0,
                      'San Juan de Miraflores':0, 'San Luis':0, 'San Martin de Porres':0, 'San Miguel':0,
                      'Santa Anita':0, 'Santa Maria del Mar':0, 'Santa Rosa':0,'Santiago de Surco':0, 'Surquillo':0,
                      'Ventanilla':0, 'Villa El Salvador':0, 'Villa Maria del Triunfo':0}

    inmueble = Casa.objects.last()

    dict_distritos[inmueble.distrito] = 1
    valores_distritos = list(dict_distritos.values())

    lat, long = inmueble.ubicacion.y, inmueble.ubicacion.x

    inicio = time.time()
    precio_reg = make_regression('casas', inmueble.area, inmueble.dormitorios, inmueble.baños, valores_distritos, lat, long)
    print(inicio-time.time())

    precio_reg_soles = round(precio_reg * 3.46,2)
    return render(request, 'tasador_app/tasador_result_casa.html', {'inmueble': inmueble, 'precio_reg': precio_reg,
                                                               'precio_reg_soles':precio_reg_soles})


def tasador_result_terreno(request):
    dict_distritos = {'Ancon': 0, 'Ate': 0, 'Barranco': 0, 'Bellavista': 0, 'Breña': 0, 'Callao': 0, 'Carabayllo': 0,
                      'Carmen De La Legua Reynoso': 0,
                      'Cercado de Lima': 0, 'Chaclacayo': 0, 'Chorrillos': 0, 'Chosica': 0, 'Cieneguilla': 0,
                      'Comas': 0,
                      'El Agustino': 0, 'Independencia': 0, 'Jesus Maria': 0, 'La Molina': 0, 'La Perla': 0,
                      'La Victoria': 0, 'Lince': 0, 'Los Olivos': 0, 'Lurin': 0, 'Magdalena del Mar': 0,
                      'Miraflores': 0,
                      'Pachacamac': 0, 'Pucusana': 0, 'Pueblo Libre': 0, 'Puente Piedra': 0, 'Punta Hermosa': 0,
                      'Punta Negra': 0, 'Rimac': 0,
                      'San Bartolo': 0, 'San Borja': 0, 'San Isidro': 0, 'San Juan de Lurigancho': 0,
                      'San Juan de Miraflores': 0, 'San Luis': 0, 'San Martin de Porres': 0, 'San Miguel': 0,
                      'Santa Anita': 0, 'Santa Maria del Mar': 0, 'Santa Rosa': 0, 'Santiago de Surco': 0,
                      'Surquillo': 0,
                      'Ventanilla': 0, 'Villa El Salvador': 0, 'Villa Maria del Triunfo': 0}

    inmueble = Terreno.objects.last()

    dict_distritos[inmueble.distrito] = 1
    valores_distritos = list(dict_distritos.values())

    lat, long = inmueble.ubicacion.y, inmueble.ubicacion.x

    precio_reg = make_regression2('terrenos', inmueble.area, valores_distritos, lat, long)
    precio_reg_soles = round(precio_reg * 3.46,2)
    return render(request, 'tasador_app/tasador_result_terreno.html', {'inmueble': inmueble, 'precio_reg': precio_reg,
                                                                       'precio_reg_soles': precio_reg_soles})

@login_required(login_url="/accounts/login/")
def crear_departamento(request):
    if request.method == 'POST':
        form = forms.createDepartamento(request.POST)
        if form.is_valid():
            instace = form.save(commit=False)
            instace.empresario = request.user
            instace.save()
            return redirect('tasador_app:result_depa')
    else:
        form = forms.createDepartamento()
    return render(request, 'tasador_app/build_create_departamento.html', {'form':form})

@login_required(login_url="/accounts/login/")
def crear_casa(request):
    if request.method == 'POST':
        form = forms.createCasa(request.POST)
        if form.is_valid():
            instace = form.save(commit=False)
            instace.empresario = request.user
            instace.save()
            return redirect('tasador_app:result_casa')
    else:
        form = forms.createCasa()
    return render(request, 'tasador_app/build_create_casa.html', {'form':form})

@login_required(login_url="/accounts/login/")
def crear_terreno(request):
    if request.method == 'POST':
        form = forms.createTerreno(request.POST)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.empresario = request.user
            instance.save()
            return redirect('tasador_app:result_terreno')
    else:
        form = forms.createTerreno()
    return render(request, 'tasador_app/build_create_terreno.html', {'form':form})
