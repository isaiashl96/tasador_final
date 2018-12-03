import os
import time

import numpy as np
from django.core.cache import cache
from googleplaces import GooglePlaces, types
from haversine import haversine
from sklearn.externals import joblib
from .interpolation_learner import interpolationLearner

def make_regression2(tipo, area, valores_distritos, lat, long):
    path = 'C:/Users/Isaias HL/Desktop/DjangoProjects/tasador_final/tasador_app/scripts/'

    model_cache_key = 'model_cache'
    model_rel_path = path + '{}_model_cache/ada_cache.pkl'.format(tipo)

    model = cache.get(model_cache_key)

    if model is None:
        model = joblib.load(model_rel_path)
        #print(model.score)
        #print(model.estimators_)
        #print(model.estimator_weights_)
        #print(model.feature_importances_)
        #cache.set(model_cache_key, model, None)

    place_types = [types.TYPE_SCHOOL, types.TYPE_HOSPITAL, types.TYPE_SHOPPING_MALL, types.TYPE_RESTAURANT,
                   types.TYPE_GROCERY_OR_SUPERMARKET, types.TYPE_BANK]

    geo_variables_tmp = list()
    for place in place_types:
        geo_variables_tmp.append(search_places(lat, long, place))

    geo_variables = [item for sublist in geo_variables_tmp for item in sublist]
    print(len(geo_variables))

    data_tmp = np.array([[area]])
    data_tmp = np.append(data_tmp, geo_variables).reshape(1, -1)
    print(len(data_tmp[0]))

    scaled_data_tmp = np.log1p(data_tmp)
    scaled_data = np.append(scaled_data_tmp, valores_distritos).reshape(1, -1)
    print(len(scaled_data[0]))


    interpolation = interpolationLearner(scaled_data, tipo)

    enet = joblib.load(path+'{}_model_cache/enet.pkl'.format(tipo)).predict(scaled_data)

    krr = joblib.load(path+'{}_model_cache/krr.pkl'.format(tipo)).predict(scaled_data)

    gbr = joblib.load(path+'{}_model_cache/gbr.pkl'.format(tipo)).predict(scaled_data)

    final_data = np.append(scaled_data, [interpolation, enet, krr, gbr]).reshape(1, -1)
    print(len(final_data[0]))

    prediction = model.predict(final_data)
    prediction_final = np.expm1(prediction)

    return round(prediction_final[0], 2)


def make_regression(tipo, area, cuartos, aseos, valores_distritos, lat, long):
    path = 'C:/Users/Isaias HL/Desktop/DjangoProjects/tasador_final/tasador_app/scripts/'
    if tipo=='departamentos':

        model_cache_key = 'model_cache'
        model_rel_path = path + '{}_model_cache/ada_cache.pkl'.format(tipo)
    else:
        model_cache_key = 'model_cache'
        model_rel_path = path + '{}_model_cache/ada_cache.pkl'.format(tipo)

    model = cache.get(model_cache_key)

    if model is None:
        #model_path = os.path.realpath(model_rel_path)
        model = joblib.load(model_rel_path)
        print(model.best_estimator_.score)
        #print(model.estimators_)
        #print(model.estimator_weights_)
        print(model.best_estimator_.feature_importances_)

        #cache.set(model_cache_key, model, None)

    place_types = [types.TYPE_SCHOOL, types.TYPE_HOSPITAL, types.TYPE_SHOPPING_MALL, types.TYPE_RESTAURANT,
                   types.TYPE_GROCERY_OR_SUPERMARKET, types.TYPE_BANK]

    geo_variables_tmp = list()
    for place in place_types:
        geo_variables_tmp.append(search_places(lat, long, place))

    geo_variables = [item for sublist in geo_variables_tmp for item in sublist]
    print(geo_variables)

    data_tmp = np.array([[area, cuartos, aseos]])
    data_tmp = np.append(data_tmp, geo_variables).reshape(1,-1)
    print(len(data_tmp[0]))

    scaled_data_tmp = np.log1p(data_tmp)
    scaled_data = np.append(scaled_data_tmp, valores_distritos).reshape(1,-1)
    print(len(scaled_data[0]))

    if tipo=='departamentos':
        interpolation = interpolationLearner(scaled_data, tipo)

        enet = joblib.load(path+'{}_model_cache/enet.pkl'.format(tipo)).predict(scaled_data)

        krr = joblib.load(path+'{}_model_cache/krr.pkl'.format(tipo)).predict(scaled_data)

        gbr = joblib.load(path+'{}_model_cache/gbr.pkl'.format(tipo)).predict(scaled_data)
    else:
        interpolation = interpolationLearner(scaled_data, tipo)

        enet = joblib.load(path + '{}_model_cache/enet.pkl'.format(tipo)).predict(scaled_data)

        krr = joblib.load(path + '{}_model_cache/krr.pkl'.format(tipo)).predict(scaled_data)

        gbr = joblib.load(path + '{}_model_cache/gbr.pkl'.format(tipo)).predict(scaled_data)

    final_data = np.append(scaled_data, [interpolation, enet, krr, gbr]).reshape(1,-1)
    print(len(final_data[0]))

    prediction = model.predict(final_data)
    prediction_final = np.expm1(prediction)

    return round(prediction_final[0],2)


def search_places(lat, lng, place_type):
    API_KEY = 'AIzaSyAUlp6OXdNnSlpO56yVru2gt3_6FbW8QS4'
    google_places = GooglePlaces(API_KEY)

    active_latlng = (lat, lng)
    query_result = google_places.nearby_search(lat_lng={'lat': lat, 'lng': lng}, rankby='distance', types=[place_type])

    time.sleep(1.8)

    if query_result.has_next_page_token:
        query_result2 = google_places.nearby_search(lat_lng={'lat': lat, 'lng': lng}, rankby='distance',
                                                    types=[place_type], pagetoken=query_result.next_page_token)

        lista_places = query_result.places + query_result2.places
        time.sleep(1.8)

        if query_result2.has_next_page_token:
            query_result3 = google_places.nearby_search(lat_lng={'lat': lat, 'lng': lng}, rankby='distance',
                                                        types=[place_type], pagetoken=query_result2.next_page_token)

            lista_places = lista_places + query_result3.places
            resultado = compute_results(lista_places, active_latlng)
        else:
            resultado = compute_results(lista_places, active_latlng)

    else:
        lista_places = query_result.places
        resultado = compute_results(lista_places, active_latlng)

    return resultado


def compute_results(lista_places, latlng):
    n_places = 0
    distance_list = list()
    for place in lista_places:
        place_latlng = (place.geo_location['lat'], place.geo_location['lng'])
        distance = haversine(latlng, place_latlng)
        distance_list.append(distance)

        if distance < 0.7:
            n_places = n_places + 1

    return [n_places, min(distance_list)]
