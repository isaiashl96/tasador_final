import numpy as np
import time
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error


def interpolationLearner(instance, tipo):
    path = 'C:/Users/Isaias HL/Desktop/DjangoProjects/tasador_final/tasador_app/scripts/'
    estimators = []

    # model1
    lasso = joblib.load(path+'{}_model_cache/int_lasso_cache.pkl'.format(tipo))
    estimators.append(lasso)  # https://www.kaggle.com/apapiu/regularized-linear-models

    # model2
    enet = joblib.load(path+'{}_model_cache/int_elastic_cache.pkl'.format(tipo))
    estimators.append(enet)

    # model3
    krr = joblib.load(path+'{}_model_cache/int_kernel_cache.pkl'.format(tipo))
    estimators.append(krr)

    # model4
    gboost = joblib.load(path+'{}_model_cache/int_gboost_cache.pkl'.format(tipo))
    estimators.append(gboost)

    predict = np.column_stack([estimator.predict(instance) for estimator in estimators])
    mean_predict = np.column_stack([i.mean() for i in predict])

    return mean_predict[0]
