import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, Product, Sum
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso, ElasticNet
from sklearn.externals import joblib
from scipy.stats import skew
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import xgboost as xgb

df = pd.read_csv('data_prueba.csv', sep=';', encoding='latin-1')
df = df.dropna(how='any')

df = df.set_index('n_inmueble')
distrito = df.distrito
df = df.loc[:, df.columns != 'distrito']

data = df.loc[:, df.columns != 'precio']
target = df.precio

# Normaliza las variables con asimetria
# #numeric_feats = data.dtypes[data.dtypes != "object"].index
# skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())) # compute skewness
# skewed_feats = skewed_feats[skewed_feats > 0.75]
# skewed_feats = skewed_feats.index

#data[skewed_feats] = np.log1p(data[skewed_feats])
data = np.log1p(data)
data['distrito'] = distrito
data = pd.get_dummies(data)

#print(data)
scaled_target = np.log1p(target)

data_train, data_test, target_train, target_test = train_test_split(data, scaled_target, test_size=0.2)

numColXTraino = len(data_train.columns)
numColXTesto = len(data_test.columns)

def interpolationLearner(xtrain, ytrain, xtest):

  estimators = []

  #model1
  param_grid = {"alpha": [5e-05], "tol" : [0.0001,0.00001]}
  model1 = GridSearchCV(Lasso(), cv=4, param_grid=param_grid, scoring = "neg_mean_squared_error")
  estimators.append(('Lasso', model1)) # https://www.kaggle.com/apapiu/regularized-linear-models

  #model2
  param_grid = {"alpha": [5e-05,0.0005], "l1_ratio":[0.9,0.8], "random_state":[3,4]}
  model2 = GridSearchCV(ElasticNet(), cv = 4, param_grid=param_grid, scoring = "neg_mean_squared_error") # http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html#sphx-glr-auto-examples-gaussian-process-plot-compare-gpr-krr-py
  estimators.append(('ENet', model2))

  #model3
  param_grid = {"alpha": [0.6,0.5], "kernel": [ConstantKernel(1.0, (1e-3, 1e3)) + RBF()]}
  model3 = GridSearchCV(KernelRidge(), cv = 4, param_grid=param_grid, scoring = "neg_mean_squared_error") # http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html#sphx-glr-auto-examples-gaussian-process-plot-compare-gpr-krr-py
  estimators.append(('KRR', model3))

  #model4
  param_grid = {"loss": ["huber"], "max_features": ["sqrt"], "max_depth":[4,5,6]}
  model4 = GridSearchCV(GradientBoostingRegressor(learning_rate=0.05, max_depth=3, n_estimators=2200), cv = 4, param_grid=param_grid, scoring = "neg_mean_squared_error") # http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html#sphx-glr-auto-examples-gaussian-process-plot-compare-gpr-krr-py
  estimators.append(('GBoost', model4))

  learners = []
  # Hagamos fit
  for learner in np.arange(len(estimators)):
    stime = time.time()
    learners.append(estimators[learner][1].fit(xtrain,ytrain))
    print("For " + str(estimators[learner][0]).upper())
    print("")
    print("Time for learner "+ str(estimators[learner][0]) +" "+  str(time.time() - stime))
    print("El learner " + str(estimators[learner][0]) + " tiene un score de " + str(learners[learner].best_score_))
    print("Los parametros para el learner "+ str(estimators[learner][0]) +" fueron "+  str(learners[learner].best_params_))
    print("")

  joblib.dump(model1, 'model_cache/int_lasso_cache.pkl')
  joblib.dump(model2, 'model_cache/int_elastic_cache.pkl')
  joblib.dump(model3, 'model_cache/int_kernel_cache.pkl')
  joblib.dump(model4, 'model_cache/int_gboost_cache.pkl')


  # CV score para la media de los learners
  predictTraino = np.column_stack([learner.predict(xtrain) for learner in learners])
  mean_predictTraino = np.column_stack([np.mean(i) for i in predictTraino])
  print("")
  print("El score final de toda la interpolación en traino es " + str(mean_squared_error(ytrain,mean_predictTraino[0])))
  print("")
  predict = np.column_stack([learner.predict(xtest) for learner in learners])
  mean_predict = np.column_stack([i.mean() for i in predict])
  return mean_predict[0]


def ElasticNetLearner(xtrain, ytrain, xtest):
  param_grid = {"alpha": [5e-05,0.0005], "l1_ratio":[0.9,0.8], "random_state":[3,4]}
  model = GridSearchCV(ElasticNet(), cv = 4, param_grid=param_grid, scoring = "neg_mean_squared_error")

  stime = time.time()
  model.fit(xtrain,ytrain)
  joblib.dump(model, 'model_cache/enet.pkl')
  print("")
  print("Time for ENET "+  str(time.time() - stime))
  print("Score of ENET " + str(model.best_score_))
  print("Parametros of ENET " + str(model.best_params_))
  print("")
  predict = model.predict(xtest)
  return predict


def KRRLearner(xtrain, ytrain, xtest):
  param_grid = {"alpha": [0.6,0.5], "kernel": [ConstantKernel(1.0, (1e-3, 1e3)) + RBF()]}
  model = GridSearchCV(KernelRidge(), cv = 4, param_grid=param_grid, scoring = "neg_mean_squared_error")
  stime = time.time()
  model.fit(xtrain,ytrain)
  joblib.dump(model, 'model_cache/krr.pkl')
  print("")
  print("Time for KRR "+  str(time.time() - stime))
  print("Score of KRR " + str(model.best_score_))
  print("Parametros of KRR " + str(model.best_params_))
  print("")
  predict = model.predict(xtest)
  return predict


def GboostRegressorLearner(xtrain, ytrain, xtest):
  param_grid = {"loss": ["huber"], "max_features": ["sqrt"], "max_depth":[4,5]}
  model = GridSearchCV(GradientBoostingRegressor(learning_rate=0.05, max_depth=3, n_estimators=2200), cv = 4, param_grid=param_grid, scoring = "neg_mean_squared_error") # http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html#sphx-glr-auto-examples-gaussian-process-plot-compare-gpr-krr-py
  stime = time.time()
  model.fit(xtrain,ytrain)
  joblib.dump(model, 'model_cache/gbr.pkl')
  print("")
  print("Time for GBR "+  str(time.time() - stime))
  print("Score of GBR " + str(model.best_score_))
  print("Parametros of GBR " + str(model.best_params_))
  print("")
  predict = model.predict(xtest)
  return predict


def atributosBaseLearner(xtrain, ytrain, xtest, modelo = "interpolation"):
  stime1 = time.time()
  xtrain1 = xtrain[:]
  xtest1 = xtest[:]

  print(np.any(np.isnan(ytrain)))

  nombreColumna = str("New") # nombreColumna = str(str(modelo)[0:str(modelo).find("(")])
  xtrain[nombreColumna] = 0 # Atributo nuevo para el testo
  xtest[nombreColumna] = 0 # Atributo nuevo para el traino
  kf = KFold(n_splits=4) #Split data
  a = 1
  if modelo == "interpolation":
    for i,j in kf.split(ytrain):
      print("Fold Numero" + str(a))
      print("")
      x_fold_train, y_fold_train = xtrain1.iloc[i], ytrain.iloc[i]

      x_fold_test, y_fold_test = xtrain1.iloc[j], ytrain.iloc[j]
      prediction_fold = interpolationLearner(x_fold_train, y_fold_train, x_fold_test) # Entrena el modelo con las folds de traino y predice en el fold de testo
      xtrain[nombreColumna].iloc[j] = prediction_fold
      a=a+1
    print("------ TEST DEL ATRIBUTO DE INTERPOLACION TERMINADO -------")
    print("")
    print("--------- Se comienza con la prediccion para el atributo de Testing ----------")
    print("")
    resultsForTest = interpolationLearner(xtrain1,ytrain,xtest1) # Ahora entrena pero solo con la data original en todo el traino y predice en el testo
    print("---------------------------------  Se tienen los resultadooos !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ----------------------------------------- ")
    xtest[nombreColumna] = resultsForTest
    print("Todo ha demorado" + str(time.time() - stime1))
    return xtrain[str(nombreColumna)], xtest[str(nombreColumna)]
  if modelo == "ENET":
    for i,j in kf.split(ytrain):
      x_fold_train, y_fold_train = xtrain1.iloc[i], ytrain.iloc[i]
      x_fold_test, y_fold_test = xtrain1.iloc[j], ytrain.iloc[j]
      prediction_fold = ElasticNetLearner(x_fold_train, y_fold_train, x_fold_test) # Entrena el modelo con las folds de traino y predice en el fold de testo
      xtrain[nombreColumna].iloc[j] = prediction_fold
    print("TEST DEL ATRIBUTO DE ENET TERMINADO")
    print("")
    print("Se comienza con la prediccion para el atributo de Testing")
    resultsForTest = ElasticNetLearner(xtrain1,ytrain,xtest1) # Ahora entrena pero solo con la data original en todo el traino y predice en el testo
    xtest[nombreColumna] = resultsForTest
    print("Todo ha demorado" + str(time.time() - stime1))
    return xtrain[str(nombreColumna)], xtest[str(nombreColumna)]
  if modelo == "KRR":
    for i,j in kf.split(ytrain):
      x_fold_train, y_fold_train = xtrain1.iloc[i], ytrain.iloc[i]
      x_fold_test, y_fold_test = xtrain1.iloc[j], ytrain.iloc[j]
      prediction_fold = KRRLearner(x_fold_train, y_fold_train, x_fold_test) # Entrena el modelo con las folds de traino y predice en el fold de testo
      xtrain[nombreColumna].iloc[j] = prediction_fold
    print("TEST DEL ATRIBUTO DE KRR TERMINADO")
    print("")
    print("Se comienza con la prediccion para el atributo de Testing")
    resultsForTest = KRRLearner(xtrain1,ytrain,xtest1) # Ahora entrena pero solo con la data original en todo el traino y predice en el testo
    xtest[nombreColumna] = resultsForTest
    print("Todo ha demorado" + str(time.time() - stime1))
    return xtrain[str(nombreColumna)], xtest[str(nombreColumna)]
  if modelo == "GBR":
    for i,j in kf.split(ytrain):
      x_fold_train, y_fold_train = xtrain1.iloc[i], ytrain.iloc[i]
      x_fold_test, y_fold_test = xtrain1.iloc[j], ytrain.iloc[j]
      prediction_fold = GboostRegressorLearner(x_fold_train, y_fold_train, x_fold_test) # Entrena el modelo con las folds de traino y predice en el fold de testo
      xtrain[nombreColumna].iloc[j] = prediction_fold
    print("TEST DEL ATRIBUTO DE GBR TERMINADO")
    print("")
    print("Se comienza con la prediccion para el atributo de Testing")
    resultsForTest = GboostRegressorLearner(xtrain1,ytrain,xtest1) # Ahora entrena pero solo con la data original en todo el traino y predice en el testo
    xtest[nombreColumna] = resultsForTest
    print("Todo ha demorado" + str(time.time() - stime1))
    return xtrain[str(nombreColumna)], xtest[str(nombreColumna)]


interpolationTrain,interpolationTest = atributosBaseLearner(data_train.iloc[:,:numColXTraino],target_train,
                                                            data_test.iloc[:,:numColXTesto],"interpolation") # Para el atributo de las interpolaciones
ENETTrain,ENETTest = atributosBaseLearner(data_train.iloc[:,:numColXTraino], target_train,
                                          data_test.iloc[:,:numColXTesto],"ENET") # Para el atributo de las tree regression bagging
KRRTrain,KRRTest = atributosBaseLearner(data_train.iloc[:,:numColXTraino], target_train,
                                        data_test.iloc[:,:numColXTesto],"KRR") # Para el atributo de las tree regression bagging
GBRTrain,GBRTest = atributosBaseLearner(data_train.iloc[:,:numColXTraino], target_train,
                                        data_test.iloc[:,:numColXTesto],"GBR") # Para el atributo de las tree regression bagging

data_train["interpolation"] = interpolationTrain
data_test["interpolation"] = interpolationTest
data_train["ENET"] = ENETTrain
data_test["ENET"] = ENETTest
data_train["KRR"] = KRRTrain
data_test["KRR"] = KRRTest
data_train["GBR"] = GBRTrain
data_test["GBR"] = GBRTest

mixgb=xgb.XGBRegressor()
params2= {"max_depth" : [6,7,8,9,10], "learning_rate" :[0.09,0.08,0.07,0.06,0.1], "n_estimators" : [80,90,100,120,150,200], 'subsample': [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85],'min_child_weight': [1,2,3,4,5,6,7,10],'gamma': [0.5, 0.7, 0.6, 0.8,1.5,4,5],'colsample_bytree': [0.7,0.8, 0.9,1.0]}
def pruebaXGB(params2):
  random_search = RandomizedSearchCV(mixgb, param_distributions=params2, n_iter=500, scoring='neg_mean_squared_error', n_jobs=10, cv=5, random_state=1001, verbose=3)
  random_search.fit(data_train, target_train)
  # print("")
  # print(random_search.best_params_)
  # print("")
  # print(random_search.best_score_)
  # print("")
  # print(mean_squared_error(target_train,random_search.predict(data_train)))
  return(random_search)


XGB = pruebaXGB(params2)
mse1 = mean_squared_error(target_test,XGB.predict(data_test))
best_params_1 = XGB.best_params_
print("El score de XGB es " + str(mean_squared_error(target_test,XGB.predict(data_test)))) # 0.1172

joblib.dump(XGB, 'model_cache/xgb_cache.pkl')

params= {"max_depth" : [3,4,5,6,7,8,9], "num_leaves":[64,128,256,300,500,550,600], "max_bin " :[80,70,60,50,40,30,20], "learning_rate" :[0.1,0.09,0.009,0.00001], "n_estimators":[600,700,800]}
def pruebaLGBM(params):
  random_search = RandomizedSearchCV(lgb.LGBMRegressor(objective='regression', bagging_fraction = 0.8),
    param_distributions=params, n_iter=500, scoring='neg_mean_squared_error', n_jobs=10, cv=5, random_state=1001, verbose=3)
  random_search.fit(data_train, target_train)
  # print("")
  # print(random_search.best_params_)
  # print("")
  # print(random_search.best_score_)
  # print("")
  # print(mean_squared_error(target_train,random_search.predict(data_train)))
  return(random_search)


LGBM = pruebaLGBM(params)
mse2 = mean_squared_error(target_test,LGBM.predict(data_test))
best_params_2 = LGBM.best_params_
print("El score de LGBM es " + str(mean_squared_error(target_test,LGBM.predict(data_test)))) # 0.1135

joblib.dump(LGBM, 'model_cache/lgbm_cache.pkl')

param_grid = {"base_estimator__splitter" : ["random"],"base_estimator__max_depth" : range(2,16),"learning_rate" : [0.009,0.0009,0.00009],"n_estimators": [10,20,50,80,100]}
DTC = DecisionTreeRegressor()
ABC = AdaBoostRegressor(base_estimator = DTC)
def pruebaAdaBoost(param_grid):
  random_search =  RandomizedSearchCV(ABC, param_distributions=param_grid, n_iter=100, scoring='neg_mean_squared_error', n_jobs=10, cv=5, random_state=1001, verbose=3)
  random_search.fit(data_train, target_train)
  # print("")
  # print(random_search.best_params_)
  # print("")
  # print(random_search.best_score_)
  # print("")
  # print(mean_squared_error(target_train,random_search.predict(data_train)))
  return(random_search)


Ada = pruebaAdaBoost(param_grid)
mse3 = mean_squared_error(target_test, Ada.predict(data_test))
best_params_3 = Ada.best_params_
print("El score de AdaBoost es " + str(mean_squared_error(target_test, Ada.predict(data_test)))) # 0.1081

joblib.dump(Ada, 'model_cache/ada_cache.pkl')

lista_resultados = [
    mean_squared_error(target_test,XGB.predict(data_test)),
    mean_squared_error(target_test,LGBM.predict(data_test)),
    mean_squared_error(target_test, Ada.predict(data_test))
]

print(lista_resultados)