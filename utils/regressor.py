
import math
from pyexpat import model
from tabnanny import verbose
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,ridge_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit,RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error



'''
- la fonction transform_data modifie les données en entrée tel que les données sont toujours dans un DF 
En deux dimensions. Ne plus tester la taille à partir de ce module
Voir TODO

TODO: Ajouter la regression logsitique: ok 
TODO: gridSearch pour SVR : ok 
TODO: gridSearch pour xgboost ok, 

TODO: Optimisation des hyperparametre avec une autre stratégie : random search 
TODO: possibilité d'écrire 
TODO: HPO par random search : e

TODO: comparaison de librairies


'''


class Regressor:
    def __init__(self,output_directory):
        self.model = None 
        self.params = None
        self.output_directory = output_directory
        self.model = None
        
    def fit(self,x_train,y_train):

        if len(x_train.shape) == 3:
            # TODO : virez la suivante et tester
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        
        print(x_train.shape)
        self.cross_validation(x_train,y_train)
        self.model.fit(x_train, y_train)
        print("fitting completed")
        
    def cross_validation(self,x_train,y_train):
        pass
    
  
class LinearRegressor(Regressor):
    def __init__(self, output_directory,model_params,type = "linear_regression"):
        super().__init__(output_directory)
        self.params = model_params
        self.build_model(**model_params)

    def build_model(self,**model_params):
        self.model = LinearRegression(**model_params)
        return self.model
        
    def predict(self,x_test: np.array):
        if len(x_test.shape) == 3:
            # TODO : virez la suivante et tester
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
        return self.model.predict(x_test)
    
    def cross_validation(self,x_train, y_train,strategie_ohp = "gridsearchcv"):
        best_param = None 
        best_param_score = None
        
        if strategie_ohp == "gridsearchcv":
            print("Stratégie gridsearch")
            # params = self.params
            search_params = {
                    "fit_intercept": [True,False]
                }
            search = GridSearchCV(LinearRegression(),
                                param_grid=search_params,
                                verbose=True,
                                cv=TimeSeriesSplit(n_splits=5),
                                scoring="neg_mean_absolute_percentage_error"
                                )
            search.fit(x_train,y_train)
            best_param = search.best_params_
            best_param_score = search.best_score_
            
        elif strategie_ohp == "randomsearchcv":
            print("Stratégie randomsearch")
            # search_params = {"kernel": ["rbf", "sigmoid", "linear"],
            #                 "gamma": loguniform(0.001, 1),
            #                 "C": loguniform(0.1, 100)}
            
            pass
            

        if best_param is not None:
            # On a trouvé un meilleur paramètrage pour l'
            print("Best Param: {}, with scores: {}".format(best_param, best_param_score))
            self.params.update(best_param)
            self.build_model(**self.best_param)


"""============================ Random forest regressor ============================ """
class RFRegressor(Regressor):
    def __init__(self, output_directory,model_params,type = "random_forest"):
        super().__init__(output_directory)
        self.params = model_params
        self.build_model(**model_params)
        
    def build_model(self,**model_params):
        self.model = RandomForestRegressor(**model_params)
        return self.model
        
        
    def cross_validation(self,x_train, y_train):
        print("cross validation")
        # params = self.params
        best_param = None 
        best_param_score = None
        search_params = {
                "n_estimators": [100],
                "max_depth": [5, 10],
                "min_samples_leaf": [10, 15]
            }
        search = GridSearchCV(RandomForestRegressor(verbose=True),
                              param_grid=search_params,
                              verbose=True,
                              cv=TimeSeriesSplit(n_splits=2),
                              scoring="neg_mean_absolute_percentage_error"
                              )
        search.fit(x_train,y_train)
        best_param = search.best_params_
        best_param_score = search.best_score_

        if best_param is not None:
            print("Best Param: {}, with scores: {}".format(best_param, best_param_score))
            self.params["n_jobs"] = -1
            self.params.update(best_param)
            self.build_model(**self.params)

    def predict(self,x_test: np.array):
        if len(x_test.shape) == 3:
            # TODO : virez la suivante et tester
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
        return self.model.predict(x_test)
    

""" ============================ SVM regressor ============================"""
class SVRegressor(Regressor):
    def __init__(self, output_directory,model_params,type="SVR"):
        super().__init__(output_directory)
        self.params = model_params
        self.build_model(**model_params)
        
    def build_model(self,**model_params):
        self.model = SVR(**model_params)
        return self.model

    def cross_validation_score(self,X,Y):
        cv_results = cross_val_score(self.model, X, Y, cv=5, scoring="mean_absolute_percentage_error")
    
    def cross_validation(self, x_train, y_train):
        print("cross validation pour SVR :")
        # params = self.params
        best_param = None 
        best_param_score = None
        search_params = {
                "kernel": ["poly","rbf"],
                "gamma":["scale"]
                
            }
        search = GridSearchCV(SVR(verbose=True),
                              param_grid=search_params,
                              verbose=True,
                              cv=TimeSeriesSplit(n_splits=2),
                              scoring="neg_mean_absolute_percentage_error"
                              )
        search.fit(x_train,y_train)
        best_param = search.best_params_
        best_param_score = search.best_score_
        
        if best_param is not None:
            print("Best Param: {}, with scores: {}".format(best_param, best_param_score))
        self.params.update(best_param)
        self.build_model(**self.params)
            
    
    
    def predict(self,x_test: np.array):
        if len(x_test.shape) == 3:
            # TODO : virez la suivante et tester
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
        return self.model.predict(x_test)
    
""" ============================ XGB regressor ============================"""
class XGBReg(Regressor):
    '''
    '''
    def __init__(self, output_directory,model_params, type="XGBR"):
        super().__init__(output_directory)
        self.params = model_params
        self.build_model(**model_params)
        
    def build_model(self,**model_params):
        self.model = XGBRegressor(**model_params)
        return self.model
    
    def predict(self,x_test: np.array):
        if len(x_test.shape) == 3:
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
        return self.model.predict(x_test)
    
    
    def cross_validation(self, x_train, y_train):
        
        params = self.params
        best_param = None 
        best_param_score = None 
        
        search_params = {
                "n_estimators": [100, 500, 1000],
                "max_depth": [5],
                "learning_rate": [0.1]
            }
        
        search = GridSearchCV(XGBRegressor(**params),
                              search_params,
                              n_jobs=1,
                              cv=3,
                              scoring="neg_mean_squared_error", verbose=True)
        search.fit(x_train, y_train)
        best_param = search.best_params_
        best_param_score = search.best_score_
    
        if best_param is not None:
            print("Best Param: {}, with scores: {}".format(best_param, best_param_score))
            self.params.update(best_param)
            self.params["n_jobs"] = 0
            self.build_model(**self.params)


'''
 ============================ Constructeurs============================ : 
'''
 
def create_regressor(regressor_name:str,output_directory):
    if regressor_name == "linear_regression" :
        kwargs = {"fit_intercept": True,"n_jobs": -1}
        return LinearRegressor(output_directory,kwargs, type=regressor_name)
    
    if regressor_name == "random_forest" :
        kwargs = {"n_estimators": 100,"n_jobs": -1,"random_state":0,"verbose":False}
        return RFRegressor(output_directory,kwargs)
    
    if regressor_name == "xgboost":
        kwargs = {"n_estimators": 100,"n_jobs": 0,"verbosity": verbose}
        return XGBReg(output_directory,kwargs)
    
    if regressor_name == "SVR":
        kwargs = {"kernel": "rbf","gamma": "scale"}
        return SVRegressor(output_directory,kwargs)

    

def calculate_metrics(y_true,y_pred):
    res = pd.DataFrame(data=np.zeros((1, 2), dtype=np.float), index=[0],columns=['rmse', 'mae'])
    res['rmse'] = math.sqrt(mean_squared_error(y_true, y_pred))
    res['mae'] = mean_absolute_error(y_true, y_pred)
    # res['mape'] = mean_absolute_percentage_error(y_true,y_pred)
    return res

    # mean absolute error pecentage 