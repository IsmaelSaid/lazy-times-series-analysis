import math
import numpy as np
import pandas as pd
from matplotlib.transforms import Transform
from scipy.stats import kendalltau
from sklearn import feature_extraction
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import (EfficientFCParameters,
                                        MinimalFCParameters)
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import add_sub_time_series_index

from utils.regressor import calculate_metrics, create_regressor
from utils.regressor_tools import min_len, process_data
from utils.tools import create_directory
from utils.ts_tools import transform_data
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFdr
from sklearn.decomposition import PCA
from tsfel import * 





params_rs_default = {
    'linear_regression':[{}],
    'random_forest':[{}],
    'xgboost':[{}],
    'SVR':[{}]
}




class Experience():

    def __init__(self,regresseurs,dossier_sortie,x,y,nom_probleme):

        '''
        --------------------------------------------------------------------
        Configuration
        '''
        self.config_general = {
            'normalisation' : False,
            'train_size' : 0.7,
            'cv':10,
            'scorer':'neg_root_mean_squared_error',
            'optimisation':GridSearchCV,
            'extraction_features':True,
            'librairie_extraction_features':'tsfresh'

        }

        
        self.config_tsfresh = {
            'choix_features':EfficientFCParameters()
        }

        self.config_tsfel = {
            'choix_features':tsfel.get_features_by_domain()
        }



        self.params_rs_default = {
            'linear_regression':{
                "fit_intercept": [True,False]},
            'random_forest':{
                'n_estimators':[20,40,60,80,100],
                'max_features':['log2','auto','sqrt'],
                'max_depth':[4,8,12,16]
        },
            'xgboost':[{}],
            'SVR':{
                "kernel": ["poly","rbf"],
                "gamma":["scale"]
                }
        }

        self.optimisation_features = {
            'alpha'  : np.logspace(start=0.01,endpoint=0,stop=0.0001)/100,
            'k_best' : [10,20,30,40,50]
            
        }

        self.optimisation = {
            'random_forest':{
                'reg__n_estimators':[20,40,60,80,100],
                'reg__max_features':['log2','auto','sqrt'],
                'reg__max_depth':[4,8,12,16]
                }
        }

        '''
        -----------------------------------------------------------------------------------
        '''
        

        self.nom_probleme = nom_probleme
        self.og_x = x
        self.og_y = y
        self._X = None 
        self._Y = None 
        self.Regresseurs = regresseurs
        self.dossier_sortie = dossier_sortie

    '''
    TODO Ajouter a la classe des configurations préfaites 
    TODO Ajouter au constructeur la possibilité de choisir une configuration préfaite
    TODO pour chaque paramètre, ajouter la possibilité de la modifier individuellement
    '''

        # ------------------------------------------------------Configuration experience  -------------------------------------------------------------
    def changer_librairie_extraction(self, nom_librairie):
        self.config_general['librairie_extraction_features'] = nom_librairie

    def changer_config_general(self,nouvelle_config):
        self.config_general = nouvelle_config

    def changer_config_tsfresh(self,nouvelle_config):
        self.config_tsfresh = nouvelle_config
    
    def changer_config_optimisation_features(self,nouvelle_config):
        self.changer_config_optimisation_features = nouvelle_config

    def maj_espace_recherche_k(self,espace_recherche):
        self.optimisation_features['k_best'] = espace_recherche


    def init_regresseurs(self):
        self.dico_regresseurs = {}
        
        for nom_reg in self.Regresseurs:
            sortie = self.dossier_sortie + self.nom_probleme + nom_reg
            self.dico_regresseurs[nom_reg] = create_regressor(nom_reg, sortie)

    

    def transformation(self):
        self._X, self._Y = self.transform_data(
            self.og_x,
            self.og_y,
            normalise = self.config_general['normalisation'],
            features_extraction = self.config_general['extraction_features'],
            librairie=self.config_general['librairie_extraction_features']
            )

        # return (self._X,self._Y)





        # ------------------------------------------------------Comparaison modèles  -------------------------------------------------------------
    def experience_train_test(self,test_size = 0.3):
        '''
        TODO sauvegarde dataframe 
        '''
        resultat = {}
        x_train , x_test , y_train, y_test = train_test_split(self._X,self._Y,test_size=test_size,shuffle=False)
        for clee,val in self.dico_regresseurs.items(): 
            print("--------- Algorithme : {} ---------".format(clee))
            regresseur = val 
            regresseur.fit(x_train,y_train)
            y_pred = regresseur.predict(x_test)
            resultat[clee] = {
                'RMSE' : math.sqrt(mean_squared_error(y_test, y_pred)),
                'MAPE' : mean_absolute_error(y_test, y_pred) ,
                'MAE':  mean_absolute_percentage_error(y_test,y_pred)
            }
        return resultat

    def comparer_modele(self,cv = 5,sauvegarder_plotbox = False):
        '''
        TODO sauvegarde boxplot 
        '''
        resultat = {}
        for clee,val in self.dico_regresseurs.items():
            print(clee)
            reg = val 
            regreseur = reg.model
            resultat[clee] = cross_val_score(regreseur,self._X,self._Y,cv=cv,scoring = self.config_general['scorer'] )
        if(sauvegarder_plotbox == True):
            # Sauvegarder plot quelquepart 
            pass 
        return resultat

        # ------------------------------------------------------ Optimisation hyperparamètres -------------------------------------------------------------
    def optimisation_hpo(self,strategie,niter = 10,cv = 5):
        '''
        TODO description
        '''
        resultat = {}
        if strategie == 'randomized_search':
            for clee, val in  self.dico_regresseurs.items():
                print("Optimisation {}".format(clee))
                print(self.params_rs_default[clee])
                regresseur = val
                rand_search = RandomizedSearchCV(
                    # allez chercher les paramètres dans le dictionnaire 
                    estimator = regresseur.model,
                    param_distributions = self.params_rs_default[clee],
                    cv = cv,
                    n_iter = niter,
                    scoring = self.config_general['scorer']
                    )
                rand_search.fit(self._X,self._Y)
                self.dico_regresseurs[clee].model = rand_search.best_estimator_
                resultat[clee] = {
                    'meilleur' : rand_search.best_estimator_
                }

        elif strategie == 'grid_search':
            for clee, val in  self.dico_regresseurs.items():
                print("Optimisation {}".format(clee))
                print(self.params_rs_default[clee])
                regresseur = val
                grid_search = GridSearchCV(
                    estimator = regresseur.model,
                    param_grid = self.params_rs_default[clee],
                    cv = cv,
                    scoring = self.config_general['scorer']
                    )
                grid_search.fit(self._X,self._Y)
                self.dico_regresseurs[clee].model = grid_search.best_estimator_
                resultat[clee] = {
                    'meilleur' : grid_search.best_estimator_
                }
        return resultat
    
    
    def custom_kendall_tau(self,x,y):
        '''
        TODO description 
        '''
        liste_tau = []
        liste_pvalues = []
        for values in np.array(x).T:
            tau, p_value  = kendalltau(values,y)
            liste_tau.append(tau)
            liste_pvalues.append(p_value)

        return (liste_tau,liste_pvalues)

        # ------------------------------------------------------Optimisation nombre de features -------------------------------------------------------------


    def optimisation_nombre_features_kendall_tau(self, nom_nodele, strategie_recherche="grid_searchcv",cv=5):
        '''
        Pour la visualisation
        Selectionne les attributs en fonctions du test de kendall tau et procédure BH
        entrees: 
        sortie: une pipeline contenant la l'optimisatin des features et le regresseur donné en argument
        TODO: Strategie randomized search
        
        '''

        regresseur = self.dico_regresseurs.get(nom_nodele).model
        pipe = Pipeline([
            ('sel',SelectFdr(score_func=self.custom_kendall_tau)),
            ('reg',regresseur)])
        grid_search = GridSearchCV(
            estimator = pipe,
            param_grid={'sel__alpha':self.optimisation_features.get('alpha')},
            scoring = 'neg_mean_squared_error',
            cv=cv
            )
        grid_search.fit(self._X,self._Y)
        return grid_search




       




 




    




    

            




        
        




        

             

            

        
         


                

                

        

        # ------------------------------------------------------Transformation des données  -------------------------------------------------------------
        
    # TODO Il faudrait que les formatter puissent travailler sur des données déjà process 
    
    def tsfresh_format(self,X,Y,normalise = None,verbose = True):
        print('TSFRESH format')
        x = process_data(X,min_len(X),normalise)
        X_df = pd.DataFrame(x.reshape(x.shape[0]*x.shape[1],x.shape[2]))
        records_length = x.shape[1]
        
        # Transformation vers df 
        X_with_id = add_sub_time_series_index(X_df,records_length)
        
        # On va rajouter la colonne time, permettant d'identifier a quel moment la mesure a était prise 
        X_tsfresh_id_timed = pd.concat([X_with_id,pd.Series([i%records_length for i in range(len(X_with_id))],name="time")],axis=1)
        _Y = pd.DataFrame(
            X_tsfresh_id_timed.loc[:,"id"].unique()).reset_index(drop=True,inplace=False)
        Y_tsf = pd.DataFrame(columns=["id","target"])
        Y.reset_index(drop=True,inplace=True)
        Y_tsf["id"] = _Y
        Y_tsf["target"] = Y
        
        ex = extract_relevant_features(
            X_tsfresh_id_timed,Y_tsf["target"],
            column_id="id",
            column_sort="time",
            default_fc_parameters = self.config_tsfresh['choix_features'])
        return (ex,Y)


    def tsfel_format(self,X,Y,normalise = None, verbose = True):
        '''
        TODO 
        '''

        print('TSFEL format')
        x = process_data(X,min_len(X),normalise)
        X_df = pd.DataFrame(x.reshape(x.shape[0]*x.shape[1],x.shape[2]))
        taille_serie_temporelle = x.shape[1]
        nombre_serie_temporelle = x.shape[0]
        print("taille series temporelles = {}".format(taille_serie_temporelle))
        print("id différent  = {}".format(nombre_serie_temporelle))

        cfg_file = self.config_tsfel['choix_features']
        result = tsfel.time_series_features_extractor(cfg_file,X_df,fs=None,window_size = taille_serie_temporelle) 
        return (result,Y)

        
       







       




        


        

        
    def transform_data(self,X,Y,librairie,normalise = None, features_extraction = None):
        # print("Transformation des données")
        # print("Normalisation :{}".format(normalise))
        

        if(features_extraction == True):
            if(librairie == 'tsfresh'):
                return self.tsfresh_format(X,Y,normalise = normalise)
            elif(librairie == 'tsfel'):
                return self.tsfel_format(X,Y,normalise = normalise)
        else:
    #         Vérifier les dimensions
            x = process_data(X,min_len(X),normalise = normalise)
    #     TODO: il n'est plus necessaire de faire des vérifications avec les regresseurs, 
        # désormais tout les données en entrée des regresseurs sont des dataframes 2D
            if len(x.shape) == 3:
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            _x = pd.DataFrame(x)
            return _x,Y
            
