from calendar import c
from datetime import datetime
import math
from statistics import linear_regression
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox, Transform
from scipy.stats import kendalltau
from sklearn import feature_extraction
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from tqdm import tqdm
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import (EfficientFCParameters,
                                        MinimalFCParameters)
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import add_sub_time_series_index
from utils.data_loader import load_from_tsfile_to_dataframe

from utils.regressor import calculate_metrics, create_regressor
from utils.regressor_tools import min_len, process_data
from utils.tools import create_directory
from utils.ts_tools import transform_data
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFdr, SelectKBest, SequentialFeatureSelector
from sklearn.decomposition import PCA
from tsfel import *
from sklearn.feature_selection import f_regression, mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
from utils.tools import create_directory

params_rs_default = {
    'linear_regression': [{}],
    'random_forest': [{}],
    'xgboost': [{}],
    'SVR': [{}]
}


class Experience():

    def __init__(self, regresseurs, dossier_sortie,dossier_donnees, nom_probleme):
        '''
        --------------------------------------------------------------------
        Configuration
        '''
        self.config_general = {
            'normalisation': True,
            'train_size': 0.7,
            'cv': 5,
            'scorer': 'neg_root_mean_squared_error',
            'optimisation': 'randomized_search',
            'extraction_features': True,
            'librairie_extraction_features': 'tsfresh',
            'n_iter': 5,
            'sauvegarder_boxplot': True,
            'sequential_forward_search': False,
            'select_k_best': True

        }

        self.config_tsfresh = {
            'choix_features': EfficientFCParameters()
        }

        self.config_tsfel = {
            'choix_features': tsfel.get_features_by_domain('statistical')
        }

        self.params_rs_default = {
            'linear_regression': {
                "fit_intercept": [True, False]},
            'random_forest': {
                'n_estimators': [i for i in np.arange(300,1000,100)],
                'max_features': ['log2','sqrt'],
                'max_depth': [4, 8, 12, 16]
            },
             'xgboost': {
                "eta":[0.1,0.05,0.01],
                "gamma":[0.1],
                "max_depth":[5,10,15,20],
                "subsample":[i for i in np.arange(0.25,0.5,0.1)],
                "colsample_bytree":[i for i in np.arange(0.25,0.5,0.1)],
                "alpha": [0.5],
                "tree_method": ["approx"],
                "objective": ["reg:linear"],
                "eval_metric": ["rmse"]
                },
            'SVR': {
                "kernel": ["poly", "rbf"],
                "gamma": ["scale","auto"],
                "shrinking": [True],
                "C": [1],
                "epsilon": [1]
            }
        }


        self.optimisation = {
            'random_forest': {
                'reg__n_estimators': [i for i in np.arange(300,1000,100)],
                'reg__max_features': ['log2', 'sqrt'],
                'reg__max_depth': [i for i in np.arange(5,15,5)]
            },
            'linear_regression': {
                "reg__fit_intercept": [True, False]
            },
            'xgboost': {
                "reg__eta":[0.1,0.05,0.01],
                "reg__gamma":[0.1],
                "reg__max_depth":[5,10,15,20],
                "reg__subsample":[i for i in np.arange(0.25,0.5,0.1)],
                "reg__colsample_bytree":[i for i in np.arange(0.2,0.9,0.1)],
                "reg__alpha": [i for i in np.arange(1,10,1)],
                "reg__tree_method": ["approx"],
                "reg__objective": ["reg:linear"],
                "reg__eval_metric": ["rmse"]
                },
            'SVR': {
                "reg__kernel": ["poly", "rbf"],
                "reg__gamma": ["scale","auto"],
                "reg__shrinking": [True],
                "reg__C": [i for i in np.arange(0.5,1,0.1)],
                "reg__epsilon": [i for i in np.arange(0.5,1,0.1)]
            }
        }
        '''
        TODO: maj hp
        '''

        # Visualisation
        sns.set(rc={'figure.figsize': (15, 8)})

        '''
        -----------------------------------------------------------------------------------
        '''
        
        self.nom_probleme = nom_probleme
        self.Regresseurs = regresseurs
        self.dossier_sortie = dossier_sortie
        self.dossier_donees = dossier_donnees
        self.og_x = None
        self.og_y = None
        self._X = None
        self._Y = None

    '''
    TODO Ajouter a la classe des configurations préfaites 
    TODO Ajouter au constructeur la possibilité de choisir une configuration préfaite
    TODO pour chaque paramètre, ajouter la possibilité de la modifier individuellement
    '''

    # ------------------------------------------------------Configuration experience  -------------------------------------------------------------
    def changer_librairie_extraction(self, nom_librairie):
        self.config_general['librairie_extraction_features'] = nom_librairie

    def changer_config_general(self, nouvelle_config):
        self.config_general = nouvelle_config

    def changer_config_tsfresh(self, nouvelle_config):
        self.config_tsfresh = nouvelle_config

    def changer_config_optimisation_features(self, nouvelle_config):
        self.changer_config_optimisation_features = nouvelle_config

    def maj_espace_recherche_k(self, espace_recherche):
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
            normalise=self.config_general['normalisation'],
            features_extraction=self.config_general['extraction_features'],
            librairie=self.config_general['librairie_extraction_features']
        )

        # return (self._X,self._Y)

        # ------------------------------------------------------Comparaison modèles  -------------------------------------------------------------

    def experience_train_test(self, test_size=0.3):
        '''
        TODO sauvegarde dataframe 
        '''
        resultat = {}
        x_train, x_test, y_train, y_test = train_test_split(
            self._X, self._Y, test_size=test_size, shuffle=False)
        for clee, val in self.dico_regresseurs.items():
            print("--------- Algorithme : {} ---------".format(clee))
            regresseur = val
            regresseur.fit(x_train, y_train)
            y_pred = regresseur.predict(x_test)
            resultat[clee] = {
                'RMSE': math.sqrt(mean_squared_error(y_test, y_pred)),
                'MAPE': mean_absolute_error(y_test, y_pred),
                'MAE':  mean_absolute_percentage_error(y_test, y_pred)
            }
        return resultat

    def experience_train_test(self, nom_regresseur):
        '''
        TODO sauvegarde dataframe 
        '''
        resultat = {}
        x_train, x_test, y_train, y_test = train_test_split(
            self._X, self._Y, test_size=self.config_general['train_size'], shuffle=False)

        regresseur = self.dico_regresseurs[nom_regresseur].model
        start_time = time.perf_counter()
        regresseur.fit(x_train, y_train)
        train_duration = time.perf_counter() - start_time
        print('[{}] temps entraînement {:.3f}s'.format(nom_regresseur, train_duration))
        y_pred = regresseur.predict(x_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))

        return rmse


    def comparer_modele(self):
        '''
        TODO sauvegarde boxplot 
        '''
        resultat = {}
        score={}
        for clee, val in self.dico_regresseurs.items():
            reg = val
            regreseur = reg.model
            resultat[clee] = cross_val_score(
                # regreseur, self._X, self._Y, cv=self.config_general['cv'], scoring=self.config_general['scorer'])
                regreseur, self._X, self._Y, cv=10, scoring=self.config_general['scorer'])
            score[clee] = np.mean(resultat[clee])
            print("[{}] {}  : {:.3f}".format(clee,self.config_general['scorer'],score[clee] * -1))

        if(self.config_general['sauvegarder_boxplot'] == True):
            finale = []
            titre = str("nom problème : {} \n librairie : {} \n strategie de recherche : {} \n métrique : {} \n extraction features : {} \n cv : {} \n n iter : {}").format(
                self.nom_probleme,
                self.config_general['librairie_extraction_features'],
                self.config_general['optimisation'],
                self.config_general['scorer'],
                str(self.config_general['extraction_features']),
                self.config_general['cv'],
                self.config_general['n_iter']

            )
            for clee, val in self.dico_regresseurs.items():
                moyenne =  np.mean(resultat[clee])
                res = str("Modèle :{} \n moyenne rmse : {:.3f} ").format(
                    clee,
                   score[clee] * -1)
                finale.append(res)

            fig, ax = plt.subplots(figsize=(15, 15))
            ax.set_title(titre)
            ax.boxplot(resultat.values(), labels=finale)
            create_directory(self.dossier_sortie)
            time = datetime.now()

            nom_fichier = self.nom_probleme+'-'+self.config_general['librairie_extraction_features']+"{}-{}-{}-{}.png".format(time.strftime("%Y"),time.strftime("%B"),time.strftime("%d"),time.strftime("%S"))
            plt.savefig(self.dossier_sortie+'/'+nom_fichier)

            # Sauvegarder plot quelquepart
            pass
        return self.dico_regresseurs

        # ------------------------------------------------------ Optimisation hyperparamètres -------------------------------------------------------------
    def optimisation_hpo(self):
        '''
        TODO description
        '''
        resultat = {}
        
        if self.config_general["optimisation"] == 'randomized_search':
            for clee, val in self.dico_regresseurs.items():
                regresseur = val
                rand_search = RandomizedSearchCV(
                    # allez chercher les paramètres dans le dictionnaire
                    estimator=regresseur.model,
                    param_distributions=self.params_rs_default[clee],
                    cv=self.config_general["cv"],
                    n_iter=self.config_general['n_iter'],
                    scoring=self.config_general['scorer']
                )
                rand_search.fit(self._X, self._Y)
                self.dico_regresseurs[clee].model = rand_search.best_estimator_
                resultat[clee] = {
                    'meilleur': rand_search.best_estimator_
                }

        elif self.config_general["optimisation"] == 'grid_searchcv':
            for clee, val in self.dico_regresseurs.items():
                regresseur = val
                grid_search = GridSearchCV(
                    estimator=regresseur.model,
                    param_grid=self.params_rs_default[clee],
                    cv=self.config_general["cv"],
                    scoring=self.config_general['scorer']
                )
                grid_search.fit(self._X, self._Y)
                self.dico_regresseurs[clee].model = grid_search.best_estimator_
                resultat[clee] = {
                    'meilleur': grid_search.best_estimator_
                }
        return resultat

        # ------------------------------------------------------Optimisation nombre de features -------------------------------------------------------------

    def recuperer_model(self, nom_model):
        return self.dico_regresseurs.get(nom_model)

    def fscore(self):
        '''
        TODO: Description
        '''
        fs = SelectKBest(score_func=f_regression, k='all')
        fs.fit(self._X, self._Y)
        result = pd.DataFrame({
            'nom_features': [nom_feature for nom_feature in fs.feature_names_in_],
            'fscore': [fscore for fscore in fs.scores_]})

        return result

    def mutual_regression(self):
        '''
        TODO: Description
        '''
        fs = SelectKBest(score_func=mutual_info_regression, k='all')
        fs.fit(self._X, self._Y)
        result = pd.DataFrame({
            'nom_features': [nom_feature for nom_feature in fs.feature_names_in_],
            'fscore': [fscore for fscore in fs.scores_]})

        return result

    # Exploration

    def optimisation_nombre_feature(self):
        resultat = dict()
        tmp = self.dico_regresseurs.copy()
        if(self.config_general['select_k_best'] == True):
            num_features = [i for i in range(2, self._X.columns.size)]
            grid = dict()
            grid['sel__k'] = num_features
            grid['sel__score_func'] = [f_regression,mutual_info_regression]
            for clee, val in tmp.items():
                pipeline = Pipeline(
                    steps=[('sel', SelectKBest()), ('reg', val.model)])

                search = RandomizedSearchCV(estimator=pipeline,
                                            param_distributions=grid | self.optimisation[clee],
                                            scoring=self.config_general['scorer'],
                                            n_jobs=-1,
                                            cv=self.config_general['cv'],
                                            n_iter=self.config_general['n_iter'])

                search.fit(self._X, self._Y)
                nouveau_regresseur = create_regressor(clee, self.dossier_sortie)
                nouveau_regresseur.model = search.best_estimator_
                
                
                
                self.dico_regresseurs[clee+'_selectkbest'] = nouveau_regresseur

        if(self.config_general['sequential_forward_search'] == True):
            grid_sfs = dict()
            direction = ['forward']
            grid_sfs['sfs__direction'] = direction
            for clee, val in tmp.items():
                from sklearn.linear_model import LinearRegression
                pipeline = Pipeline(steps=[('sfs', SequentialFeatureSelector(
                    LinearRegression(), n_features_to_select=0.25)), ('reg', val.model)])

                search = RandomizedSearchCV(estimator=pipeline,
                                            param_distributions=grid_sfs | self.optimisation[clee],
                                            scoring=self.config_general['scorer'],
                                            n_jobs=-1,
                                            cv=self.config_general['cv'],
                                            n_iter=self.config_general['n_iter'])

                search.fit(self._X, self._Y)
                nouveau_regresseur = create_regressor(
                    clee, self.dossier_sortie)
                nouveau_regresseur.model = search.best_estimator_
                self.dico_regresseurs[clee+'_sfs'] = nouveau_regresseur

    def plot_mutual_regression(self):
        plt.figure(figsize=(20, 10))
        mr = self.mutual_regression()
        mr = mr.sort_values(by='fscore', inplace=False, ascending=False)
        sns.barplot(data=mr, x='fscore', y='nom_features', palette='rocket')
        plt.show()

    def plot_fscore(self):
        plt.figure(figsize=(20, 10))
        fs = self.fscore()
        fs = fs.sort_values(by='fscore', inplace=False, ascending=False)
        sns.barplot(data=fs, x='fscore', y='nom_features', palette='rocket')
        plt.show()
        # ------------------------------------------------------Transformation des données  -------------------------------------------------------------

    # TODO Il faudrait que les formatter puissent travailler sur des données déjà process

    def tsfresh_format(self, X, Y, normalise=None, verbose=True):
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
            X_with_id, self.og_y,
            column_id="id",
            default_fc_parameters=self.config_tsfresh['choix_features'])
        return (ex,Y)

    def tsfel_format(self, X, Y, normalise=None, verbose=True):
        '''
        TODO 
        '''

        x = process_data(X, min_len(X), normalise)
        X_df = pd.DataFrame(x.reshape(x.shape[0]*x.shape[1], x.shape[2]))
        taille_serie_temporelle = x.shape[1]
        nombre_serie_temporelle = x.shape[0]
        cfg_file = self.config_tsfel['choix_features']
        result = tsfel.time_series_features_extractor(
            cfg_file, X_df, fs=None, window_size=taille_serie_temporelle)
        return (result, Y)

    def transform_data(self, X, Y, librairie, normalise, features_extraction=None):
        if(features_extraction == True):
            if(librairie == 'tsfresh'):
                return self.tsfresh_format(X, Y, normalise=normalise)
            elif(librairie == 'tsfel'):
                return self.tsfel_format(X, Y, normalise=normalise)
        else:
            #         Vérifier les dimensions
            x = process_data(X, min_len(X), normalise=normalise)
    #     TODO: il n'est plus necessaire de faire des vérifications avec les regresseurs,
        # désormais tout les données en entrée des regresseurs sont des dataframes 2D
            if len(x.shape) == 3:
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            _x = pd.DataFrame(x)
            return _x, Y
        
    def load(self):
        train_file = self.dossier_donees + self.nom_probleme + "_TRAIN.ts"
        test_file = self.dossier_donees + self.nom_probleme + "_TEST.ts"
        X_train, y_train = load_from_tsfile_to_dataframe(train_file)
        X_test, y_test = load_from_tsfile_to_dataframe(test_file)

        self.og_x = pd.concat([X_train, X_test], axis=0)
        self.og_y = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)
        
        self.transformation()

        
        
    def run(self):
        warnings.filterwarnings("ignore")
        # debug
        self.init_regresseurs()
        self.load()
        print(len(self._X))
        print(self.config_general)
        
        # script

        print("TSFRESH")
        self.optimisation_hpo()
        self.optimisation_nombre_feature()
        self.comparer_modele()
        
        print("TSFEL")
        self.changer_librairie_extraction('tsfel')
        self.transformation()
        print(self._X.shape)
        self.init_regresseurs()
        self.optimisation_hpo()
        self.optimisation_nombre_feature()
        self.comparer_modele()
        
        print("Originaux")
        self.config_general['extraction_features'] = False
        self.transformation()
        self.init_regresseurs()
        self.optimisation_hpo()
        # self.optimisation_nombre_feature()
        self.comparer_modele()
    
        
  
              
             
