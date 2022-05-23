import warnings
from ast import arguments
from locale import normalize
import pandas as pd
from utils.regressor import create_regressor, calculate_metrics
from utils.data_loader import load_from_tsfile_to_dataframe
from utils.tools import create_directory, min_len
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from tsfresh import extract_relevant_features
from sklearn.model_selection import train_test_split
from utils.ts_tools import transform_data
import argparse
from utils.experience import *

''''
TODO: utiliser le francais uniquement dans code pour harmoniser 
TODO: choisir strategie de tsfresh de création de feature en argument
TODO: possibilité d'écrire l'extraction de features dans un fichier 
TODO: possibilité de choisir la taille de train size 
TODO: Possibilité d'écrire les predictions dans un fichier
TODO: Pour chaque regresseur utilisé écrire les paramètre trouvé lors de HPO dans un fichier  
TODO: Quelques visualisations
'''

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--problem", required=False,
                    nargs="+", default=["BeijingPM10Quality"])
parser.add_argument("-c", "--regressors", required=False,
                    nargs="+", default=["xgboost","random_forest","SVR","linear_regression"])
parser.add_argument("-o", "--output", required=False, default="output/")
parser.add_argument("-fx", "--features_extraction",
                    required=False, default=None)
parser.add_argument("-n", "--normalise", required=False, default="minmax")
parser.add_argument("-cv", "--cross_validation", required=False, default=3)
parser.add_argument("-ni", "--n_iteration", required=False, default=5)
parser.add_argument("-lib", "--librairie", required=False, default='tsfresh')
arguments = parser.parse_args()


data_path = "data/"
regressors = arguments.regressors
problems = arguments.problem
fx = arguments.features_extraction == "True"
normalisation = arguments.normalise
output_dir = arguments.output
cv =  int(arguments.cross_validation)
ni = int(arguments.n_iteration)
lib = arguments.librairie

if __name__ == '__main__':

    for pb in problems:
        print("--------- Traitement de : {} ---------".format(pb))
        exp = Experience(regressors,output_dir,data_path,pb)
        exp.changer_config_general({
            'normalisation': normalisation,
            'train_size': 0.7,
            'cv': cv,
            'scorer': 'neg_root_mean_squared_error',
            'optimisation': 'randomized_search',
            'extraction_features': True,
            'librairie_extraction_features': 'tsfel',
            'n_iter':ni,
            'sauvegarder_boxplot': True,
            'sequential_forward_search': False,
            'select_k_best': True

        })
        exp.low_cost_analysis()
        
