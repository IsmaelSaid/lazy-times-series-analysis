from ast import arguments
import pandas as pd 
from utils.regressor import create_regressor,calculate_metrics
from utils.data_loader import load_from_tsfile_to_dataframe
from utils.tools import create_directory, min_len
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from tsfresh import extract_relevant_features
from sklearn.model_selection import train_test_split
from utils.ts_tools import transform_data
import argparse

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
parser.add_argument("-d", "--data", required=False, default="data/")
parser.add_argument("-p", "--problem", required=False,nargs="+" ,default=["Sample"])  
parser.add_argument("-c", "--regressors", required=False,nargs="+", default=["random_forest"])  
parser.add_argument("-o", "--output", required=False, default="output/")
parser.add_argument("-fx", "--features_extraction", required=False, default=None)
parser.add_argument("-n", "--normalise", required=False, default=None)
arguments = parser.parse_args()


data_path="data/"
regressors = arguments.regressors
problems = arguments.problem
fx = arguments.features_extraction == "True"
arg_normalise = arguments.normalise
output_dir = arguments.output

if __name__ == '__main__':
    # parcours de tout les problèmes
    print("Dossier de sortie : {}".format(output_dir))
    print("Problèmes : {}".format(problems))
    print("Regressors : {}".format(regressors))
    print("Feature extraction : {}".format(fx))
    print("Normalise  : {}".format(arg_normalise))
    # print(arguments.regressors)
    for pb in problems:
        print("--------- Traitement de : {} ---------".format(pb))
        train_file = data_path + pb + "_TRAIN.ts"
        test_file = data_path + pb + "_TEST.ts"
        X_train, y_train = load_from_tsfile_to_dataframe(train_file)
        X_test, y_test = load_from_tsfile_to_dataframe(test_file)
        
        X = pd.concat([X_train,X_test],axis=0)
        Y = pd.concat([pd.Series(y_train),pd.Series(y_test)],axis=0)
        
      
        # --------------------------------------------------------------------------------------
        _X ,_Y = transform_data(X,Y,normalise=arg_normalise,features_extraction=fx)
        x_train , x_test , y_train, y_test = train_test_split(_X,_Y,test_size=0.3,shuffle=False)
        # --------------------------------------------------------------------------------------

        for regressor_name in regressors:
            print("--------- Algorithme : {} ---------".format(regressor_name))
            # Application d'un modèle 
            output_directory = output_dir + pb +"/" + regressor_name +"/"
            create_directory(output_directory)
            regressor = create_regressor(regressor_name,output_directory)
            regressor.fit(x_train,y_train)
            y_pred = regressor.predict(x_test)
            df_metrics = calculate_metrics(y_test,y_pred)
            df_metrics.to_csv(output_directory+"metrics.csv",index=False)