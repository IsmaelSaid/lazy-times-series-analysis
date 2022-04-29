from utils.regressor_tools import process_data,min_len
from tsfresh.utilities.dataframe_functions import add_sub_time_series_index
from tsfresh.feature_extraction import MinimalFCParameters,EfficientFCParameters
from tsfresh import extract_relevant_features
import pandas as pd

def tsfresh_format(X,Y,normalise = None,verbose = True):
    """
    Input: X: Dataframe
    Input: Y: ground value
    """
    print("Conversion tsfresh format")
    print("Normalisation :{}".format(normalise))
    
    #Transformation vers numpy array
    # x = process_data(X,min_len = min_len(X),normalise = normalise)
    x = process_data(X,min_len(X),normalise)
    X_df = pd.DataFrame(x.reshape(x.shape[0]*x.shape[1],x.shape[2]))

    records_length = x.shape[1]
    # Transformation vers df 
    X_with_id = add_sub_time_series_index(X_df,records_length)
    # On va rajouter la colonne time, permettant d'identifier a quel moment la mesure a était prise 
    X_tsfresh_id_timed = pd.concat([X_with_id,pd.Series([i%records_length for i in range(len(X_with_id))],name="time")],axis=1)
    
    _Y = pd.DataFrame(X_tsfresh_id_timed.loc[:,"id"].unique()).reset_index(drop=True,inplace=False)
    Y_tsf = pd.DataFrame(columns=["id","target"])
    Y.reset_index(drop=True,inplace=True)
    Y_tsf["id"] = _Y
    Y_tsf["target"] = Y
    
    ex = extract_relevant_features(X_tsfresh_id_timed,Y_tsf["target"],column_id="id",column_sort="time",default_fc_parameters=EfficientFCParameters())
    
    return (ex,Y_tsf)



def transform_data(X,Y,normalise = None, features_extraction = None):
    '''
    X:
    Y: 
    normalise: 
    feature_extraction = 
    '''
    print("Transformation des données")
    print("Normalisation :{}".format(normalise))
    
    
    
    if(features_extraction == True):
        x,y = tsfresh_format(X,Y,normalise = normalise)
        return x,Y
    else:
#         Vérifier les dimensions
        x = process_data(X,min_len(X),normalise = normalise)
#     TODO: il n'est plus necessaire de faire des vérifications avec les regresseurs, 
    # désormais tout les données en entrée des regresseurs sont des dataframes 2D
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        _x = pd.DataFrame(x)
        return _x,Y
        
        
    
    
