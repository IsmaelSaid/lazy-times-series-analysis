# Time Series Extrinsic Regression


### Dependencies

* sklearn, TSFRESH, TSFEL, numpy, matplotlib


### Executing program

* How to run the program
```
reg = ['random_forest','SVR']
Exp  = Experience(
    nom_probleme="benzene",
    x = pd.concat([X_train,X_test],axis=0),
    y = pd.concat([pd.Series(Y_train),pd.Series(Y_test)],axis=0),
    regresseurs =  reg ,
    dossier_sortie = "default"
    )

# On précise que cette expérience réalise une extraction de features 
Exp.changer_config_tsfresh({
    'choix_features' :ComprehensiveFCParameters()
})


# Transformation des données de départ pour les rendres utilisables par les regresseurs 
# Peut réaliser une normalisation (minmax etc..) 
# Peut remplacer les données de départ par des features extraites
Exp.transformation()

# Innitialisation des différents regresseurs données en argument
Exp.init_regresseurs()

```


## Authors

Said Ismael 


## Version History


