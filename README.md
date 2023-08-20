# Time Series Extrinsic Regression
### Comparaison xgboost/tsfel random forest/tsfel
![](/out/Covid3Month-tsfel2023-August-20-05.png-1.png)

### Comparaison xgboost/tsfresh random forest/tsfresh
![](/out/Covid3Month-tsfresh2023-August-20-20.png-1.png) 


### Dépendances 

* sklearn, TSFRESH, TSFEL, numpy, matplotlib, xgboost, pandas, scipy, tqdm
* utilisez pip install pour installer les dépendances


### Execution du programme

* Cloner ce répertoire 
* Se placer à la racine 
* Exécutez la commande suivante 
```

python3 -W ignore .\main.py -p BeijingPM10Quality

```
Arguments:
```
-p --problem        : nom du problèmes (par défault BeijingPM10Quality)
-c --regressors     : nom des modèles (par défault tout les modèles sont inclus)
-n --normalisation  : normalisation (par défault minmax)
```
/!\ les données doivent être dans le répertoire data/

## Sources 
Les données  :  http://tseregression.org/

## Auteur

Said Ismael 


## Version History


