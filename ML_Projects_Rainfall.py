from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from ydata_profiling import ProfileReport
from lightgbm import LGBMClassifier
import optuna as opt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from catboost import CatBoostClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# ----------------------------------------------------------------------------------------------------------------------
# -1. Settings (proposed)
# ----------------------------------------------------------------------------------------------------------------------
random.seed(33)
pd.set_option('display.max_columns', None)
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

# ----------------------------------------------------------------------------------------------------------------------
# 0. Loading data
# ----------------------------------------------------------------------------------------------------------------------
data = pd.read_csv('weatherAUS.csv')

# ----------------------------------------------------------------------------------------------------------------------
# 1. Data overview
# ----------------------------------------------------------------------------------------------------------------------
# Instant look at shape, and NaNs:
print(data.shape)
print(data.head(10), data.isna().sum(), data.dtypes)

# Complete summary:
# profile = ProfileReport(data)
# profile.to_file("rainfall_view.html")

# ----------------------------------------------------------------------------------------------------------------------
# 2. Data preparation
# ----------------------------------------------------------------------------------------------------------------------
data['RainTomorrow'].value_counts(normalize=True).plot(kind='bar')
plt.show()
# Target value is highly unbalanced. First we change it to numeric, later we resample it before splitting to train and
# test, being aware of creating a possible small data leakage, hopefully leveraged by the better model generalization
# abilities

data['RainToday'].replace({'Yes': 1, 'No': 0}, inplace=True)
data['RainTomorrow'].replace({'Yes': 1, 'No': 0}, inplace=True)

negative_preds = data[data['RainTomorrow'] == 0]
positive_preds = data[data['RainTomorrow'] == 1]

positive_preds_oversampled = resample(positive_preds, replace=True, n_samples=len(negative_preds), random_state=33)
data_oversampled = pd.concat([negative_preds, positive_preds_oversampled])

data_oversampled['RainTomorrow'].value_counts(normalize=True).plot(kind='bar')
plt.show()

data = data_oversampled

# Filling NaNs with most frequent value
def nan_mode_filler(df, features):
    for feature in features:
        df[feature].fillna(df[feature].mode()[0], inplace=True)

nan_mode_filler(data, ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'])

# Temporarily dropping categorical cols
cat_cols = ['Date', 'Location','WindGustDir', 'WindDir9am', 'WindDir3pm' ]
removed_cols = pd.DataFrame(data, columns=cat_cols)
data.drop(cat_cols, axis=1, inplace=True)

# Droping target value, and 'RISK_MM' value which creates high data leakage
y = data['RainTomorrow']
data.drop(['RainTomorrow', 'RISK_MM'], axis=1, inplace=True)

# Filling NaNs with IterativeImputer based on multivariate imputation by chained equations (MICE), and merging cols
# together
iterative_imputer = IterativeImputer()
data.iloc[:, :] = iterative_imputer.fit_transform(data)

imputed_data = pd.concat([data, removed_cols], axis=1)

# ----------------------------------------------------------------------------------------------------------------------
# 3. Feature Engineering
# ----------------------------------------------------------------------------------------------------------------------
# Creating new time features
imputed_data['Date'] = pd.to_datetime(imputed_data['Date'])
imputed_data['Year'] = imputed_data['Date'].dt.year
imputed_data['Month'] = imputed_data['Date'].dt.month
imputed_data['Day'] = imputed_data['Date'].dt.day
imputed_data.drop(['Date'], axis=1, inplace=True)

# Creating 'Delta' features
imputed_data['Wind_Delta'] = imputed_data['WindSpeed3pm'] - imputed_data['WindSpeed9am']
imputed_data['Humidity_Delta'] = imputed_data['Humidity3pm'] - imputed_data['Humidity9am']
imputed_data['Pressure_Delta'] = imputed_data['Pressure3pm'] - imputed_data['Pressure9am']
imputed_data['Temp_Delta'] = imputed_data['Temp3pm'] - imputed_data['Temp9am']

# One-Hot Encoding by get_dummies, and to numeric
dummy_imputed_data = pd.get_dummies(imputed_data)
dummy_imputed_data.replace({True: 1, False: 0}, inplace=True)

# ----------------------------------------------------------------------------------------------------------------------
# 4. Model testing and hypertuning
# ----------------------------------------------------------------------------------------------------------------------
X = dummy_imputed_data
# Splitting data into train, valid and test sets
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

X_train, X_valid, y_train, y_valid = train_test_split(X_tr, y_tr, random_state=33)


# Hypertuning the models with optuna. Scaling is proceeded in pipelines to avoid any data leakage
def objective(trial, model_chosen):
    scaler = RobustScaler()
    if model_chosen == 1:
        #xgbc
        params = {
                        "gamma": trial.suggest_int("gamma",0,0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree",0,1),
                        "min_child_weight": trial.suggest_int("min_child_weight",0,20),
                        "max_depth": trial.suggest_int("max_depth",0,20),
                        "n_estimators": trial.suggest_int("n_estimators",1000,4500),
                        "alpha": trial.suggest_float("alpha",0.00001,75),
                        "learning_rate": trial.suggest_float("learning_rate",0.001,1),
                        "colsample_bylevel": trial.suggest_float("colsample_bylevel",0,1),
                        "colsample_bynode": trial.suggest_float("colsample_bynode",0,1),
                        "random_state": trial.suggest_int("random_state",0,0),
                        "subsample": trial.suggest_float("subsample",0,1),
                        "lambda": trial.suggest_float("lambda", 0.001, 75)
                    }

        model = make_pipeline(scaler, XGBClassifier(**params))

    elif model_chosen == 2:
        # lgbm
        params = {
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1),
            "max_depth": trial.suggest_int("max_depth", 0, 20),
            "n_estimators": trial.suggest_int("n_estimators", 1000, 4000),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0, 1),
            "random_state": trial.suggest_int("random_state", 0, 0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 50),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0, 1),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 8),
            "bagging_seed": trial.suggest_int("bagging_seed", 0, 8),
            "feature_fraction_seed": trial.suggest_int("feature_fraction_seed", 0, 8),
            "verbose": trial.suggest_int("verbose", -1, -1)
        }

        model = make_pipeline(scaler, LGBMClassifier(**params))

    elif model_chosen == 3:
        # rf
        param = {'n_estimators': trial.suggest_int("n_estimators", 1000, 2000),
                 "max_depth": trial.suggest_int("max_depth", 1, 30),
                 "max_samples": trial.suggest_float("max_samples", 0.4, 1),
                 "max_features": trial.suggest_int("max_features", 1, 40),
                 "min_samples_split": trial.suggest_int("min_samples_split", 2, 5),
                 "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 5)

                 }

        model = make_pipeline(scaler, RandomForestClassifier(**param))

    elif model_chosen == 4:
        #ada
        param = {'n_estimators': trial.suggest_int("n_estimators", 50, 1500),
                 "learning_rate": trial.suggest_float("learning_rate", 0.001, 2),
                 "algorithm": trial.suggest_categorical("algorithm", ['SAMME', 'SAMME.R'])
                 }
        model = make_pipeline(scaler, AdaBoostClassifier(**param))

    elif model_chosen == 5:
        #catboost
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 0, 15),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        }


        model = make_pipeline(scaler, CatBoostClassifier(**param, silent=True))

    results = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=5)
    result = results.mean()

    return result

models = {'xgbc': 1, 'lgbm': 2, 'rf': 3, 'ada':4, 'cat': 5}

# study = opt.create_study(direction='maximize')
# study.optimize(lambda trial: objective(trial, models['rf']), n_trials=15, n_jobs=-1)
# print(study.best_trial.params)
# print(study.best_value)

# Best cross_val ROC_AUC score for these models:
# xgbc: 0.9699133549839732
# lgbm: 0.9691090055757423
# cat: 0.9708023351224775
# rf: 0.9648440997910359
# ada: 0.8841847893587417
# TOP3 has been chosen (xgbc, lgbm, cat) for final evaluation and for averaging predicitions

scaler = RobustScaler()

# Best hyperparameters:
#xgbc
params_xgbc = {'gamma': 0, 'colsample_bytree': 0.8830950034798855, 'min_child_weight': 16, 'max_depth': 10, 'n_estimators': 3451,
          'alpha': 0.19231792821014804, 'learning_rate': 0.15211549793002932, 'colsample_bylevel': 0.9998462060691387,
          'colsample_bynode': 0.9076813380237919, 'random_state': 0, 'subsample': 0.8048856625897495, 'lambda': 0.4485699994795813}
xgbc = XGBClassifier(**params_xgbc)
xgbc_pipe = make_pipeline(scaler, xgbc)

#lgbm
params_lgbm = {'colsample_bytree': 0.99218551285717, 'max_depth': 0, 'n_estimators': 1917, 'reg_alpha': 0.1316738749133919,
               'reg_lambda': 0.9209032905893757, 'learning_rate': 0.20343945326370325, 'colsample_bynode': 0.17541754095531037,
               'random_state': 0, 'num_leaves': 48, 'bagging_fraction': 0.7935330146415063, 'bagging_freq': 5, 'bagging_seed': 3,
               'feature_fraction_seed': 6, 'verbose': -1}
lgbm = LGBMClassifier(**params_lgbm)
lgbm_pipe = make_pipeline(scaler, lgbm)

#catboost
params_cat = {'learning_rate': 0.09265385938943387, 'colsample_bylevel': 0.08683012745870629, 'max_depth': 12,
              'boosting_type': 'Plain', 'min_data_in_leaf': 67}
cat = CatBoostClassifier(**params_cat, silent=True)
cat_pipe = make_pipeline(scaler, cat)


def final_test(model, X_train, X_valid, X_test, y_train, y_valid, y_test):
    model.fit(X_train, y_train)
    rc_score_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, -1])
    accuracy_test = model.score(X_test, y_test)
    rc_score_valid = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, -1])
    accuracy_valid = model.score(X_valid, y_valid)
    report = classification_report(y_test, model.predict(X_test))
    confusion = list(np.ravel(confusion_matrix(y_test, model.predict(X_test))))
    confusion = [str(x) for x in confusion]
    print(classification_report(y_test, model.predict(X_test)))
    print(confusion_matrix(y_test, model.predict(X_test)))

    return accuracy_valid, accuracy_test, rc_score_valid, rc_score_test


def final_test_stack(models, X_train, X_test, y_train, y_test):
    preds_valid = [0] * 44127
    preds_test = [0] * 44127
    for model in models:
        model.fit(X_train, y_train)
        preds_valid += model.predict_proba(X_valid)[:, -1]
        preds_test += model.predict_proba(X_test)[:, -1]

    preds_valid = preds_valid / (len(models))
    preds_test = preds_test/(len(models))

    rc_score_valid = roc_auc_score(y_valid, preds_valid)
    preds_valid = np.round(preds_valid)
    accuracy_valid = accuracy_score(y_valid, preds_valid)
    rc_score_test = roc_auc_score(y_test, preds_test)
    preds_test = np.round(preds_test)
    accuracy_test = accuracy_score(y_test, preds_test)


    return accuracy_valid, accuracy_test, rc_score_valid, rc_score_test

# ----------------------------------------------------------------------------------------------------------------------
# 5. Results
# ----------------------------------------------------------------------------------------------------------------------

# print(final_test(xgbc, X_train, X_valid, X_test, y_train, y_valid, y_test))

# Best scores got XGBC:
# valid_accuracy: 0.9306773630656967, valid_rc_score: 0.9775549078184906
# test_accuracy: 0.9315385138350669, test_rc_score:0.9783065200769245

print(final_test_stack([xgbc_pipe, lgbm_pipe, cat_pipe], X_train, X_test, y_train, y_test))

# Best scores from averaged predictions of these 3 models (xgbc, lgbm, cat) better roughly 0.03:
# valid_accuracy: 0.9333514628232148, valid_rc_score: 0.9800651185639248
# test_accuracy: 0.9338953475196592, test_rc_score: 0.9810433315691748