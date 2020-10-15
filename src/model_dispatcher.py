# -*- coding: utf-8 -*-
"""モデルディスパッチャー

学習するモデルの一覧を記載したモジュール. train.pyの引数にmodelsのkeyを指定して、
該当のモデルを学習するための割り当てをする

Attributes：
    models(dict): モデル名をkey、モデルインスタンスをvalueとした辞書

"""
import config
from models import (
    ModelXGB,
    ModelRandomForestClassifier,
    ModelDecisionTreeClassifier,
    ModelLogisticRegression,
    ModelLGB,
    ModelAE,
    ModelSVM,
    ModelBernoulliNB,
    ModelGaussianNB,
    )


# XGBoostのハイパーパラメータ設定
params_xgb = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 9,
    'max_depth': 12,
    'eta': 0.1,
    'min_child_weight': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'silent': 1,
    'random_state': config.RANDOM_SEED,
    'num_boost_round': 200,  # 10000
    'early_stopping_rounds': 10,
}

# LightGBMのハイパーパラメータ設定
params_lgb = {
    'task': 'train',
    'num_class': 1,
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'metric_freq': 50,
    'is_training_metric': False,
    'max_depth': 4,
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'bagging_seed': config.RANDOM_SEED,
    'verbose': 0,
    'num_boost_round': 1000,  # 20000
    'early_stopping_rounds': 200,
}

# モデルディスパッチャ

models = {
    'decision_tree_gini': ModelDecisionTreeClassifier(
        params={'criterion': 'gini'},
    ),
    'decision_tree_entropy': ModelDecisionTreeClassifier(
        params={'criterion': 'entropy'},
    ),
    'logistic_regression': ModelLogisticRegression(
        params={'n_jobs': -2},
    ),
    'random_forest': ModelRandomForestClassifier(
        params={'n_jobs': -2},
    ),
    'xgboost': ModelXGB(
        params=params_xgb,
    ),
    'lightgbm': ModelLGB(
        params=params_lgb,
    ),
    'auto_encoder': ModelAE(
        params={},
    ),
    'svm_rbf': ModelSVM(
        params={'C': 1.0,
                'kernel': 'linear',
                'probability': True,
                'max_iter': 10,
                },
    ),
    'bernoulli_nb': ModelBernoulliNB(
        params={},
    ),
    'gaussian_nb': ModelBernoulliNB(
        params={}
    ),
}
