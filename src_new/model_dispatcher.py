# -*- coding: utf-8 -*-
"""モデルディスパッチャー

学習するモデルの一覧を記載したモジュール. train.pyの引数にmodelsのkeyを指定して、
該当のモデルを学習するための割り当てをする

Attributes：
    models(dict): モデル名をkey、モデルインスタンスをvalueとした辞書

"""
from models import (
    ModelXgb,
    ModelRandomForestClassifier,
    ModelDecisionTreeClassifier,
    ModelLogisticRegression,
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
    'random_state': 71,
    'num_round': 200,  # 10000
    'early_stopping_rounds': 10,
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
    'xgboost': ModelXgb(
        params=params_xgb,
    ),
}
