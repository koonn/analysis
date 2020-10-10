# -*- coding: utf-8 -*-
"""モデルディスパッチャー

学習するモデルの一覧を記載したモジュール. train.pyの引数にmodelsのkeyを指定して、
該当のモデルを学習するための割り当てをする

Attributes：
    models(dict): モデル名をkey、モデルインスタンスをvalueとした辞書

"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from models import ModelXgb

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
    'num_round': 2000,  # 10000
    'early_stopping_rounds': 10,
}

# モデルディスパッチャ

models = {
    'decision_tree_gini': DecisionTreeClassifier(
        criterion='gini',
    ),
    'decision_tree_entropy': DecisionTreeClassifier(
        criterion='entropy',
    ),
    'logistic_regression': LogisticRegression(),
    'random_forest': RandomForestClassifier(),
    'xgboost': ModelXgb(
        params=params_xgb,
    ),
}
