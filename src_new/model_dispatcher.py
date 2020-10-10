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

models = {
    'decision_tree_gini': DecisionTreeClassifier(
        criterion='gini',
    ),
    'decision_tree_entropy': DecisionTreeClassifier(
        criterion='entropy',
    ),
    'logistic_regression': LogisticRegression(),
    'random_forest': RandomForestClassifier(),
}