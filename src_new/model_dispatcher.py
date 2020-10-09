"""学習するモデルの一覧を記載したモジュール
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