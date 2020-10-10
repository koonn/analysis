# -*- coding: utf-8 -*-
"""ランダムフォレストモデルを記載するモジュール

Absモデルを継承したモデルを作成する

"""
from sklearn.ensemble import RandomForestClassifier
from models.interface import BaseSklearnModel


class ModelRandomForestClassifier(BaseSklearnModel):
    """RandomForestのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.
            _model_class(ModelClass): モデルクラス

    """
    def __init__(self, params):
        """コンストラクタ"""
        super().__init__(params)
        self._model_class = RandomForestClassifier
