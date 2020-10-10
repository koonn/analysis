# -*- coding: utf-8 -*-
"""ランダムフォレストモデルを記載するモジュール

Absモデルを継承したモデルを作成する

"""
import os

import joblib
from sklearn.ensemble import RandomForestClassifier

import config
from models.interface import AbsModel


class ModelRandomForestClassifier(AbsModel):
    """RandomForestのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params):
        """コンストラクタ"""
        super().__init__(params)

    def train(self, train_x, train_y, valid_x=None, valid_y=None):
        """モデルの学習を行う関数"""
        # モデルの構築・訓練
        model = RandomForestClassifier(**self.params)
        model.fit(train_x, train_y)

        # モデルを保持する
        self.model = model

    def predict(self, x):
        """予測確率を算出する関数"""
        pred_proba = self.model.predict_proba(x)

        return pred_proba

    def save_model(self):
        """モデルを保存する関数"""
        model_path = os.path.join(config.MODEL_OUTPUT_DIR, f'{self.run_name}.pkl')
        joblib.dump(self.model, model_path)

    def load_model(self):
        """モデルを作成する関数"""
        pass