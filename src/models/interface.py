# -*- coding: utf-8 -*-
"""クラス作成の雛形の抽象クラスを記載するモジュール

Notes:
    AbsModel：　学習・予測・モデルの保存・読み込み機能を持ったモデルクラスを作るための抽象クラス

"""
import os
import joblib
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Optional

import config


class AbsModel(metaclass=ABCMeta):
    """抽象Modelクラス

    学習・予測・モデルの保存・読み込み機能を持ったモデルクラスを作るための抽象クラス
    この抽象モデルを継承して各学習モデルを作成する

    Attributes:
        run_name(str): 実行の名前とfoldの番号を組み合わせた名前
        params(dict): ハイパーパラメータ
        model(Model): 初期値はNoneで、train後にモデルを保持するのに使う

    """
    def __init__(self, params):
        """コンストラクタ

        Args:
            # run_name(str): 実行モデル名とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ

        """
        self.params = params
        self.run_name = None
        self.model = None

    @abstractmethod
    def train(self, x_train, y_train, x_valid, y_valid):
        """モデルの学習を行い、学習済のモデルを保存する

        Args:
            x_train(pd.DataFrame): 学習データの特徴量
            y_train(pd.Series): 学習データの目的変数
            x_valid(Optional[pd.DataFrame]): バリデーションデータの特徴量
            y_valid(Optional[pd.Series]): バリデーションデータの目的変数

        """
        pass

    @abstractmethod
    def predict(self, x):
        """学習済のモデルでの予測値を返す

        Args:
            x(pd.DataFrame): バリデーションデータやテストデータの特徴量

        Returns:
            np.array: 予測値

        Notes:
            予測値には、0〜1の予測確率を返すこと

        """
        pass

    @abstractmethod
    def save_model(self):
        """モデルの保存を行う"""
        pass

    @abstractmethod
    def load_model(self):
        """モデルの読み込みを行う"""
        pass


class BaseSklearnModel(AbsModel):
    """Sklearnのモデルを使ったときのBaseModelクラス

    学習・予測・モデルの保存・読み込み機能を持ったモデルクラスを作るためのベースクラス
    Sklearnの分類クラス単体を使うのであればこれを継承すれば楽

    Attributes:
        run_name(str): 実行の名前とfoldの番号を組み合わせた名前
        params(dict): ハイパーパラメータ
        model(Model): モデルのインスタンス
        _model_class(ModelClass): モデルクラス

    """
    def __init__(self, params):
        """コンストラクタ"""
        super().__init__(params)
        self._model_class = None  # モデルインスタンスとは別なのでmodelとは別にしている

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        """モデルの学習を行う関数"""
        # モデルの構築・訓練
        model = self._model_class(**self.params)
        model.fit(x_train, y_train)

        # モデルを保持する
        self.model = model

    def predict(self, x):
        """ラベルが1である予測確率を算出する関数"""
        pred_proba = self.model.predict_proba(x)

        return pred_proba[:, 1]

    def save_model(self):
        """モデルを保存する関数"""
        model_path = os.path.join(config.MODEL_OUTPUT_DIR, f'{self.run_name}.pkl')
        joblib.dump(self.model, model_path)

    def load_model(self):
        """モデルを読み込む関数"""
        model_path = os.path.join(config.MODEL_OUTPUT_DIR, f'{self.run_name}.pkl')
        self.model = joblib.load(model_path)