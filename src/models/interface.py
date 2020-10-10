# -*- coding: utf-8 -*-
"""クラス作成の雛形の抽象クラスを記載するモジュール

Notes:
    AbsModel：　学習・予測・モデルの保存・読み込み機能を持ったモデルクラスを作るための抽象クラス

"""
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Optional


class AbsModel(metaclass=ABCMeta):
    """抽象Modelクラス

    学習・予測・モデルの保存・読み込み機能を持ったモデルクラスを作るための抽象クラス
    この抽象モデルを継承して各学習モデルを作成する

    Attributes:
        run_name(str): 実行の名前とfoldの番号を組み合わせた名前
        params(dict): ハイパーパラメータ
        model(AbsModel): 初期値はNoneで、train後にモデルを保持するのに使う

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
    def train(self, train_x, train_y, valid_x, valid_y):
        """モデルの学習を行い、学習済のモデルを保存する

        Args:
            train_x(pd.DataFrame): 学習データの特徴量
            train_y(pd.Series): 学習データの目的変数
            valid_x(Optional[pd.DataFrame]): バリデーションデータの特徴量
            valid_y(Optional[pd.Series]): バリデーションデータの目的変数

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


