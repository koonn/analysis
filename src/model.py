import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional


class Model(metaclass=ABCMeta):
    """Modelクラス

    学習・予測・モデルの保存。読み込みなどを行う

    Attributes:
        run_fold_name(str): 実行の名前とfoldの番号を組み合わせた名前
        params(dict): ハイパーパラメータ
        model(None): モデル

    """
    def __init__(self, run_fold_name, params):
        """コンストラクタ

        Args:
            run_fold_name(str): ランの名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ

        Returns:
            None

        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None

    @abstractmethod
    def train(self, train_x, train_y, valid_x, valid_y):
        """モデルの学習を行い、学習済のモデルを保存する

        Args:
            train_x(pd.DataFrame): 学習データの特徴量
            train_y(pd.Series): 学習データの目的変数
            valid_x(Optional[pd.DataFrame]): バリデーションデータの特徴量
            valid_y(Optional[pd.Series]): バリデーションデータの目的変数

        Returns:
            None

        """
        pass

    @abstractmethod
    def predict(self, test_x):
        """学習済のモデルでの予測値を返す

        Args:
            test_x(pd.DataFrame): バリデーションデータやテストデータの特徴量

        Returns:
            np.array: 予測値

        """
        pass

    @abstractmethod
    def save_model(self):
        """モデルの保存を行う

        Returns:
            None

        """
        pass

    @abstractmethod
    def load_model(self):
        """モデルの読み込みを行う

        Returns:
            None

        """
        pass
