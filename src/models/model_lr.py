# -*- coding: utf-8 -*-
"""ロジスティック回帰モデルを記載するモジュール

Absモデルを継承したモデルを作成する

"""
import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config
from models.interface import AbsModel


class ModelLogisticRegression(AbsModel):
    """LogisticRegressionのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.
            scaler(Model): train後に学習済みスケーラーを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params):
        super().__init__(params)
        self.scaler = None

    def train(self, train_x, train_y, valid_x=None, valid_y=None, features_to_scale=None):
        """モデルの学習を行う関数

        Args:
            train_x(pd.DataFrame of [n_samples, n_features]): 学習データの特徴量
            train_y(1-D array-like shape of [n_samples]): 学習データのラベル配列
            valid_x(array-like shape of [n_samples, n_features]): バリデーションデータの特徴量
            valid_y(1-D array-like shape of [n_samples]): バリデーションデータのラベル配列
            features_to_scale(Optional[List[str]]): スケール対象の特徴量を指定する

        """
        # データのスケーリング
        # スケールするカラムを指定
        if features_to_scale is None:
            features_to_scale = train_x.columns

        # スケーラを作成
        scaler = StandardScaler()
        scaler.fit(train_x[features_to_scale])

        # スケーリングを実行
        train_x.loc[:, features_to_scale] = scaler.transform(train_x[features_to_scale])

        # モデルの構築・学習
        model = LogisticRegression(**self.params)
        model = model.fit(train_x, train_y)

        # モデル・スケーラーを保持する
        self.model = model
        self.scaler = scaler

    def predict(self, x):
        """ラベルが1である予測確率を算出する関数"""
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)

        return pred[:, 1]

    def save_model(self):
        """モデルを保存する関数"""
        # パスの設定
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'lr')
        model_path = os.path.join(model_dir, f'{self.run_name}-model_archived.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')

        # 保存先のディレクトリがなければ作成
        os.makedirs(model_dir, exist_ok=True)

        # モデル・スケーラーの保存
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self):
        """モデルを作成する関数"""
        pass