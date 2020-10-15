# -*- coding: utf-8 -*-
"""異常検知系のモデルを記載するモジュール

Absモデルを継承したモデルを作成する
    - One-class SVM
    - LOF
    - IsolationForest

TODO:
    - OCSVM, LOFがまだ動作確認できていない

"""
import os
import joblib

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config
from .interface import AbsModel
from .util import scale_scores


class ModelOCSVM(AbsModel):
    """One-class SVMのモデルクラス

    特徴量を標準化して、One-class SVMで異常度スコアを算出するモデル

    Attributes:
        run_name(str): 実行の名前とfoldの番号を組み合わせた名前
        params(dict): ハイパーパラメータ
        features_to_scale(Optional[List[str]]): スケール対象の特徴量を指定する
        model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.
        scaler(Model): train後に学習済みスケーラーを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params, features_to_scale=None):
        super().__init__(params)
        self.features_to_scale = features_to_scale
        self.scaler = None

    def train(self, train_x, train_y=None, valid_x=None, valid_y=None):
        """モデルの学習を行う関数

        Args:
            train_x(pd.DataFrame of [n_samples, n_features]): 学習データの特徴量
            train_y(1-D array-like shape of [n_samples]): 学習データのラベル配列. 教師なしモデルのためtrain_yは受け取るが使用しない
            valid_x(array-like shape of [n_samples, n_features]): バリデーションデータの特徴量
            valid_y(1-D array-like shape of [n_samples]): バリデーションデータのラベル配列

        Notes:
            教師なしモデルのためtrain_yは受け取るが使用しない
            教師ありモデルと同じtrain.pyで実行できるよう、train_yは引数として受け取っている

        TODO:
            - 少なくともモデル側ではtrain_yを使わないことを明示するため、**kwargsで書き換える(書き換えられるか試す)

        """
        # データのスケーリング
        # スケールするカラムを指定
        if self.features_to_scale is None:
            self.features_to_scale = train_x.columns

        # スケーラを作成
        scaler = StandardScaler()
        scaler.fit(train_x[self.features_to_scale])

        # スケーリングを実行
        train_x.loc[:, self.features_to_scale] = scaler.transform(train_x[self.features_to_scale])

        # モデルの構築・学習
        model = OneClassSVM(**self.params)
        model = model.fit(train_x)

        # モデル・スケーラーを保持する
        self.model = model
        self.scaler = scaler

    def predict(self, x):
        """異常度スコアを算出する関数"""
        # スケールするカラムを指定
        if self.features_to_scale is None:
            self.features_to_scale = x.columns

        # xの前処理の変換
        x.loc[:, self.features_to_scale] = self.scaler.transform(x[self.features_to_scale])

        # 異常度スコアの算出
        scores = self.model.score_samples(x)

        # スコアを0-1で1の方が異常になるように変換
        scaled_scores = scale_scores(scores, is_reversed=True)

        return scaled_scores

    def save_model(self):
        """モデルを保存する関数"""
        # パスの設定
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'ocsvm')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')

        # 保存先のディレクトリがなければ作成
        os.makedirs(model_dir, exist_ok=True)

        # モデル・スケーラーの保存
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self):
        """モデルを読み込む関数"""
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'ocsvm')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)


class ModelLOF(AbsModel):
    """LOFのモデルクラス

    特徴量を標準化して、One-class SVMで異常度スコアを算出するモデル

    Attributes:
        run_name(str): 実行の名前とfoldの番号を組み合わせた名前
        params(dict): ハイパーパラメータ
        features_to_scale(Optional[List[str]]): スケール対象の特徴量を指定する
        model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.
        scaler(Model): train後に学習済みスケーラーを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params, features_to_scale=None):
        super().__init__(params)
        self.features_to_scale = features_to_scale
        self.scaler = None

    def train(self, train_x, train_y=None, valid_x=None, valid_y=None):
        """モデルの学習を行う関数

        LOFでは、事前にモデルを学習するのではなく、fit_transformでそのデータセット内での異常度を計算することになるので、
        trainではスケーラーだけを学習するようにする

        Args:
            train_x(pd.DataFrame of [n_samples, n_features]): 学習データの特徴量
            train_y(1-D array-like shape of [n_samples]): 学習データのラベル配列. 教師なしモデルのためtrain_yは受け取るが使用しない
            valid_x(array-like shape of [n_samples, n_features]): バリデーションデータの特徴量
            valid_y(1-D array-like shape of [n_samples]): バリデーションデータのラベル配列

        Notes:
            教師なしモデルのためtrain_yは受け取るが使用しない
            教師ありモデルと同じtrain.pyで実行できるよう、train_yは引数として受け取っている

        TODO:
            - 少なくともモデル側ではtrain_yを使わないことを明示するため、**kwargsで書き換える(書き換えられるか試す)
            - 今のtrain.pyとの組み合わせだとvalidごとに学習することになっているので、allで学習できるように修正
            - save_modelのモデルの方は意味ないのでどうするか検討する

        """
        # データのスケーリング
        # スケールするカラムを指定
        if self.features_to_scale is None:
            self.features_to_scale = train_x.columns

        # スケーラを作成
        scaler = StandardScaler()
        scaler.fit(train_x[self.features_to_scale])

        # スケーリングを実行
        train_x.loc[:, self.features_to_scale] = scaler.transform(train_x[self.features_to_scale])

        # モデルインスタンスを作成(学習はしない)
        model = LocalOutlierFactor(**self.params)

        # モデル・スケーラーを保持する
        self.model = model
        self.scaler = scaler

    def predict(self, x):
        """異常度スコアを算出する関数"""
        # スケールするカラムを指定
        if self.features_to_scale is None:
            self.features_to_scale = x.columns

        # xの前処理の変換
        x.loc[:, self.features_to_scale] = self.scaler.transform(x[self.features_to_scale])

        # モデルの学習と、異常度スコアの算出
        self.model.fit(x)
        scores = self.model.negative_outlier_factor_

        # スコアを0-1で1の方が異常になるように変換
        scaled_scores = scale_scores(scores, is_reversed=True)

        return scaled_scores

    def save_model(self):
        """モデルを保存する関数"""
        # パスの設定
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'lof')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')

        # 保存先のディレクトリがなければ作成
        os.makedirs(model_dir, exist_ok=True)

        # モデル・スケーラーの保存
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self):
        """モデルを読み込む関数"""
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'lof')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)


class ModelIsolationForest(AbsModel):
    """One-class SVMのモデルクラス

    特徴量をIsolationForestにかける

    """
    def __init__(self, params):
        super().__init__(params)

    def train(self, train_x, train_y=None, valid_x=None, valid_y=None):
        """モデルの学習を行う関数

        Args:
            train_x(pd.DataFrame of [n_samples, n_features]): 学習データの特徴量
            train_y(1-D array-like shape of [n_samples]): 学習データのラベル配列. 教師なしモデルのためtrain_yは受け取るが使用しない
            valid_x(array-like shape of [n_samples, n_features]): バリデーションデータの特徴量
            valid_y(1-D array-like shape of [n_samples]): バリデーションデータのラベル配列

        Notes:
            教師なしモデルのためtrain_yは受け取るが使用しない
            教師ありモデルと同じtrain.pyで実行できるよう、train_yは引数として受け取っている

        TODO:
            - 少なくともモデル側ではtrain_yを使わないことを明示するため、**kwargsで書き換える(書き換えられるか試す)

        """
        # モデルの構築・学習
        model = IsolationForest(**self.params)
        model = model.fit(train_x)

        # モデル・スケーラーを保持する
        self.model = model

    def predict(self, x):
        """異常度スコアを算出する関数"""
        # 異常度スコアの算出
        scores = self.model.score_samples(x)

        # スコアを0-1で1の方が異常になるように変換
        scaled_scores = scale_scores(scores, is_reversed=True)

        return scaled_scores

    def save_model(self):
        """モデルを保存する関数"""
        # パスの設定
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'if')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')

        # 保存先のディレクトリがなければ作成
        os.makedirs(model_dir, exist_ok=True)

        # モデルの保存
        joblib.dump(self.model, model_path)

    def load_model(self):
        """モデルを読み込む関数"""
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'if')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')

        self.model = joblib.load(model_path)
