# -*- coding: utf-8 -*-
"""ナイーブベイズモデルを記載するモジュール

Absモデルを継承したロジスティック回帰モデルを作成する

TODO:
    - GaussianNBに対して、BaseScaledSklearnModelみたいなクラスを作成する(コードを共通化出来そうなため)
    - ModelMixedNBクラスを実装する

"""
# util
import os
import joblib

# モデル
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.preprocessing import StandardScaler

# 設定
import config
from .interface import AbsModel, BaseSklearnModel


class ModelBernoulliNB(BaseSklearnModel):
    """Bernoulli Naive Bayesのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.
            _model_class(ModelClass): モデルクラス

        Notes:
            ベルヌーイ分布を使っているので、2値変数の特徴量向け

    """
    def __init__(self, params):
        """コンストラクタ"""
        super().__init__(params)
        self._model_class = BernoulliNB


class ModelGaussianNB(AbsModel):
    """Gaussian Naive Bayesのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.
            _model_class(ModelClass): モデルクラス

        Notes:
            正規分布を使っているので、連続変数の特徴量向け

    """
    def __init__(self, params, features_to_scale=None):
        super().__init__(params)
        self.features_to_scale = features_to_scale
        self.scaler = None

    def train(self, train_x, train_y, valid_x=None, valid_y=None):
        """モデルの学習を行う関数

        Args:
            train_x(pd.DataFrame of [n_samples, n_features]): 学習データの特徴量
            train_y(1-D array-like shape of [n_samples]): 学習データのラベル配列
            valid_x(array-like shape of [n_samples, n_features]): バリデーションデータの特徴量
            valid_y(1-D array-like shape of [n_samples]): バリデーションデータのラベル配列

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
        model = GaussianNB(**self.params)
        model = model.fit(train_x, train_y)

        # モデル・スケーラーを保持する
        self.model = model
        self.scaler = scaler

    def predict(self, x):
        """ラベルが1である予測確率を算出する関数"""
        # スケールするカラムを指定
        if self.features_to_scale is None:
            self.features_to_scale = x.columns

        # xの前処理の変換
        x.loc[:, self.features_to_scale] = self.scaler.transform(x[self.features_to_scale])

        # 予測確率の算出
        pred = self.model.predict_proba(x)

        return pred[:, 1]

    def save_model(self):
        """モデルを保存する関数"""
        # パスの設定
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'gaussianNB')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')

        # 保存先のディレクトリがなければ作成
        os.makedirs(model_dir, exist_ok=True)

        # モデル・スケーラーの保存
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self):
        """モデルを読み込む関数"""
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'gaussianNB')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)


class ModelMixedNB(AbsModel):
    """Naive Bayesのモデルクラス

    特徴量を標準化した上で、ナイーブベイズにかけるモデル
    2値変数はBernoulliNBで確率を予測、連続変数はGaussianNBで確率を予測して、あとで合成するようなモデル

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            features_to_scale(Optional[List[str]]): スケール対象の特徴量を指定する
            model_gaussian(Model): ガウシアン型ナイーブベイズのモデル(連続変数用)
            model_bernoulli(Model): ベルヌーイ型ナイーブベイズのモデル(二値変数用)
            scaler(Model): train後に学習済みスケーラーを保持. trainを実行するまでは、初期値のNoneをかえす.

        Notes:
            model属性は存在しているが、今回は使わないので呼び出してもNoneを返す

        TODO:
            - features_to_scaleが直接的な名前じゃないので修正を検討する
            - PR曲線を見てみる

    """
    def __init__(self, params, features_to_scale=None):
        super().__init__(params)
        self.features_to_scale = features_to_scale
        self.model_gaussian = None
        self.model_bernoulli = None
        self.scaler = None

    def train(self, train_x, train_y, valid_x=None, valid_y=None):
        """モデルの学習を行う関数

        Args:
            train_x(pd.DataFrame of [n_samples, n_features]): 学習データの特徴量
            train_y(1-D array-like shape of [n_samples]): 学習データのラベル配列
            valid_x(array-like shape of [n_samples, n_features]): バリデーションデータの特徴量
            valid_y(1-D array-like shape of [n_samples]): バリデーションデータのラベル配列

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

        # 特徴量を連続変数と2値変数とに分ける
        train_x_continuous = train_x.loc[:, self.features_to_scale]
        train_x_binary = train_x.drop(self.features_to_scale, axis=1)

        # モデルの構築・学習

        # 連続変数用のモデルの構築・学習
        model_gaussian = GaussianNB(**self.params)
        model_gaussian.fit(train_x_continuous, train_y)

        # 2値変数用のモデルの構築・学習
        model_bernoulli = BernoulliNB(**self.params)
        model_bernoulli.fit(train_x_binary, train_y)

        # モデル・スケーラーを保持する
        self.model_gaussian = model_gaussian
        self.model_bernoulli = model_bernoulli
        self.scaler = scaler

    def predict(self, x):
        """ラベルが1である予測確率を算出する関数"""
        # スケールするカラムを指定
        if self.features_to_scale is None:
            self.features_to_scale = x.columns

        # xの前処理の変換
        x.loc[:, self.features_to_scale] = self.scaler.transform(x[self.features_to_scale])

        # 特徴量を連続変数と2値変数とに分ける
        x_continuous = x.loc[:, self.features_to_scale]
        x_binary = x.drop(self.features_to_scale, axis=1)

        # 予測確率の算出
        pred_proba_continuous = self.model_gaussian.predict_proba(x_continuous)
        pred_proba_binary = self.model_bernoulli.predict_proba(x_binary)

        pred = pred_proba_continuous * pred_proba_binary

        return pred[:, 1]

    def save_model(self):
        """モデルを保存する関数"""
        # パスの設定
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'mixedNB')
        model_gaussian_path = os.path.join(model_dir, f'{self.run_name}-model_gaussian.pkl')
        model_bernoulli_path = os.path.join(model_dir, f'{self.run_name}-model_bernoulli.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')

        # 保存先のディレクトリがなければ作成
        os.makedirs(model_dir, exist_ok=True)

        # モデル・スケーラーの保存
        joblib.dump(self.model, model_gaussian_path)
        joblib.dump(self.model, model_bernoulli_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self):
        """モデルを読み込む関数"""
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'gaussianNB')
        model_gaussian_path = os.path.join(model_dir, f'{self.run_name}-model_gaussian.pkl')
        model_bernoulli_path = os.path.join(model_dir, f'{self.run_name}-model_bernoulli.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')

        self.model = joblib.load(model_gaussian_path)
        self.model = joblib.load(model_bernoulli_path)
        self.scaler = joblib.load(scaler_path)
