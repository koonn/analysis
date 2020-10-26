# -*- coding: utf-8 -*-
"""SVMモデルを記載するモジュール

を継承したSVMモデルを作成する

TODO:
    - BaseScaledSklearnModelみたいなクラスを作成する(コードを共通化出来そうなため)
    - 死ぬほど遅くなるのと精度が全然出てないのでハイパラ調整(特にmax_iter)

"""
# util
import os
import joblib

# モデル
from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler

import config
from .interface import BaseSklearnModel


class ModelSVM(BaseSklearnModel):
    """SVMのモデルクラス

    特徴量を標準化した上で、C-supported SVMにかけるモデル

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            features_to_scale(Optional[List[str]]): スケール対象の特徴量を指定する
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.
            scaler(Model): train後に学習済みスケーラーを保持. trainを実行するまでは、初期値のNoneをかえす.
            kernel_mapper(Model): train後に学習済みkernelマッパーを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params, features_to_scale=None):
        super().__init__(params)
        self.features_to_scale = features_to_scale
        self.scaler = None
        self.kernel_mapper = None

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

        # 特徴量のサブサンプルでのカーネル変換(featureが多いため、普通にSVMやると遅すぎる)
        kernel_mapper = Nystroem(gamma=.2,
                                 random_state=config.RANDOM_SEED,
                                 n_components=300
                                 )
        train_x_mapped = kernel_mapper.fit_transform(train_x)

        # モデルの構築・学習
        model = SVC(**self.params)  # probability=Trueじゃないと確率を返さずpredictメソッドが使えないため、常にTrueにする
        model = model.fit(train_x_mapped, train_y)

        # モデル・スケーラーを保持する
        self.model = model
        self.kernel_mapper = kernel_mapper
        self.scaler = scaler

    def predict(self, x):
        """ラベルが1である予測確率を算出する関数"""
        # スケールするカラムを指定
        if self.features_to_scale is None:
            self.features_to_scale = x.columns

        # xの前処理の変換
        x.loc[:, self.features_to_scale] = self.scaler.transform(x[self.features_to_scale])
        x_mapped = self.kernel_mapper.transform(x)

        # 予測確率を算出
        pred = self.model.predict_proba(x_mapped)

        return pred[:, 1]

    def save_model(self):
        """モデルを保存する関数"""
        # パスの設定
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'svm')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')
        kernel_mapper_path = os.path.join(model_dir, f'{self.run_name}-kernel_mapper.pkl')

        # 保存先のディレクトリがなければ作成
        os.makedirs(model_dir, exist_ok=True)

        # モデル・スケーラーの保存
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.kernel_mapper, kernel_mapper_path)

    def load_model(self):
        """モデルを読み込む関数"""
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'svm')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')
        kernel_mapper_path = os.path.join(model_dir, f'{self.run_name}-kernel_mapper.pkl')

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.kernel_mapper = joblib.load(kernel_mapper_path)
