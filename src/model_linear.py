import os
from typing import List, Optional

from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as pp

from .model import AbsModel
from .util import Util


class ModelLogisticRegression(AbsModel):
    """LogisticRegressionのモデルクラス

        Attributes:
            run_fold_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.
            scaler(Model): train後に学習済みスケーラーを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, run_fold_name, params):
        super().__init__(run_fold_name, params)
        self.scaler = None

    def train(self, train_x, train_y, valid_x=None, valid_y=None, features_to_scale=None):
        """モデルの学習を行う関数

        Args:
            train_x(array-like shape of [n_samples, n_features]): 学習データの特徴量
            train_y(1-D array-like shape of [n_samples]): 学習データのラベル配列
            valid_x(array-like shape of [n_samples, n_features]): バリデーションデータの特徴量
            valid_y(1-D array-like shape of [n_samples]): バリデーションデータのラベル配列
            features_to_scale(Optional[List[str]]): スケール対象の特徴量を指定する

        Returns:
            None

        """
        # validationデータがあるかどうか。valid_xがNoneでないならTrue
        do_validation = (valid_x is not None)

        # データのスケーリング

        # スケールするカラムを指定

        if features_to_scale is None:
            features_to_scale = train_x.columns

        # スケーラを作成
        scaler = pp.StandardScaler()
        scaler.fit(train_x[features_to_scale])

        # スケーリングを実行
        train_x.loc[:, features_to_scale] = scaler.transform(train_x[features_to_scale])

        # モデルの構築・学習
        model = LogisticRegression(**self.params)
        model = model.fit(train_x, train_y)

        # モデル・スケーラーを保持する
        self.model = model
        self.scaler = scaler

    def predict(self, array_x):
        """予測確率を算出する関数

        Args:
            array_x(array-like shape of [n_samples, n_features]): 予測をしたい対象の特徴量

        Returns:
            np.array(shape of [n_samples, ]): 予測値

        """
        array_x = self.scaler.transform(array_x)
        pred = self.model.predict_proba(array_x)
        return pred

    def save_model(self, model_dir):
        """モデルを保存する関数

        Returns:
            None

        Todos:
            - pathの設定を自分で設定できるようにする
            - かつ、インスタンス変数で持つなどするか検討
        """
        # モデルの保存先の指定と、ディレクトリの準備
        model_dir = Util.script_based_path('../model/model/lr')

        model_path = os.path.join(model_dir, f'{self.run_fold_name}.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_fold_name}-scaler.pkl')

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # モデル・スケーラーを保存
        Util.dump(self.model, model_path)
        Util.dump(self.scaler, scaler_path)

    def load_model(self, model_dir):
        model_dir = Util.script_based_path('../model/model/lr')

        model_path = os.path.join(model_dir, f'{self.run_fold_name}.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_fold_name}-scaler.pkl')

        self.model = Util.load(model_path)
        self.scaler = Util.load(scaler_path)









