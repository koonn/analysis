import os

import xgboost as xgb

from .model import AbsModel
from .util import Util


class ModelXGB(AbsModel):
    """XGBoostのモデルクラス

    XGBoostの学習・予測・モデルの保存・読み込みなどを行うクラス

    """
    def train(self, train_x, train_y, valid_x=None, valid_y=None):
        """モデルの学習を行う関数

        Args:
            train_x(array-like shape of [n_samples, n_features]): 学習データの特徴量
            train_y(1-D array-like shape of [n_samples]): 学習データのラベル配列
            valid_x(array-like shape of [n_samples, n_features]): バリデーションデータの特徴量
            valid_y(1-D array-like shape of [n_samples]): バリデーションデータのラベル配列

        Returns:
            None

        Notes：
            has_validation_dataが、valid_yだけない時にどう振る舞うべきか
        """
        # データの準備

        # validationデータがあるかどうか。valid_xがNoneでないならTrue
        has_validation_data = (valid_x is not None)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')

        # 学習データをDMatrix形式に変換
        d_train = xgb.DMatrix(train_x, label=train_y)

        if has_validation_data:
            # バリデーションデータがあれば、バリデーションデータもDMatrix形式に変換
            d_valid = xgb.DMatrix(valid_x, label=valid_y)

            # バリデーション時のみのハイパーパラメータの設定
            early_stopping_rounds = params.pop('early_stopping_rounds')

            # 学習を実行. watchlistは、学習中に評価指標を計算する対象のデータセット
            watchlist = [(d_train, 'train'), (d_valid, 'eval')]
            self.model = xgb.train(params,
                                   d_train,
                                   num_round,
                                   evals=watchlist,
                                   early_stopping_rounds=early_stopping_rounds
                                   )

        else:
            # 学習を実行. watchlistは、学習中に評価指標を計算する対象のデータセット
            watchlist = [(d_train, 'train')]
            self.model = xgb.train(params,
                                   d_train,
                                   num_round,
                                   evals=watchlist
                                   )

    def predict(self, array_x):
        """予測確率を行う関数

        Args:
            array_x(array-like shape of [n_samples, n_features]): 予測をしたい対象の特徴量

        Returns:
            np.array(shape of [n_samples, ]): 予測値

        """
        d_x = xgb.DMatrix(array_x)
        return self.model.predict(d_x, ntree_limit=self.model.best_ntree_limit)

    def save_model(self):
        """モデルを保存する関数

        Returns:
            None

        """
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # best_ntree_limitが消えるのを防ぐため、pickleで保存することとした
        Util.dump(self.model, model_path)

    def load_model(self):
        """モデルを読み込む関数

        Returns:
            None

        """
        model_path = os.path.join('../model/model', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)
