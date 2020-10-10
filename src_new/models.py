# -*- coding: utf-8 -*-
"""モデルを記載するモジュール

Absモデルを継承したモデルを作成する

"""
import os
import joblib
import xgboost as xgb

from interface import AbsModel


class ModelXgb(AbsModel):
    """XGBoostのモデルクラス

        Attributes:
            run_fold_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.
            scaler(Model): train後に学習済みスケーラーを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def fit(self, train_x, train_y):
        # データの準備
        d_train = xgb.DMatrix(train_x, label=train_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')

        # 学習を実行. watchlistは、学習中に評価指標を計算する対象のデータセット
        watchlist = [(d_train, 'train')]
        self.model = xgb.train(params,
                               d_train,
                               num_round,
                               evals=watchlist,
                               verbose_eval=100,  # 何回のイテレーションごとに出力するか
                               )

    def predict(self, x):
        """予測確率をの算出を行う関数

        Args:
            x(array-like shape of [n_samples, n_features]): 予測をしたい対象の特徴量

        Returns:
            np.array(shape of [n_samples, ]): 予測値

        """
        d_x = xgb.DMatrix(x)

        return self.model.predict(d_x, ntree_limit=self.model.best_ntree_limit)

    def save_model(self):
        """モデルを保存する関数"""
        pass

    def load_model(self):
        pass


if __name__ == '__main__':

    # XGBoostのハイパーパラメータ設定
    params_xgb = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'max_depth': 12,
        'eta': 0.1,
        'min_child_weight': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
        'random_state': 71,
        'num_round': 10000,
        'early_stopping_rounds': 10,
    }

    # インスタンスの作成確認用
    model_xgb = ModelXgb(params_xgb)
