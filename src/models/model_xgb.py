# -*- coding: utf-8 -*-
"""XGBoostモデルを記載するモジュール

Absモデルを継承したモデルを作成する

"""
import os

import joblib
import xgboost as xgb

import config
from models.interface import AbsModel


class ModelXgb(AbsModel):
    """XGBoostのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params):
        """コンストラクタ"""
        super().__init__(params)

    def train(self, train_x, train_y, valid_x=None, valid_y=None):
        """モデルの学習を行う関数"""
        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')
        early_stopping_rounds = params.pop('early_stopping_rounds')

        # 学習データ・バリデーションデータをDMatrix形式に変換
        d_train = xgb.DMatrix(train_x, label=train_y)
        d_valid = xgb.DMatrix(valid_x, label=valid_y)

        # 学習を実行. watchlistは、学習中に評価指標を計算する対象のデータセット
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]
        self.model = xgb.train(params=params,
                               dtrain=d_train,
                               num_boost_round=num_round,
                               evals=watchlist,
                               early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=100,  # 何回のイテレーションごとに出力するか
                               )

    def predict(self, x):
        """ラベルが1である予測確率をの算出を行う関数"""
        d_x = xgb.DMatrix(x)
        pred = self.model.predict(d_x, ntree_limit=self.model.best_ntree_limit)

        return pred[:, 1]

    def save_model(self):
        """モデルを保存する関数"""
        model_path = os.path.join(config.MODEL_OUTPUT_DIR, f'{self.run_name}.pkl')
        joblib.dump(self.model, model_path)

    def load_model(self):
        """モデルを読み込む関数"""
        model_path = os.path.join(config.MODEL_OUTPUT_DIR, f'{self.run_name}.pkl')
        self.model = joblib.load(model_path)