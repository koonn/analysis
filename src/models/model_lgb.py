# -*- coding: utf-8 -*-
"""LightGBMモデルを記載するモジュール

Absモデルを継承したモデルを作成する

"""
import os

import joblib
import lightgbm as lgb

import config
from models.interface import AbsModel


class ModelLGB(AbsModel):
    """LightGBMのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params):
        """コンストラクタ"""
        super().__init__(params)

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        """モデルの学習を行う関数"""
        # ハイパーパラメータの設定
        params = dict(self.params)
        num_boost_round = params.pop('num_boost_round')
        early_stopping_rounds = params.pop('early_stopping_rounds')

        # 学習データ・バリデーションデータをlgb.Dataset形式に変換
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_valid = lgb.Dataset(x_valid, y_valid)

        # 学習を実行. watchlistは、学習中に評価指標を計算する対象のデータセット
        self.model = lgb.train(params=params,
                               train_set=lgb_train,
                               num_boost_round=num_boost_round,
                               valid_sets=lgb_valid,
                               early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=1000,  # 何回のイテレーションごとに出力するか
                               )

    def predict(self, x):
        """ラベルが1である予測確率をの算出を行う関数"""
        pred = self.model.predict(x.values, num_iteration=self.model.best_iteration)

        return pred

    def save_model(self):
        """モデルを保存する関数"""
        model_path = os.path.join(config.MODEL_OUTPUT_DIR, f'{self.run_name}.pkl')
        joblib.dump(self.model, model_path)

    def load_model(self):
        """モデルを読み込む関数"""
        model_path = os.path.join(config.MODEL_OUTPUT_DIR, f'{self.run_name}.pkl')
        self.model = joblib.load(model_path)