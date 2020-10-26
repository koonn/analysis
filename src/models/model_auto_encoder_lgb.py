# -*- coding: utf-8 -*-
"""AutoEncoder + lightGBM

Absモデルを継承したモデルを作成する

"""
import os
import joblib

import pandas as pd

import lightgbm as lgb

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from keras.layers import BatchNormalization, Input, Lambda
from keras import regularizers
from keras.losses import mse, binary_crossentropy

import config
from models.interface import AbsModel
from models.util import anomary_scores_ae


class ModelAELightGBM(AbsModel):
    """オートエンコーダ+LightGBMのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.

    """

    def __init__(self, params):
        """コンストラクタ"""
        super().__init__(params)

    def train(self, x_train, y_train=None, x_valid=None, y_valid=None):
        """モデルの学習を行う関数"""
        # -------------------
        # オートエンコーダの学習部分
        # -------------------

        num_features = len(x_train.columns)

        # AEモデルの作成
        model_ae = Sequential()
        model_ae.add(Dense(units=70,
                           activation='linear',
                           activity_regularizer=regularizers.l1(10e-5),
                           input_dim=num_features,
                           name='hidden_layer',
                           ))
        model_ae.add(Dropout(0.01))
        model_ae.add(Dense(units=num_features,
                           activation='linear'))

        # 学習の設定
        opt = keras.optimizers.Adam(learning_rate=0.05)
        model_ae.compile(
            optimizer=opt,
            loss='mean_squared_error',
            metrics=['accuracy'],
        )

        # 学習を実行
        num_epochs = 10
        batch_size = 32

        history = model_ae.fit(x=x_train,
                               y=x_train,
                               epochs=num_epochs,
                               batch_size=batch_size,
                               shuffle=True,
                               validation_split=0.2,
                               verbose=1,
                               )

        self.model = model_ae

        # ------------------------------
        # オートエンコーダから特徴量を抽出する部分
        # ------------------------------

        # 隠れ層から特徴量を抽出する
        layer_name = 'hidden_layer'

        intermediate_layer_model = Model(inputs=model_ae.input,
                                         outputs=model_ae.get_layer(layer_name).output,
                                         )

        intermediate_output_train = intermediate_layer_model.predict(x_train)
        intermediate_output_valid = intermediate_layer_model.predict(x_valid)

        intermediate_output_train_df = pd.DataFrame(data=intermediate_output_train,
                                                    index=x_train.index
                                                    )
        intermediate_output_valid_df = pd.DataFrame(data=intermediate_output_valid,
                                                    index=x_valid.index
                                                    )

        # 元の特徴量と隠れ層からの特徴量を合わせたデータフレームを作成
        x_train = x_train.merge(intermediate_output_train_df,
                                left_index=True,
                                right_index=True,
                                )
        x_valid = x_valid.merge(intermediate_output_valid_df,
                                left_index=True,
                                right_index=True,
                                )

        # -----------
        # LightGBM部分
        # -----------
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
        """0ー1の範囲の異常度の算出を行う関数"""
        x_reduced = self.model.predict(x, verbose=1)
        anomary_score_ae = anomary_scores_ae(x, x_reduced)

        return anomary_score_ae

    def save_model(self):
        """モデルを保存する関数"""


    def load_model(self):
        """モデルを読み込む関数"""
        pass