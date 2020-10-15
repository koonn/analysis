# -*- coding: utf-8 -*-
"""AutoEncoder

Absモデルを継承したモデルを作成する

"""
import os
import joblib

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


class ModelAE(AbsModel):
    """オートエンコーダのモデルクラス

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
        # ハイパーパラメータの設定

        num_features = len(x_train.columns)

        # モデルの作成
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

    def predict(self, x):
        """0ー1の範囲の異常度の算出を行う関数"""
        x_reduced = self.model.predict(x, verbose=1)
        anomary_score_ae = anomary_scores_ae(x, x_reduced)

        return anomary_score_ae

    def save_model(self):
        """モデルを保存する関数"""
        pass

    def load_model(self):
        """モデルを読み込む関数"""
        pass
