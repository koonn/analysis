# -*- coding: utf-8 -*-
"""学習の実行時の設定

実行の設定を記載したモジュール

Attributes:
    TRAINING_FILE(str): もともとの学習データ
    TRAINING_FOLD_FILE(str): 交差検証用のfoldカラムを追加した学習データ
    MODEL_OUTPUT_DIR(str): モデルの格納先ディレクトリ
    TARGET_COLUMN(str): 目的変数のカラム名

"""
from func_util import script_based_path

# ----------------
# モデルの保存先の設定
# ----------------

# モデルの保存先ディレクトリ
MODEL_OUTPUT_DIR = script_based_path('../new_model/')

# ---------
# データの設定
# ---------

# 学習データファイル
# TRAINING_FILE = script_based_path('../data/features/train.csv')
TRAINING_FILE = script_based_path('../data/credit_card_data/train.csv')

# CV用学習データファイル
# TRAINING_FOLD_FILE = script_based_path('../data/features/train_folds.csv')
TRAINING_FOLD_FILE = script_based_path('../data/credit_card_data/train_folds.csv')

# targetカラムの名前
# TARGET_COLUMN = 'target'
TARGET_COLUMN = 'Class'
