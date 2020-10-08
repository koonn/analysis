"""学習の実行時の設定を記載するモジュール"""
from func_util import script_based_path

TRAINING_FILE = script_based_path('../data/features/train_folds.csv')

MODEL_OUTPUT_DIR = script_based_path('../new_model/')
