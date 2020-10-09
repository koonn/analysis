"""学習の実行時の設定を記載するモジュール"""
from func_util import script_based_path


# 学習データファイル
TRAINING_FILE = script_based_path('../data/features/train.csv')

# CV用学習データファイル
TRAINING_FOLD_FILE = script_based_path('../data/features/train_folds.csv')

# モデルの保存先ディレクトリ
MODEL_OUTPUT_DIR = script_based_path('../new_model/')
