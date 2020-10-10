"""Runnerの設定を記述したモジュール

run_settingsの引数として使う

以下の要素を含める:
    - file_path_train(str): トレインデータのファイルパス
    - file_path_test(str): テストデータのファイルパス
    - model_dir(str): モデルの保存先ディレクトリ
    - features(List[str]): 特徴量の名前のリスト
    - features_to_scale(list[str]): 特徴量のうち、スケールするものの名前のリスト

Examples:
    # Kaggleの本のはこれ
    run_settings_kagglebook = RunSettings(
        file_path_train=Util.script_based_path('../data/features/train.csv'),
        file_path_test=Util.script_based_path('../data/features/test.csv'),
        model_dir=Util.script_based_path('../model_archived/model_archived/lr/'),
        target=['target'],
        features=[f'feat_{i}' for i in range(1, 94)],
        features_to_scale=[f'feat_{i}' for i in range(1, 94)],
        )

    # Handson本のはこれ
    run_settings_handson = RunSettings(
        file_path_train=Util.script_based_path('../data/features/train.csv'),
        file_path_test=Util.script_based_path('../data/features/test.csv'),
        model_dir=Util.script_based_path('../model_archived/model_archived/lr/'),
        target=['Class'],
        features=[f'feat_{i}' for i in range(1, 94)],
        features_to_scale=[f'feat_{i}' for i in range(1, 94)],
        )
"""
from typing import NamedTuple, List


class RunSettings(NamedTuple):
    """Runnerのデータ取得のセッティング

    run_settingsの引数として使う

    Notes:
        file_path_train(str): トレインデータのファイルパス
        file_path_test(str): テストデータのファイルパス
        model_dir(str): モデルの保存先ディレクトリ
        target(List[str]):  目的変数の名前のリスト
        features(List[str]): 特徴量の名前のリスト
        features_to_scale(list[str]): 特徴量のうち、スケールするものの名前のリスト

    """
    file_path_train: str  # トレインデータのファイルパス
    file_path_test: str  # テストデータのファイルパス
    model_dir: str  # モデルの保存先ディレクトリ
    target: List  # 目的変数の名前のリスト
    features: List  # 特徴量の名前のリスト
    features_to_scale: List  # 特徴量のうち、スケールするものの名前のリスト
