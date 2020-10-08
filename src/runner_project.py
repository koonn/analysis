"""Runner

- KaggleFeaturesRunner
- CreditCardRunner

"""
from typing import List
import pandas as pd
import numpy as np

from .runner import AbsRunner


class KaggleFeaturesRunner(AbsRunner):
    """kaggle用のモデルの学習・予測などを行う

    Attributes:
        run_name(str): ランの名前
        model_class(Callable[[str, dict], AbsModel]): モデルのクラス
        params(dict): ハイパーパラメータ
        n_fold(int): クロスバリデーションの分割数
        file_path_train(str): トレインデータのファイルパス
        file_path_test(str): テストデータのファイルパス
        features(List[str]): 特徴量の名前のリスト
        features_to_scale(list[str]): 特徴量のうち、スケールするものの名前のリスト

    """
    def _load_y_train(self):
        """学習データの目的変数を読み込む

        Returns:
            pd.Series: 学習データの目的変数

        """
        # ファイルを読み込んで目的変数を抽出
        file_path_train = self.run_settings.file_path_train
        train_y = pd.read_csv(file_path_train)['target']

        # 'Class_1'みたいな形式で入っているので、番号部分を抽出する
        # ラベルが1始まりになっているので、1引いて0始まりにする
        train_y = np.array([int(st[-1]) for st in train_y]) - 1
        train_y = pd.Series(train_y)

        return train_y
