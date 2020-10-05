"""Runner

- KaggleFeaturesRunner
- CreditCardRunner

"""
from typing import NamedTuple, List

import pandas as pd
import numpy as np

from .util import Util
from .runner import AbsRunner


class KaggleFeaturesRunner(AbsRunner):
    """モデルの学習・予測などを行う

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

    def __init__(self, run_name, model_class, params, run_settings, n_fold=4):
        """コンストラクタ

        Args:
            run_name(str): ランの名前
            model_class(Callable[[str, dict], AbsModel]): モデルのクラス
            params(dict): ハイパーパラメータ
            n_fold(int): クロスバリデーションの分割数
            run_settings(namedtuple): データ取得のセッティング

        """
        self.run_name = run_name
        self.model_class = model_class
        self.params = params
        self.n_fold = n_fold

        # run_settingsからの変数をバラして入れる
        self.run_settings = run_settings
        self.file_path_train = run_settings.file_path_train
        self.file_path_test = run_settings.file_path_test
        self.features = run_settings.features
        self.features_to_scale = run_settings.features_to_scale

    def train_fold(self, i_fold):
        """クロスバリデーションの指定した1つのfoldに対して学習・評価を行う

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        Args:
            i_fold(Union[int, str]): n_foldで指定した分割のうち、何番目の分割をバリデーション用に使うか（バリデーションしない時には'all'とする）

        Returns:
            Tuple[AbsModel, Optional[np.array], Optional[np.array], Optional[float]]:
                （モデルのインスタンス、バリデーションデータのインデックス、バリデーションデータの予測値、評価関数によるスコア）のタプル
        """
        return super().train_fold(i_fold)

    def run_train_cv(self):
        """クロスバリデーションでの学習・評価を行う

        Returns:
            None

        Notes:
            学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う

        """
        return super().run_train_cv()

    def run_predict_cv(self):
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        Returns:
            None

        Notes:
            あらかじめrun_train_cvを実行しておく必要がある
        """
        super().run_predict_cv()

    def run_train_all(self):
        """学習データすべてで学習し、そのモデルを保存する

        Returns:
            None

        """
        super().run_train_all()

    def run_predict_all(self):
        """学習データすべてで学習したモデルにより、テストデータの予測を行う

        Returns:
            None

        Notes:
            あらかじめrun_train_allを実行しておく必要がある

        """
        super().run_predict_all()

    def build_model(self, i_fold):
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        Args:
            i_fold(Union[int, str]): foldの番号または、'all'

        Returns:
            AbsModel: モデルのインスタンス

        """
        return super().build_model(i_fold)

    def load_x_train(self):
        """学習データの特徴量を読み込む

        Returns:
            pd.DataFrame: 学習データの特徴量

        Notes:
            学習データの読込を行う
            列名で抽出する以上のことを行う場合、このメソッドの修正が必要
            毎回train.csvを読み込むのは効率が悪いため、データに応じて適宜対応するのが望ましい（他メソッドも同様）

        """
        return super().load_x_train()

    def load_y_train(self):
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

    def load_x_test(self):
        """テストデータの特徴量を読み込む

        Returns:
            pd.DataFrame: テストデータの特徴量

        """
        # 読み込みファイルを、実行ファイルからの相対パスで指定(スクリプトの実行場所によらず読み込めるようにするため)
        file_path_test = self.run_settings.file_path_test

        return pd.read_csv(file_path_test)[self.features]

    def load_index_fold(self, i_fold):
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        学習データ・バリデーションデータを分けるインデックスを返す
        ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある

        Args:
            i_fold(int): foldの番号

        Returns:
            np.array: foldに対応するレコードのインデックス

        """
        return super().load_index_fold(i_fold)


class KaggleFeaturesRunner(AbsRunner):
    """kaggle用のモデルの学習・予測などを行う"""

    def train_fold(self, i_fold):
        """クロスバリデーションの指定した1つのfoldに対して学習・評価を行う"""
        return super().train_fold(i_fold)

    def run_train_cv(self):
        """クロスバリデーションでの学習・評価を行う

        Returns:
            None

        Notes:
            学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う

        """
        return super().run_train_cv()

    def run_predict_cv(self):
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        Returns:
            None

        Notes:
            あらかじめrun_train_cvを実行しておく必要がある
        """
        super().run_predict_cv()

    def run_train_all(self):
        """学習データすべてで学習し、そのモデルを保存する

        Returns:
            None

        """
        super().run_train_all()

    def run_predict_all(self):
        """学習データすべてで学習したモデルにより、テストデータの予測を行う

        Returns:
            None

        Notes:
            あらかじめrun_train_allを実行しておく必要がある

        """
        super().run_predict_all()

    def build_model(self, i_fold):
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        Args:
            i_fold(Union[int, str]): foldの番号または、'all'

        Returns:
            AbsModel: モデルのインスタンス

        """
        return super().build_model(i_fold)

    def load_x_train(self):
        """学習データの特徴量を読み込む

        Returns:
            pd.DataFrame: 学習データの特徴量

        Notes:
            学習データの読込を行う
            列名で抽出する以上のことを行う場合、このメソッドの修正が必要
            毎回train.csvを読み込むのは効率が悪いため、データに応じて適宜対応するのが望ましい（他メソッドも同様）

        """
        return super().load_x_train()

    def load_y_train(self):
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

    def load_x_test(self):
        """テストデータの特徴量を読み込む

        Returns:
            pd.DataFrame: テストデータの特徴量

        """
        # 読み込みファイルを、実行ファイルからの相対パスで指定(スクリプトの実行場所によらず読み込めるようにするため)
        file_path_test = self.run_settings.file_path_test

        return pd.read_csv(file_path_test)[self.features]

    def load_index_fold(self, i_fold):
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        学習データ・バリデーションデータを分けるインデックスを返す
        ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある

        Args:
            i_fold(int): foldの番号

        Returns:
            np.array: foldに対応するレコードのインデックス

        """
        return super().load_index_fold(i_fold)
