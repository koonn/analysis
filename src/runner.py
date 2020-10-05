"""Runner

"""
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from .model import AbsModel
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from typing import List, Optional, Tuple, Union

from .util import Logger, Util

logger = Logger()


class AbsRunner(metaclass=ABCMeta):
    """モデルの学習・予測などを行う

    Attributes:
        run_name(str): ランの名前
        model_class(Callable[[str, dict], AbsModel]): モデルのクラス
        params(dict): ハイパーパラメータ
        n_fold(int): クロスバリデーションの分割数
        file_path_train(str): トレインデータのファイルパス
        file_path_test(str): テストデータのファイルパス
        model_dir(str): モデルの保存先ディレクトリ
        target(List[str]):  目的変数の名前のリスト
        features(List[str]): 特徴量の名前のリスト
        features_to_scale(list[str]): 特徴量のうち、スケールするものの名前のリスト

    """
    def __init__(self, run_name, model_class, params, run_settings, n_fold=4):
        """コンストラクタ

        Args:
            run_name(str): ランの名前
            model_class(Callable[[str, dict], AbsModel]): モデルのクラス
            params(dict): ハイパーパラメータ
            run_settings(namedtuple): データ取得のセッティング
            n_fold(int): クロスバリデーションの分割数. デフォルトは4

        """
        self.run_name = run_name
        self.model_class = model_class
        self.params = params
        self.n_fold = n_fold

        # run_settingsからの変数をバラして入れる
        self.run_settings = run_settings
        self.file_path_train = run_settings.file_path_train
        self.file_path_test = run_settings.file_path_test
        self.model_dir = run_settings.model_dir
        self.target = run_settings.target
        self.features = run_settings.features
        self.features_to_scale = run_settings.features_to_scale

    # ----------------
    # データの読み込み関連
    # ----------------
    def load_x_train(self):
        """学習データの特徴量を読み込む

        Returns:
            pd.DataFrame: 学習データの特徴量

        Notes:
            学習データの読込を行う
            列名で抽出する以上のことを行う場合、このメソッドの修正が必要
            毎回train.csvを読み込むのは効率が悪いため、データに応じて適宜対応するのが望ましい（他メソッドも同様）

        """
        return pd.read_csv(self.file_path_train)[self.features]

    def load_y_train(self):
        """学習データの目的変数を読み込む

        Returns:
            pd.Series: 学習データの目的変数

        Notes:
            csvを読み込んでtargetカラムを取り出す以上のことをしたい場合は書き換える
        """
        return pd.read_csv(self.file_path_train)[self.target]

    def load_x_test(self):
        """テストデータの特徴量を読み込む

        Returns:
            pd.DataFrame: テストデータの特徴量

        Notes:
            csvを読み込んでfeatureカラムを取り出す以上のことをしたい場合は書き換える
        """
        return pd.read_csv(self.file_path_test)[self.features]

    def load_index_fold(self, i_fold):
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        学習データ・バリデーションデータを分けるインデックスを返す
        ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある

        Args:
            i_fold(int): foldの番号

        Returns:
            Tuple[np.array, np.array]: foldに対応するレコードのインデックス

        """
        train_y = self.load_y_train()
        dummy_x = np.zeros(len(train_y))

        # k分割交差検証用に分割するためのインスタンス作成
        s_kfold = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=71)

        return list(s_kfold.split(dummy_x, train_y))[i_fold]

    # ----------
    # 学習・予測関連
    # ----------
    def train_fold(self, i_fold):
        """クロスバリデーションの単一のfoldに対して学習・評価を行う関数

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        Args:
            i_fold(Union[int, str]): n_foldで指定した分割のうち、何番目の分割をバリデーション用に使うか（バリデーションしない時には'all'とする）

        Returns:
            Tuple[AbsModel, Optional[np.array], Optional[np.array], Optional[float]]:
                （モデルのインスタンス、バリデーションデータのインデックス、バリデーションデータの予測値、評価関数によるスコア）のタプル

        Todos:
            - バリデーションしない時には'all'とするのはなんか変な気がするので、Noneとかに直す
            - 評価関数も選択できるようにする
        """
        # 全学習データの読込
        data_x = self.load_x_train()
        data_y = self.load_y_train()

        # バリデーションするかどうか
        do_validation = (i_fold != 'all')

        if do_validation:
            # 全学習をさらに学習データ・バリデーションデータに分割する
            train_index, valid_index = self.load_index_fold(i_fold)  # 学習/バリデーションの分割用インデックスを取得
            train_y, train_x = data_y.iloc[train_index].copy(), data_x.iloc[train_index].copy()  # 学習データを取り出す
            valid_y, valid_x = data_y.iloc[valid_index].copy(), data_x.iloc[valid_index].copy()  # バリデーションデータを取り出す

            # モデルインスタンスを準備
            model = self.build_model(i_fold)

            # モデルの訓練
            model.train(train_x, train_y, valid_x, valid_y)

            # バリデーションデータへの予測
            valid_pred = model.predict(valid_x)

            # バリデーションデータの対数損失を計算
            score = log_loss(valid_y, valid_pred, eps=1e-15, normalize=True)

            # (モデルのインスタンス、バリデーションデータのインデックス、バリデーションデータの予測値、評価関数による対数損失スコア)のタプルを返す
            return model, valid_index, valid_pred, score

        else:
            # モデルインスタンスを準備
            model = self.build_model(i_fold)

            # 全学習データでモデルの訓練
            model.train(data_x, data_y)

            # (モデルのインスタンス、None、None、None)のタプルを返す
            return model, None, None, None

    def run_train_cv(self):
        """クロスバリデーションでの学習・評価を行う

        Returns:
            None

        Notes:
            学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う

        """
        # CVの学習開始時にログを出力
        logger.info(f'{self.run_name} - start training cv')

        cv_scores = []
        valid_indexes = []
        predictions_based_on_kfolds = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 各foldの学習開始時にログを出力
            logger.info(f'{self.run_name} fold {i_fold} - start training')

            # 単一のfoldでの学習を行う
            model, valid_index, valid_pred, score = self.train_fold(i_fold)

            # 各foldの学習終了時にログを出力
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model(self.model_dir)

            # 結果を保持する
            valid_indexes.append(valid_index)  # バリデーションのデータのインデックス
            cv_scores.append(score)  # 対数損失スコア
            predictions_based_on_kfolds.append(valid_pred)  # バリデーションに対する予測値

        # 各foldの結果をまとめる
        valid_indexes = np.concatenate(valid_indexes)  # 全てのindexが入ったnp.arrayを作成
        order = np.argsort(valid_indexes)  # valid_indexをソートするためのindexが入ったnp.arrayを作成
        predictions_based_on_kfolds = np.concatenate(predictions_based_on_kfolds, axis=0) # バリデーション予測結果が入ったnp.arrayを作成
        predictions_based_on_kfolds = predictions_based_on_kfolds[order]  # もともとのindexの昇順に、バリデーション予測結果を並べる

        # CVの学習完了時にログを出力
        logger.info(f'{self.run_name} - end training cv - score {np.mean(cv_scores)}')

        # 予測結果の保存
        Util.dump(predictions_based_on_kfolds,
                  Util.script_based_path(f'../model/pred/{self.run_name}-train.pkl')
                  )

        # 結果まとめをログに出力
        logger.result_scores(self.run_name, cv_scores)

    def run_predict_cv(self):
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        Returns:
            None

        Notes:
            あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction cv')

        test_x = self.load_x_test()

        predictions_based_on_kfolds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            # 各foldの予測開始時にログを出力
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')

            # モデルを読み込んで予測
            model = self.build_model(i_fold)
            model.load_model(self.model_dir)
            test_pred = model.predict(test_x)

            # 各モデルのテストデータへの予測値を作成
            predictions_based_on_kfolds.append(test_pred)

            # 各foldの予測終了時にログを出力
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        avg_test_pred = np.mean(predictions_based_on_kfolds, axis=0)

        # 予測結果の保存
        Util.dump(avg_test_pred,
                  Util.script_based_path(f'../model/pred/{self.run_name}-tests.pkl')
                  )

        logger.info(f'{self.run_name} - end prediction cv')

    def run_train_all(self):
        """学習データすべてで学習し、そのモデルを保存する

        Returns:
            None

        """
        logger.info(f'{self.run_name} - start training all')

        # 学習データ全てで学習を行う
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model(self.model_dir)

        logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self):
        """学習データすべてで学習したモデルにより、テストデータの予測を行う

        Returns:
            None

        Notes:
            あらかじめrun_train_allを実行しておく必要がある

        """
        logger.info(f'{self.run_name} - start prediction all')

        test_x = self.load_x_test()

        # 学習データ全てで学習したモデルで予測を行う
        i_fold = 'all'
        model = self.build_model(i_fold)
        model.load_model(self.model_dir)
        pred = model.predict(test_x)

        # 予測結果の保存
        Util.dump(pred,
                  Util.script_based_path(f'../model/pred/{self.run_name}-tests.pkl')
                  )

        logger.info(f'{self.run_name} - end prediction all')

    def build_model(self, i_fold):
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        Args:
            i_fold(Union[int, str]): foldの番号または、'all'

        Returns:
            AbsModel: モデルのインスタンス

        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_class(run_fold_name, self.params)
