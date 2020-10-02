import os
import numpy as np
import pandas as pd
from .model import Model
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from typing import List, Optional, Tuple, Union

from .util import Logger, Util

logger = Logger()


class Runner:

    def __init__(self, run_name, model_class, features, params, n_fold=4):
        """コンストラクタ

        Args:
            run_name(str): ランの名前
            model_class(Callable[[str, dict], Model]): モデルのクラス
            features(List[str]): 特徴量のリスト
            params(dict): ハイパーパラメータ
            n_fold(int): クロスバリデーションの分割数

        """
        self.run_name = run_name
        self.model_class = model_class
        self.features = features
        self.params = params
        self.n_fold = n_fold

    def train_fold(self, i_fold):
        """クロスバリデーションの指定した1つのfoldに対して学習・評価を行う

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        Args:
            i_fold(Union[int, str]): n_foldで指定した分割のうち、何番目の分割をバリデーション用に使うか（バリデーションしない時には'all'とする）

        Returns:
            Tuple[Model, Optional[np.array], Optional[np.array], Optional[float]]:
                （モデルのインスタンス、バリデーションデータのインデックス、バリデーションデータの予測値、評価関数によるスコア）のタプル

        Todo:
            - バリデーションしない時には'all'とするのはなんか変な気がするので、Noneとかに直す
            - 評価関数も選択できるようにする
        """
        # 学習データの読込
        data_x = self.load_x_train()
        data_y = self.load_y_train()

        # バリデーションするかどうか
        do_validation = (i_fold != 'all')

        if do_validation:

            # 学習データ・バリデーションデータをセットする

            # 学習データ・バリデーションデータの分割用インデックスを取得
            train_index, valid_index = self.load_index_fold(i_fold)

            # 学習データを抽出
            train_x = data_x.iloc[train_index]
            train_y = data_y.iloc[train_index]

            # バリデーションデータを抽出
            valid_x = data_x.iloc[valid_index]
            valid_y = data_y.iloc[valid_index]

            # 学習を行う
            model = self.build_model(i_fold)
            model.train(train_x, train_y, valid_x, valid_y)

            # バリデーションデータへの予測・評価を行う
            valid_pred = model.predict(valid_x)
            score = log_loss(valid_y, valid_pred, eps=1e-15, normalize=True)

            # (モデルのインスタンス、バリデーションデータのインデックス、バリデーションデータの予測値、評価関数によるスコア)のタプルを返す
            return model, valid_index, valid_pred, score
        else:
            # 学習データ全てで学習を行う
            model = self.build_model(i_fold)
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
        logger.info(f'{self.run_name} - start training cv')

        scores = []
        valid_indexes = []
        preds = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, valid_index, valid_pred, score = self.train_fold(i_fold)
            logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            model.save_model()

            # 結果を保持する

            # 各foldでバリデーションの対象になったデータのインデックス
            valid_indexes.append(valid_index)

            # スコア
            scores.append(score)

            # バリデーションに対する予測値
            preds.append(valid_pred)

        # 各foldの結果をまとめる
        valid_indexes = np.concatenate(valid_indexes)
        order = np.argsort(valid_indexes)
        preds = np.concatenate(preds, axis=0)
        preds = preds[order]

        logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 予測結果の保存
        Util.dump(preds, f'../model/pred/{self.run_name}-train.pkl')

        # 評価結果の保存
        logger.result_scores(self.run_name, scores)

    def run_predict_cv(self):
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        Returns:
            None

        Notes:
            あらかじめrun_train_cvを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction cv')

        test_x = self.load_x_test()

        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            model = self.build_model(i_fold)
            model.load_model()
            pred = model.predict(test_x)
            preds.append(pred)
            logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        pred_avg = np.mean(preds, axis=0)

        # 予測結果の保存
        Util.dump(pred_avg, f'../model/pred/{self.run_name}-tests.pkl')

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
        model.save_model()

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
        model.load_model()
        pred = model.predict(test_x)

        # 予測結果の保存
        Util.dump(pred, f'../model/pred/{self.run_name}-tests.pkl')

        logger.info(f'{self.run_name} - end prediction all')

    def build_model(self, i_fold):
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        Args:
            i_fold(Union[int, str]): foldの番号または、'all'

        Returns:
            Model: モデルのインスタンス

        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        run_fold_name = f'{self.run_name}-{i_fold}'
        return self.model_class(run_fold_name, self.params)

    def load_x_train(self):
        """学習データの特徴量を読み込む

        Returns:
            pd.DataFrame: 学習データの特徴量

        Notes:
            学習データの読込を行う
            列名で抽出する以上のことを行う場合、このメソッドの修正が必要
            毎回train.csvを読み込むのは効率が悪いため、データに応じて適宜対応するのが望ましい（他メソッドも同様）

        """
        # 読み込みファイルを、実行ファイルからの相対パスで指定(スクリプトの実行場所によらず読み込めるようにするため)
        file_path = Util.convert_script_based_relative_path('../data/features/train.csv')

        return pd.read_csv(file_path)[self.features]

    @staticmethod
    def load_y_train():
        """学習データの目的変数を読み込む

        Returns:
            pd.Series: 学習データの目的変数

        """
        # 読み込みファイルを、実行ファイルからの相対パスで指定(スクリプトの実行場所によらず読み込めるようにするため)
        file_path = Util.convert_script_based_relative_path('../data/features/train.csv')

        # ファイルを読み込んで目的変数を抽出
        train_y = pd.read_csv(file_path)['target']

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
        file_path = Util.convert_script_based_relative_path('../data/features/test.csv')

        return pd.read_csv(file_path)[self.features]

    def load_index_fold(self, i_fold):
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        学習データ・バリデーションデータを分けるインデックスを返す
        ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある

        Args:
            i_fold(int): foldの番号

        Returns:
            np.array: foldに対応するレコードのインデックス

        """
        train_y = self.load_y_train()
        dummy_x = np.zeros(len(train_y))
        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=71)
        return list(skf.split(dummy_x, train_y))[i_fold]
