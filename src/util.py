import datetime
import logging
import os

import numpy as np
import pandas as pd
import joblib


class Util:
    """ファイルの入出力などのユーティリティメソッド"""

    @classmethod
    def convert_script_based_relative_path(cls, relative_path):
        """実行ファイル基準の相対パスを、絶対パスに変換する関数

        読み込みファイルを、実行ファイルからの相対パスで指定(スクリプトの実行場所によらず読み込めるようにするため)
        Pythonファイル実行時の相対パスは、実行時のshellのカレントディレクトリからの相対パスになってしまうため、
        実行場所によらず同じファイルを読むようにしたい

        Args:
            relative_path: 実行ファイルからの相対パス

        Returns:
            実行ファイルからの相対パスで指定した場所の絶対パス

        """
        dir_path = os.path.dirname(os.path.abspath(__file__))
        script_based_relative_path = os.path.normpath(os.path.join(dir_path, relative_path))

        return script_based_relative_path

    @classmethod
    def dump(cls, value, path):
        """pythonオブジェクトを保存する関数

        pythonオブジェクトの保存はjoblib.dumpで行われる

        Args:
            value: 保存するオブジェクト
            path: オブジェクトの保存先のパス

        Returns:
            None

        """
        # 保存先のパスのディレクトリを作成する
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # オブジェクトを保存する
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        """pythonオブジェクトファイルを読み込む関数

        pythonオブジェクトの読み込みはjoblib.loadで行われる

        Args:
            path: オブジェクトのパス

        Returns:
            読み込んだオブジェクト

        """

        return joblib.load(path)


class Logger:
    """ログの出力・表示を行う"""

    def __init__(self):

        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(Util.convert_script_based_relative_path('../model/general.log'))
        file_result_handler = logging.FileHandler(Util.convert_script_based_relative_path('../model/result.log'))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        """時刻をつけてコンソールとログに出力"""
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        """計算結果をコンソールと計算結果用ログに出力

        Args:
            run_name(str): 実行の名前
            scores(np.array): スコア

        Returns:
            None

        """
        dic = dict()
        dic['name'] = run_name
        dic['score'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    @staticmethod
    def to_ltsv(dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


class Submission:
    """結果を書き出すクラスメソッド

    TODO:
        書き直す
    """

    @classmethod
    def create_submission(cls, run_name):
        submission = pd.read_csv('../input/sampleSubmission.csv')
        pred = Util.load(f'../model/pred/{run_name}-test.pkl')
        for i in range(pred.shape[1]):
            submission[f'Class_{i + 1}'] = pred[:, i]
        submission.to_csv(f'../submission/{run_name}.csv', index=False)
