import datetime
import logging
import os

import numpy as np
import pandas as pd
import joblib


class Util:
    """ファイルの入出力などのユーティリティメソッド"""

    @classmethod
    def script_based_path(cls, relative_path):
        """実行ファイル基準の相対パスを、絶対パスに変換する関数

        読み込みファイルを、実行ファイルからの相対パスで指定(スクリプトの実行場所によらず読み込めるようにするため)
        Pythonファイル実行時の相対パスは、実行時のshellのカレントディレクトリからの相対パスになってしまうため、
        実行場所によらず同じファイルを読むようにしたい

        Args:
            relative_path: 実行ファイルからの相対パス

        Returns:
            str: 実行ファイルからの相対パスで指定した場所の絶対パス

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
        # ロガーのインスタンス化
        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')

        # ストリームハンドラインスタンスの作成(コンソールに出力するためのハンドラ)
        stream_handler = logging.StreamHandler()

        # ファイルハンドラクラスのインスタンスを作成(指定されたファイルをオープンしてファイルにログを出力するためのハンドラ)
        file_general_handler = logging.FileHandler(Util.script_based_path('../model/general.log'))
        file_result_handler = logging.FileHandler(Util.script_based_path('../model/result.log'))

        # ロガーにハンドラが設定されていない場合は、ハンドラを追加する
        if len(self.general_logger.handlers) == 0:

            # general_loggerのログをコンソールとファイルに出力するためのハンドラ設定
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

            # result_loggerのログをコンソールとファイルに出力するためのハンドラ設定
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        """時刻をつけてメッセージをコンソールとログに出力する関数

        Args:
            message:

        Returns:

        """
        self.general_logger.info(f'[{self.now_string()}] - {message}')

    def result(self, message):
        """メッセージをコンソールとログに出力する関数

        Args:
            message:

        Returns:

        """
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        """

        Args:
            dic:

        Returns:

        """
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, cv_scores):
        """計算結果をコンソールと計算結果用ログに出力

        Args:
            run_name(str): 実行の名前
            cv_scores(np.array): スコア

        Returns:
            None

        """
        dic = dict()
        dic['name'] = run_name
        dic['average_score'] = np.mean(cv_scores)
        for i, score in enumerate(cv_scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    @staticmethod
    def now_string():
        """実行時の時刻を文字列で返す関数

        Returns:
            str: 関数の実行時の時刻の文字列

        """
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    @staticmethod
    def to_ltsv(dic):
        """引数の辞書の各要素をタブ区切りの文字列で返す関数

        Args:
            dic: 辞書

        Returns:
            str: 辞書の各要素をタブ区切りにした文字列

        """
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


class Submission:
    """結果を書き出すクラス

    Todos:
        サンプルでない実装をする

    """

    @classmethod
    def create_submission(cls, run_name):
        submission = pd.read_csv('../input/sampleSubmission.csv')
        pred = Util.load(f'../model/pred/{run_name}-test.pkl')
        for i in range(pred.shape[1]):
            submission[f'Class_{i + 1}'] = pred[:, i]
        submission.to_csv(f'../submission/{run_name}.csv', index=False)
