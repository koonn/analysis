"""便利にするための補助関数をまとめたモジュール"""
import os


def script_based_path(relative_path):
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
