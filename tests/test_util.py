import pytest
from src import Util


def test_script_based_relative_path():
    """Util.script_based_relative_pathの正常系テスト

    TODO:
        - 自分のローカルの絶対パスでテストしているので環境が変わったら書き換える
    """
    assert Util.script_based_path('../train.csv') == '/Users/takahirokonno/pyprojects/bunseki/train.csv'
