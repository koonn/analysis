"""fold番号がついたトレインデータを作成する変数"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config


def create_folds(n_folds=5):
    """fold番号がついたトレインデータを作成する変数

    Args:
        n_folds(int): foldの数

    Returns:
        None

    """
    # foldのないトレインデータの読み込み
    df = pd.read_csv(config.TRAINING_FILE)

    y_train = df['target']
    x_train = df.drop('target', axis=1)

    # k分割考査検証法で分割するためのインスタンス作成
    k_fold = StratifiedKFold(n_splits=n_folds,
                             shuffle=True,
                             random_state=2020,
                             )

    # dfにfold番号のカラムを追加する
    for i, (_train_index, cv_index) in enumerate(k_fold.split(x_train, y_train)):
        df.loc[cv_index, 'kfold'] = str(i)

    # ファイルを書き出す
    df.to_csv(config.TRAINING_FOLD_FILE, index=False)


if __name__ == '__main__':
    create_folds()
