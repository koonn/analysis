import pandas as pd
from sklearn.model_selection import StratifiedKFold

from func_util import script_based_path


def create_folds(n_folds=5):
    """fold番号がついたトレインデータを作成する変数

    Args:
        n_folds(int): foldの数

    Returns:
        None

    """
    # foldのないトレインデータの読み込み
    df = pd.read_csv(script_based_path('../data/features/train.csv'))

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
    df.to_csv(script_based_path('../data/features/train_folds.csv'), index=False)


if __name__ == '__main__':
    create_folds()
