# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np

import config
import model_dispatcher
from metrics.metrics import acc, auc, precision, recall


def train_fold(fold, model_name, save_model=True, df=None):
    """foldごとの訓練を行うコア関数

    Args:
        fold(int): 使用するfold番号の指定
        model_name(str): 使用するモデルの指定
        save_model(boolean): モデルを保存するかどうか
        df(pd.DataFrame): 訓練用のデータ. バリデーション用の番号のカラムkfoldが入っている状態のもの.デフォルトはdf=Noneで、指定しなければ、
                          configで指定された訓練用データを読み込む(config.)

    Returns:
        Model: モデルインスタンスを返す

    Notes:
        foldは学習データのkfoldカラムの値を使用している

    """
    # 実行(run)の名前を定義
    run_name = f'{model_name}-{fold}'

    # 学習データが指定されていない場合は学習データを読み込む
    if df is None:
        df = pd.read_csv(config.TRAINING_FOLD_FILE)

    # 学習データとバリデーションデータに分割して、k_foldカラムを除く
    df_train = df[df.kfold != fold].drop('kfold', axis=1)
    df_valid = df[df.kfold == fold].drop('kfold', axis=1)

    # 学習データとバリデーションデータをそれぞれ目的変数と説明変数に分ける
    x_train = df_train.drop(config.TARGET_COLUMN, axis=1)
    y_train = df_train[config.TARGET_COLUMN]

    x_valid = df_valid.drop(config.TARGET_COLUMN, axis=1)
    y_valid = df_valid[config.TARGET_COLUMN]

    # モデルの作成
    model = model_dispatcher.models[model_name]
    model.train(x_train, y_train, x_valid, y_valid)

    # バリデーションデータのindexと、それに対する予測
    index_valid = y_valid.index
    pred_valid = model.predict(x_valid)

    # 指標の計算
    print(f'Model Name: {model_name}, Fold: {fold} ',
          f'\n Validation Result:',
          f'\n      AUC={auc(y_valid, pred_valid):.5f}',
          f'\n      Accuracy={acc(y_valid, pred_valid):.5f}',
          f'\n      Precision={precision(y_valid, pred_valid):.5f}',
          f'\n      Recall={recall(y_valid, pred_valid):.5f}',
          )

    # モデルの保存
    if save_model:
        model.run_name = run_name
        model.save_model()

    return model, index_valid, pred_valid


def run_train_cv(model_name, save_model=True, validate_with_test=True):
    """

    Args:
        model_name(str): 使用するモデルの指定
        save_model(boolean): モデルを保存するかどうか. デフォルトはTrue
        validate_with_test(boolean): テストデータを使ってバリデーションするかどうか. デフォルトはTrue

    Returns:
        Model: モデルインスタンスを返す

    Notes:
        foldは学習データのkfoldカラムの値を使用している

    """
    # 学習データの読み込み
    df = pd.read_csv(config.TRAINING_FOLD_FILE)

    # データの保存先
    indexes_cv = []
    predictions_cv = []

    for i_fold in range(config.N_FOLDS):

        # モデルの学習を実行
        model, index_valid, pred_valid = train_fold(i_fold, model_name, df=df, save_model=save_model)

        # fold毎の結果を保持する
        indexes_cv.append(index_valid)  # バリデーションのデータのインデックス
        predictions_cv.append(pred_valid)  # バリデーションに対する予測値

    # 全foldの結果をまとめる
    indexes_cv = np.concatenate(indexes_cv)  # 全てのindexが入ったnp.arrayを作成
    order = np.argsort(indexes_cv)  # valid_indexをソートするためのindexが入ったnp.arrayを作成
    predictions_cv = np.concatenate(predictions_cv, axis=0)  # バリデーション予測結果が入ったnp.arrayを作成
    predictions_cv = predictions_cv[order]  # もともとのindexの昇順に、バリデーション予測結果を並べる

    # バリデーションでの指標の計算
    y_true = df[config.TARGET_COLUMN].values

    print(f'Model Name: {model_name}',
          f'\n All Validation Result:',
          f'\n      AUC={auc(y_true, predictions_cv):.5f}',
          f'\n      Accuracy={acc(y_true, predictions_cv):.5f}',
          f'\n      Precision={precision(y_true, predictions_cv):.5f}',
          f'\n      Recall={recall(y_true, predictions_cv):.5f}',
          )

    # テストデータでの指標の計算
    if validate_with_test:
        # テストデータの読み込み
        df_test = pd.read_csv(config.TEST_FILE)

        # テストデータをそれぞれ目的変数と説明変数に分ける
        x_test = df_test.drop(config.TARGET_COLUMN, axis=1)
        y_test_true = df_test[config.TARGET_COLUMN]

        # テストデータに対する予測を作成
        y_test_pred = model.predict(x_test)

        print(f'Model Name: {model_name}',
              f'\n Test Result:',
              f'\n      AUC={auc(y_test_true, y_test_pred):.5f}',
              f'\n      Accuracy={acc(y_test_true, y_test_pred):.5f}',
              f'\n      Precision={precision(y_test_true, y_test_pred):.5f}',
              f'\n      Recall={recall(y_test_true, y_test_pred):.5f}',
              )

    return model, indexes_cv, predictions_cv


if __name__ == '__main__':
    # ArgumentParserインスタンスの作成
    parser = argparse.ArgumentParser()

    # コマンドラインから受け取る引数の設定
    parser.add_argument(
        '--fold',
        type=int,
    )
    parser.add_argument(
        '--model_name',
        type=str,
    )

    # コマンドラインに入力された引数を受け取ってargsに格納
    args = parser.parse_args()

    # 学習の実行
    # train_fold(fold=args.fold, model_name=args.model_name)
    run_train_cv(model_name=args.model_name)
