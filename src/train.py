# -*- coding: utf-8 -*-
import argparse
import os
import joblib
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher


def run(fold, model):
    """

    Args:
        fold(int): 使用するfold番号の指定
        model: 使用するモデルの指定

    Returns:
        foldは学習データのkfoldカラムの値を使用している

    """
    # 学習データの読み込み
    df = pd.read_csv(config.TRAINING_FOLD_FILE)

    # 学習データとバリデーションデータに分割して、k_foldカラムを除く
    df_train = df[df.kfold != fold].reset_index(drop=True).drop('kfold', axis=1)
    df_valid = df[df.kfold == fold].reset_index(drop=True).drop('kfold', axis=1)

    # 学習データとバリデーションデータをそれぞれ目的変数と説明変数に分ける
    x_train = df_train.drop(config.TARGET_COLUMN, axis=1)
    y_train = df_train[config.TARGET_COLUMN]

    x_valid = df_valid.drop(config.TARGET_COLUMN, axis=1)
    y_valid = df_valid[config.TARGET_COLUMN]

    # モデルの作成
    clf = model_dispatcher.models[model]
    clf.train(x_train, y_train, x_valid, y_valid)

    # バリデーションデータに対する予測
    pred_valid = clf.predict(x_valid)

    # 指標の計算
    #accuracy = metrics.accuracy_score(y_valid, pred_valid)
    #print(f'Fold={fold}, Accuracy={accuracy}')

    # モデルの保存
    clf.save_model()
    # joblib.dump(clf, os.path.join(config.MODEL_OUTPUT_DIR, f'dt_{fold}.pkl'))


if __name__ == '__main__':
    # ArgumentParserインスタンスの作成
    parser = argparse.ArgumentParser()

    # コマンドラインから受け取る引数の設定
    parser.add_argument(
        '--fold',
        type=int,
    )
    parser.add_argument(
        '--model',
        type=str,
    )

    # コマンドラインに入力された引数を受け取ってargsに格納
    args = parser.parse_args()

    # 学習の実行
    run(fold=args.fold,
        model=args.model,
        )
