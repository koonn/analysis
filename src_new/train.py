import argparse
import os
import joblib
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher


def run(fold, model):
    # 学習データの読み込み
    df = pd.read_csv(config.TRAINING_FOLD_FILE)

    # 学習データとバリデーションデータに分割
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # 学習データとバリデーションデータをそれぞれ目的変数と説明変数に分ける
    x_train = df_train.drop('target', axis=1).values
    y_train = df_train['target'].values

    x_valid = df_valid.drop('target', axis=1).values
    y_valid = df_valid['target'].values

    # モデルの作成
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)

    # バリデーションデータに対する予測
    pred_valid = clf.predict(x_valid)

    # 指標の計算
    accuracy = metrics.accuracy_score(y_valid, pred_valid)
    print(f'Fold={fold}, Accuracy={accuracy}')

    # モデルの保存
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT_DIR, f'dt_{fold}.pkl'))


if __name__ == '__main__':
    run(fold=0, model='logistic_regression')
    run(fold=1, model='logistic_regression')
    run(fold=2, model='logistic_regression')
    run(fold=3, model='logistic_regression')
    run(fold=4, model='logistic_regression')
