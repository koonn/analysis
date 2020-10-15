import numpy as np
import pandas as pd

from src_archived import ModelNN, ModelXGB, ModelLogisticRegression
from src_archived.util import Util
from src_archived import RunSettings
from src_archived import KaggleFeaturesRunner


if __name__ == '__main__':

    # lrのハイパーパラメータ設定
    params_lr = {
        'penalty': 'l2',
        'C': 1.0,
        'class_weight': 'balanced',
        'random_state': 2018,
        'solver': 'liblinear',
        'n_jobs': 1,
    }

    # XGBoostのハイパーパラメータ設定
    params_xgb = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'max_depth': 12,
        'eta': 0.1,
        'min_child_weight': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'silent': 1,
        'random_state': 71,
        'num_round': 10000,
        'early_stopping_rounds': 10,
    }

    params_xgb_all = dict(params_xgb)
    params_xgb_all['num_round'] = 350

    # NNのハイパーパラメータ設定
    params_nn = {
        'layers': 3,
        # サンプルのため早く終わるように設定
        'nb_epoch': 5,  # 1000
        'patience': 10,
        'dropout': 0.5,
        'units': 512,
    }

    # 特徴量の指定
    run_settings_kagglebook = RunSettings(
        file_path_train=Util.script_based_path('../data/features/train.csv'),
        file_path_test=Util.script_based_path('../data/features/test.csv'),
        model_dir=Util.script_based_path('../model_archived/model/lr/'),
        target=['target'],
        features=[f'feat_{i}' for i in range(1, 94)],
        features_to_scale=[f'feat_{i}' for i in range(1, 94)],
        )

    # LogisticRegressionによる学習・予測
    runner = KaggleFeaturesRunner('lr1', ModelLogisticRegression, params_lr, run_settings_kagglebook, n_fold=2)
    runner.run_train_cv()
    #runner.run_predict_cv()

    # xgboostによる学習・予測
    #runner = Runner('xgb1', ModelXGB, features, params_xgb)
    #runner.run_train_cv()
    #runner.run_predict_cv()

    # ニューラルネットによる学習・予測
    #runner = Runner('nn1', ModelNN, features, params_nn)
    #runner.run_train_cv()
    #runner.run_predict_cv()

    '''
    # (参考）xgboostによる学習・予測 - 学習データ全体を使う場合
    runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb_all)
    runner.run_train_all()
    runner.run_test_all()
    Submission.create_submission('xgb1-train-all')
    '''