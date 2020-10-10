"""精度評価指標を計算するモジュール

y_trueと、y_pred_probaから精度評価指標を計算するための関数群

"""
import sklearn.metrics as skm
import numpy as np


def auc(y_true, y_pred_proba):
    """AUCを計算する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列

    Returns:
        float: (ROCの)AUCの値

    """
    return skm.auc(y_true, y_pred_proba)


def average_precision(y_true, y_pred_proba):
    """平均Precisionを計算する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列

    Returns:
        float: 平均Precisionの値
    """
    return skm.average_precision_score(y_true, y_pred_proba)


def acc(y_true, y_pred_proba, threshold=0.5):
    """Accuracyを計算する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        threshold(float, default=0.5): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する

    Returns:
        float: Accuracyの値
    """
    # 予測確率を2値ラベルに変換する
    y_pred_label = np.where(y_pred_proba >= threshold, 1, 0)

    return skm.accuracy_score(y_true, y_pred_label)


def precision(y_true, y_pred_proba, threshold=0.5):
    """Accuracyを計算する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        threshold(float, default=0.5): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する

    Returns:
        float: Accuracyの値
    """
    # 予測確率を2値ラベルに変換する
    y_pred_label = np.where(y_pred_proba >= threshold, 1, 0)

    return skm.precision_score(y_true, y_pred_label)


def recall(y_true, y_pred_proba, threshold=0.5):
    """Accuracyを計算する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        threshold(float, default=0.5): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する

    Returns:
        float: Accuracyの値
    """
    # 予測確率を2値ラベルに変換する
    y_pred_label = np.where(y_pred_proba >= threshold, 1, 0)

    return skm.recall_score(y_true, y_pred_label)


def f1_score(y_true, y_pred_proba, threshold=0.5):
    """Accuracyを計算する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        threshold(float, default=0.5): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する

    Returns:
        float: Accuracyの値
    """
    # 予測確率を2値ラベルに変換する
    y_pred_label = np.where(y_pred_proba >= threshold, 1, 0)

    return skm.f1_score(y_true, y_pred_label)
