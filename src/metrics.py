import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report


def show_all_metrics(y_true, y_pred_proba, threshold=0.5):
    """Precision, Recall, F1-Score, Accuracyを計算して表示する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        threshold(float): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する. デフォルトでは0.5

    Returns:
        None
    """
    # 先にグラフを描画するためにplt.showを使う
    plot_roc_prc(y_true, y_pred_proba)
    plt.show()

    print_classification_report(y_true, y_pred_proba, threshold)
    print_confusion_matrix(y_true, y_pred_proba, threshold)


def print_classification_report(y_true, y_pred_proba, threshold=0.5):
    """Precision, Recall, F1-Score, Accuracyを計算して表示する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        threshold(float): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する. デフォルトでは0.5

    Returns:
        None
    """
    # 予測確率を2値ラベルに変換する
    y_pred_label = np.where(y_pred_proba >= threshold, 1, 0)

    print(f'Classification Report: Positive_decision_threshold={threshold}')
    print(classification_report(y_true, y_pred_label))


def print_confusion_matrix(y_true, y_pred_proba, threshold=0.5):
    """混合行列を整形して表示する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        threshold(float, default=0.5): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する.
    Returns:
        None
    """
    # 予測確率を2値ラベルに変換する
    y_pred_label = np.where(y_pred_proba >= threshold, 1, 0)

    # 混合行列の算出
    confusion_matrix_ = confusion_matrix(y_true, y_pred_label, labels=[1, 0])
    confusion_matrix_ = pd.DataFrame(confusion_matrix_,
                                     index=['actual: 1', 'actual: 0'],
                                     columns=['predict: 1', 'predict: 0'],
                                     )
    # 混合行列を表示
    print(f'Confusion Matrix: Positive_decision_threshold={threshold}')
    print(confusion_matrix_)


def plot_roc_prc(y_true, y_pred_proba):
    """ROC曲線とPR曲線をまとめて描画する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列

    Returns:
        None

    """
    # 横並びのグラフ枠を作成
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # rocとpr曲線を描画
    plot_roc(y_true, y_pred_proba, ax=axes[0])
    plot_prc(y_true, y_pred_proba, ax=axes[1])


def plot_roc(y_true, y_pred_proba, ax=None):
    """ROC曲線を描画する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        ax (matplotlib axes, default=None): プロットするaxオブジェクト. Noneならば新しいfig, axを作成する

    Returns:
        None

    """
    # 閾値ごとの偽陽性率と真陽性率を計算(ROC曲線の入力データ)
    fpr, tpr, thresholds = roc_curve(y_true,
                                     y_pred_proba,
                                     )

    # AUCを計算
    area_under_roc = auc(fpr, tpr)

    # 描画の作成
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    ax.plot(fpr, tpr, color='r', lw=2, label='ROC Curve')
    ax.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC: auc={area_under_roc:0.2f}')


def plot_prc(y_true, y_pred_proba, ax=None):
    """PR曲線を描画する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        ax (matplotlib axes, default=None): プロットするaxオブジェクト. Noneならば新しいfig, axを作成する

    Returns:
        None

    """
    # 閾値ごとのPrecisionとRecallを計算
    precision, recall, thresholds = precision_recall_curve(y_true,
                                                           y_pred_proba
                                                           )

    # average precisionの計算
    average_precision = average_precision_score(y_true,
                                                y_pred_proba
                                                )

    # 描画の作成
    # 描画の作成
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    ax.step(recall, precision, color='k', alpha=0.7, where='post')
    ax.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'PR curve: Average_precision={average_precision:0.2f}')
