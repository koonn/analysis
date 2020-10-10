"""精度評価指標をプロットする関数

y_trueと、y_pred_probaから精度評価指標をプロットするための関数群

"""
from itertools import product
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import numpy as np


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
    # 横並びのグラフ枠を作成
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    # 指標を描画
    plot_roc(y_true, y_pred_proba, ax=axes[0, 0])
    plot_prc(y_true, y_pred_proba, ax=axes[0, 1])
    plot_confusion_matrix(y_true, y_pred_proba, ax=axes[1, 0], threshold=threshold)
    plot_classification_report(y_true, y_pred_proba, ax=axes[1, 1], threshold=threshold)


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
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred_proba)

    # AUCを計算
    area_under_roc = skm.auc(fpr, tpr)

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
    precision, recall, thresholds = skm.precision_recall_curve(y_true, y_pred_proba)

    # average precisionの計算
    average_precision = skm.average_precision_score(y_true, y_pred_proba)

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


def plot_confusion_matrix(y_true, y_pred_proba, threshold=0.5, ax=None):
    """混合行列を整形してプロットとして表示する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        threshold(float, default=0.5): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する
        ax (matplotlib axes, default=None): プロットするaxオブジェクト. Noneならば新しいfig, axを作成する

    Returns:
        None
    """
    # 予測確率を2値ラベルに変換する
    y_pred_label = np.where(y_pred_proba >= threshold, 1, 0)

    # 混合行列の算出
    confusion_matrix_ = skm.confusion_matrix(y_true, y_pred_label, labels=[1, 0])

    # 描画の作成
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    # 混合行列のディスプレイ作成のためのConfusionMatrixDisplayインスタンス作成
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_,
                                      display_labels=[1, 0]
                                      )

    # 描画の作成
    disp.plot(
        include_values=True,
        cmap='Blues',
        ax=ax,
        xticks_rotation='horizontal',
        values_format='d',
    )

    ax.set_title(f'Confusion Matrix: Pos_decision_threshold={threshold}')


def plot_classification_report(y_true, y_pred_proba, threshold=0.5, ax=None):
    """Precision, Recall, F1-Score等の指標を計算して表示する関数

    Args:
        y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
        y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列
        threshold(float): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する. デフォルトでは0.None
        ax (matplotlib axes, default=None): プロットするaxオブジェクト. Noneならば新しいfig, axを作成する

    Returns:
        None

    """
    # 予測確率を2値ラベルに変換する
    y_pred_label = np.where(y_pred_proba >= threshold, 1, 0)

    # 使用する指標を計算
    precision, recall, fscore, support = skm.precision_recall_fscore_support(y_true, y_pred_label, labels=[1, 0])
    acc = skm.accuracy_score(y_true, y_pred_label)

    # precision, recall, fscoreをヒートマップ用の行列形式に整形する
    matrix_measures = np.array([precision, recall, fscore]).T

    # 描画の作成
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    # ヒートマップを作成
    im_ = ax.imshow(matrix_measures, interpolation='nearest', cmap='Reds')

    # テキストの色を背景色に合わせていい感じに調整するための閾値を作成
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
    color_threshold = (matrix_measures.max() + matrix_measures.min()) / 2.0

    # matrix_measuresの行列数を取得(iterationで意味がわかりやすいようにするため、あえて名前をつけた)
    n_classes = matrix_measures.shape[0]
    n_measures = matrix_measures.shape[1]

    # テキストを描画する
    for i, j in product(range(n_classes), range(n_measures)):

        # テキスト色を決める
        if matrix_measures[i, j] < color_threshold:
            text_color = cmap_max
        else:
            text_color = cmap_min

        # テキストを描画
        text = ax.text(x=j,
                       y=i,
                       s=f'{matrix_measures[i, j]:.2f}',
                       ha='center',
                       va='center',
                       color=text_color,
                       )

    # カラーバーの描画設定
    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height,
                        ])
    fig.colorbar(im_, cax=cax)

    # そのほかの描画設定
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['Precision', 'Recall', 'F1-score'])

    ax.set_yticks(np.arange(2))
    ax.set_yticklabels([f'1 \n (N={support[0]})',
                        f'0 \n (N={support[1]})',
                        ])  # supportのindexとラベルが逆なので注意

    ax.set_xlabel('Measures')
    ax.set_ylabel('True label')
    ax.set_title(f'Classification report: Pos_decision_threshold={threshold} \n Accuracy={acc:.2f}')
