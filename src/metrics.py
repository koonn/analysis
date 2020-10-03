from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Metrics:
    """指標の可視化、集計を行うクラス"""

    def __init__(self, y_true, y_pred_proba):
        """イニシャライザ

        Args:
            y_true(1-D array-like shape of [n_samples, ]): 2値の目的ラベルの配列(ラベルは0または1)
            y_pred_proba(1-D array-like shape of [n_samples, ]): 陽性(ラベルが1)である確率の予測値の配列

        """
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba

    # ----------------
    # 指標計算系のメソッド
    # ----------------
    def classification_report(self, threshold=0.5):
        """Precision, Recall, F1-Score, Accuracyを計算して表示する関数

        Args:
            threshold(float): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する. デフォルトでは0.None

        Returns:
            report(string / dict): Precision, Recall, F1-Score, Accuracyの指標のレポート結果

        Notes:
            人間が見やすい形式で返すので、何かに使う場合は他の関数などで指標を計算するほうがいい

        Todos:
            いらなくなりそうなので消す予定
        """
        # 予測確率を2値ラベルに変換する
        y_pred_label = np.where(self.y_pred_proba >= threshold, 1, 0)

        # レポートを生成
        classification_report_ = classification_report(self.y_true, y_pred_label)

        return classification_report_

    def confusion_matrix(self, threshold=0.5):
        """混合行列を整形して計算

        Args:
            threshold(float, default=0.5): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する

        Returns:
            Tuple[pd.DataFrame, float]: (混合行列, 識別境界の値)のタプルを返す
        """
        # 予測確率を2値ラベルに変換する
        y_pred_label = np.where(self.y_pred_proba >= threshold, 1, 0)

        # 混合行列の算出
        confusion_matrix_ = confusion_matrix(self.y_true, y_pred_label, labels=[1, 0])
        confusion_matrix_ = pd.DataFrame(confusion_matrix_,
                                         index=['actual: 1', 'actual: 0'],
                                         columns=['predict: 1', 'predict: 0'],
                                         )

        return confusion_matrix_, threshold

    # ----------------
    # プロット系のメソッド
    # ----------------
    def show_all_metrics(self, threshold=0.5):
        """Precision, Recall, F1-Score, Accuracyを計算して表示する関数

        Args:
            threshold(float): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する. デフォルトでは0.5

        Returns:
            None
        """
        # 先にグラフを描画するためにplt.showを使う
        # 横並びのグラフ枠を作成
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.subplots_adjust(wspace=0.4, hspace=0.6)

        # 指標を描画
        self.plot_roc(ax=axes[0, 0])
        self.plot_prc(ax=axes[0, 1])
        self.plot_confusion_matrix(ax=axes[1, 0], threshold=threshold)
        self.plot_classification_report(ax=axes[1, 1], threshold=threshold)

    def plot_roc(self, ax=None):
        """ROC曲線を描画する関数

        Args:
            ax (matplotlib axes, default=None): プロットするaxオブジェクト. Noneならば新しいfig, axを作成する

        Returns:
            None

        """
        # 閾値ごとの偽陽性率と真陽性率を計算(ROC曲線の入力データ)
        fpr, tpr, thresholds = roc_curve(self.y_true,
                                         self.y_pred_proba,
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

    def plot_prc(self, ax=None):
        """PR曲線を描画する関数

        Args:
            ax (matplotlib axes, default=None): プロットするaxオブジェクト. Noneならば新しいfig, axを作成する

        Returns:
            None

        """
        # 閾値ごとのPrecisionとRecallを計算
        precision, recall, thresholds = precision_recall_curve(self.y_true,
                                                               self.y_pred_proba
                                                               )

        # average precisionの計算
        average_precision = average_precision_score(self.y_true,
                                                    self.y_pred_proba
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

    def plot_confusion_matrix(self, threshold=0.5, ax=None):
        """混合行列を整形してプロットとして表示する関数

        Args:
            threshold(float, default=0.5): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する
            ax (matplotlib axes, default=None): プロットするaxオブジェクト. Noneならば新しいfig, axを作成する

        Returns:
            None
        """
        # 予測確率を2値ラベルに変換する
        y_pred_label = np.where(self.y_pred_proba >= threshold, 1, 0)

        # 混合行列の算出
        confusion_matrix_ = confusion_matrix(self.y_true, y_pred_label, labels=[1, 0])

        # 描画の作成
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.figure

        # 混合行列のディスプレイ作成のためのConfusionMatrixDisplayインスタンス作成
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_,
                                      display_labels=[1, 0]
                                      )

        # 描画の作成
        disp.plot(
            include_values=True,
            cmap='Blues',
            ax=ax,
            xticks_rotation='horizontal',
            values_format=None,
        )

        ax.set_title(f'Confusion Matrix: Pos_decision_threshold={threshold}')

    def plot_classification_report(self, threshold=0.5, ax=None):
        """Precision, Recall, F1-Score等の指標を計算して表示する関数

        Args:
            threshold(float): 陽性と分類する確率の閾値. 陽性(ラベルが1)である確率の予測値がthreshold以上なら1に変換する. デフォルトでは0.None
            ax (matplotlib axes, default=None): プロットするaxオブジェクト. Noneならば新しいfig, axを作成する

        Returns:
            None

        """
        # 予測確率を2値ラベルに変換する
        y_pred_label = np.where(self.y_pred_proba >= threshold, 1, 0)

        # 使用する指標を計算
        precision, recall, fscore, support = precision_recall_fscore_support(self.y_true, y_pred_label, labels=[1, 0])
        acc = accuracy_score(self.y_true, y_pred_label)

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

        # そのほかの描画設定
        fig.colorbar(im_, ax=ax)

        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(['Precision', 'Recall', 'F1-score'])

        ax.set_yticks(np.arange(2))
        ax.set_yticklabels([f'1 \n (N={support[0]})',
                            f'0 \n (N={support[1]})',
                            ])  # supportのindexとラベルが逆なので注意

        ax.set_xlabel('Measures')
        ax.set_ylabel('True label')
        ax.set_title(f'Classification report: Pos_decision_threshold={threshold} \n Accuracy={acc:.2f}')

        fig.tight_layout()
