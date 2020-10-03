import numpy as np
import pandas as pd


def anomaly_scores(df_original, df_reduced):
    """再構成誤差を計算する異常スコア関数
    
    再構成誤差を0~1にスケールした異常スコアを出す関数

    Args:
        df_original(array-like): training data of shape (n_samples, n_features)
        df_reduced(array-like): prediction of shape (n_samples, n_features)

    Returns:
        pd.Series: 各データごとの異常スコア(二乗誤差をMinMaxScalingしたもの)

    """
    # サンプルごとの予測値との二乗誤差を計算
    loss = np.sum((np.array(df_original) - np.array(df_reduced)) ** 2,
                  axis=1,
                  )

    # lossをpd.Seriesに変換
    loss = pd.Series(data=loss,
                     index=df_original.index,
                     )

    # 二乗誤差をMinMaxScalingして0~1のスコアに変換
    min_max_scaled_loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))

    return min_max_scaled_loss


# 結果指標をプロットする関数

def plotResults(trueLabels, anomalyScores, returnPreds=False):
    """結果指標をプロットする関数

    Args:
        trueLabels(array, shape = [n_samples]): True binary labels

        anomalyScores(array, shape = [n_samples]): Anomary scores

    Returns:
        pd.Series: 各データごとの異常スコア(二乗誤差をMinMaxScalingしたもの)

    """
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    precision, recall, thresholds = \
        precision_recall_curve(preds['trueLabel'], \
                               preds['anomalyScore'])
    average_precision = average_precision_score( \
        preds['trueLabel'], preds['anomalyScore'])

    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall curve: Average Precision = \
        {0:0.2f}'.format(average_precision))

    fpr, tpr, thresholds = roc_curve(preds['trueLabel'], \
                                     preds['anomalyScore'])
    areaUnderROC = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: Area under the \
        curve = {0:0.2f}'.format(areaUnderROC))
    plt.legend(loc="lower right")
    plt.show()

    if returnPreds == True:
        return preds, average_precision