"""モデルないで共通して使う関数

"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def anomary_scores_ae(df_original, df_reduced):
    """AEで再生成された特徴量から異常度を計算する関数"""
    """再構成誤差を計算する異常スコア関数

    Args:
        df_original(array-like): training data of shape (n_samples, n_features)
        df_reduced(array-like): prediction of shape (n_samples, n_features)

    Returns:
        pd.Series: 各データごとの異常スコア(二乗誤差をMinMaxScalingしたもの)

    """
    # サンプルごとの予測値との二乗誤差を計算
    loss = np.sum((np.array(df_original) - np.array(df_reduced)) ** 2, axis=1)

    # lossをpd.Seriesに変換
    loss = pd.Series(data=loss, index=df_original.index)

    # 二乗誤差をMinMaxScalingして0~1のスコアに変換
    min_max_normalized_loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))

    return min_max_normalized_loss


def scale_scores(scores, is_reversed=False):
    """異常度などのスコアを0-1にスケーリングするための関数

    Args:
        scores(1-D array-like): 異常度スコアの一次元配列
        is_reversed(boolean):
            scoresで入れられた異常度スコアが逆順であるかの変数
            scoresが正順(scoresが大きいほど異常度が高い)ならFalse, scoresが逆順(scoresが小さいほど異常度が高い)ならTrueとする
            デフォルトはFalse

    Returns(1-D array-like):
        変換された異常度スコア。0-1の範囲の値をとり、1に近いほど異常度が高いようなスコア。

    Notes:
        is_reversedが正しく設定されていないと、返り値のスコアが小さいほど異常度が高いとなってしまい解釈が逆になってしまうので注意
        (特徴量として使うなら問題ないが、解釈で混乱する可能性がある)

    """
    # scoresを2D-arrayに直す
    scores = scores.reshape(-1, 1)

    # scalerインスタンスを作成
    scaler = MinMaxScaler()

    # スコアを変換して返す
    if not is_reversed:
        return scaler.fit_transform(scores)
    else:
        return 1 - scaler.fit_transform(scores)
