"""何らかの数値を以上スコアに変換する関数

"""
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
