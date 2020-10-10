# -*- coding: utf-8 -*-
"""モデルを記載するモジュール

Absモデルを継承したモデルを作成する

"""
# その他のインポート
import os
from abc import ABC
import joblib

# モデルの読み込み
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# モジュールからの読み込み
from interface import AbsModel
import config


class ModelXgb(AbsModel):
    """XGBoostのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params):
        """コンストラクタ"""
        super().__init__(params)

    def train(self, train_x, train_y, valid_x=None, valid_y=None):
        """モデルの学習を行う関数"""
        # ハイパーパラメータの設定
        params = dict(self.params)
        num_round = params.pop('num_round')
        early_stopping_rounds = params.pop('early_stopping_rounds')

        # 学習データ・バリデーションデータをDMatrix形式に変換
        d_train = xgb.DMatrix(train_x, label=train_y)
        d_valid = xgb.DMatrix(valid_x, label=valid_y)

        # 学習を実行. watchlistは、学習中に評価指標を計算する対象のデータセット
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]
        self.model = xgb.train(params=params,
                               dtrain=d_train,
                               num_boost_round=num_round,
                               evals=watchlist,
                               early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=100,  # 何回のイテレーションごとに出力するか
                               )

    def predict(self, x):
        """予測確率をの算出を行う関数"""
        d_x = xgb.DMatrix(x)

        return self.model.predict(d_x, ntree_limit=self.model.best_ntree_limit)

    def save_model(self):
        """モデルを保存する関数"""
        model_path = os.path.join(config.MODEL_OUTPUT_DIR, f'{self.run_name}.pkl')
        joblib.dump(self.model, model_path)

    def load_model(self):
        """モデルを作成する関数"""
        pass


class ModelRandomForestClassifier(AbsModel):
    """RandomForestのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params):
        """コンストラクタ"""
        super().__init__(params)

    def train(self, train_x, train_y, valid_x=None, valid_y=None):
        """モデルの学習を行う関数"""
        # モデルの構築・訓練
        model = RandomForestClassifier(**self.params)
        model.fit(train_x, train_y)

        # モデルを保持する
        self.model = model

    def predict(self, x):
        """予測確率を算出する関数"""
        pred_proba = self.model.predict_proba(x)

        return pred_proba

    def save_model(self):
        """モデルを保存する関数"""
        model_path = os.path.join(config.MODEL_OUTPUT_DIR, f'{self.run_name}.pkl')
        joblib.dump(self.model, model_path)

    def load_model(self):
        """モデルを作成する関数"""
        pass


class ModelDecisionTreeClassifier(AbsModel):
    """DecisionTreeのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params):
        """コンストラクタ"""
        super().__init__(params)

    def train(self, train_x, train_y, valid_x=None, valid_y=None):
        """モデルの学習を行う関数"""
        # モデルの構築・訓練
        model = DecisionTreeClassifier(**self.params)
        model.fit(train_x, train_y)

        # モデルを保持する
        self.model = model

    def predict(self, x):
        """予測確率を算出する関数"""
        pred_proba = self.model.predict_proba(x)

        return pred_proba

    def save_model(self):
        """モデルを保存する関数"""
        model_path = os.path.join(config.MODEL_OUTPUT_DIR, f'{self.run_name}.pkl')
        joblib.dump(self.model, model_path)

    def load_model(self):
        """モデルを作成する関数"""
        pass


class ModelLogisticRegression(AbsModel):
    """LogisticRegressionのモデルクラス

        Attributes:
            run_name(str): 実行の名前とfoldの番号を組み合わせた名前
            params(dict): ハイパーパラメータ
            model(Model): train後に学習済みモデルを保持. trainを実行するまでは、初期値のNoneをかえす.
            scaler(Model): train後に学習済みスケーラーを保持. trainを実行するまでは、初期値のNoneをかえす.

    """
    def __init__(self, params):
        super().__init__(params)
        self.scaler = None

    def train(self, train_x, train_y, valid_x=None, valid_y=None, features_to_scale=None):
        """モデルの学習を行う関数

        Args:
            train_x(pd.DataFrame of [n_samples, n_features]): 学習データの特徴量
            train_y(1-D array-like shape of [n_samples]): 学習データのラベル配列
            valid_x(array-like shape of [n_samples, n_features]): バリデーションデータの特徴量
            valid_y(1-D array-like shape of [n_samples]): バリデーションデータのラベル配列
            features_to_scale(Optional[List[str]]): スケール対象の特徴量を指定する

        """
        # データのスケーリング
        # スケールするカラムを指定
        if features_to_scale is None:
            features_to_scale = train_x.columns

        # スケーラを作成
        scaler = StandardScaler()
        scaler.fit(train_x[features_to_scale])

        # スケーリングを実行
        train_x.loc[:, features_to_scale] = scaler.transform(train_x[features_to_scale])

        # モデルの構築・学習
        model = LogisticRegression(**self.params)
        model = model.fit(train_x, train_y)

        # モデル・スケーラーを保持する
        self.model = model
        self.scaler = scaler

    def predict(self, x):
        """予測確率を算出する関数"""
        x = self.scaler.transform(x)
        pred = self.model.predict_proba(x)
        return pred

    def save_model(self):
        """モデルを保存する関数"""
        # パスの設定
        model_dir = os.path.join(config.MODEL_OUTPUT_DIR, 'lr')
        model_path = os.path.join(model_dir, f'{self.run_name}-model.pkl')
        scaler_path = os.path.join(model_dir, f'{self.run_name}-scaler.pkl')

        # 保存先のディレクトリがなければ作成
        os.makedirs(model_dir, exist_ok=True)

        # モデル・スケーラーの保存
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self):
        """モデルを作成する関数"""
        pass


if __name__ == '__main__':

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

    # インスタンスの作成確認用
    model_xgb = ModelXgb(params_xgb)
    model_rf = ModelRandomForestClassifier()
