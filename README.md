# Bunseki
分析用に手法やモジュールを整理したもの

## フォルダ構成

```shell script
.
├── LICENSE
├── README.md
├── bin
│   └── run.sh
├── data
│   ├── credit_card_data
│   ├── externals
│   └── features
├── doc
├── model
├── notebook
├── poetry.lock → 
├── pyproject.toml → プロジェクトの設定
├── src → pythonスクリプトを記載したフォルダ
│   ├── config.py
│   ├── create_folds.py
│   ├── metrics
│   ├── model_dispatcher.py
│   ├── models
│   ├── train.py
│   └── util.py
├── src_archived
│   ├── __init__.py
│   ├── calc_score.py
│   ├── metrics.py
│   ├── model.py
│   ├── model_linear.py
│   ├── model_nn.py
│   ├── model_xgb.py
│   ├── run.py
│   ├── runner.py
│   ├── runner_project.py
│   ├── runner_setting.py
│   └── util.py
└── tests → テストを記載するフォルダ(pytest形式)
```

## Dev関連

## モデルの追加方法

### 1. モデルクラスの作成

はじめに、モデルクラスを作成する。

モデルクラスは、XGBoostやscikit-learnやlightGBMなどのモデルをラップしたクラスであり、それぞれのパッケージのインターフェースの違いを吸収するためにこのようにしている。このモデルクラスを使って、学習や予測を行う。

モデルの抽象クラス`AbsModel`を継承して、実際に学習を行うそれぞれのモデルクラスを作成するのが基本だが、単純にsklearnのモデルのラッパークラスを作るだけなら`BaseSklearnModel`クラスを継承するようにしている。(ほとんど処理が同じため)

sklearn以外のパッケージを使用したり、より複雑な処理を記述したい時には、`AbsModel`を継承して各処理を記述するようにし、単純にsklearnのモデルのラッパークラスを作るだけなら`BaseSklearnModel`クラスを継承して、`__init__`の```self._model_class```を使用したいsklearnのクラスに置き換えるのが良い。

モデルは、`./src/models/`内に作成することにしている。

## ドキュメントの作成

ドキュメントをビルドしてHTMLファイルを作成するには、下記のコマンドを使う。

```shell script
sphinx-build doc/source doc/build
```

パッケージにモジュールを追加した場合には、`./doc/source/src.rst`に追加する。
