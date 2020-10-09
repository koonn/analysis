#!/usr/bin/env zsh

# shの絶対パスを取得. これによって、どこから実行しても同じように実行できる
SCRIPT_DIR="$(cd "$(dirname $0)"; pwd)"

# 実行
python "$SCRIPT_DIR/train.py" --fold 0 --model logistic_regression
python "$SCRIPT_DIR/train.py" --fold 1 --model logistic_regression
python "$SCRIPT_DIR/train.py" --fold 2 --model logistic_regression
python "$SCRIPT_DIR/train.py" --fold 3 --model logistic_regression
python "$SCRIPT_DIR/train.py" --fold 4 --model logistic_regression


# 警告を表示したくない時
# python -W ignore train.py --fold 0 --model logistic_regression