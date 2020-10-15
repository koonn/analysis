#!/usr/bin/env zsh

# shの絶対パスを取得. これによって、どこから実行しても同じように実行できる
SCRIPT_DIR="$(cd "$(dirname $0)"; pwd)"

# 実行
#python "$SCRIPT_DIR/../src/train.py" --fold 1 --model_name auto_encoder
#python "$SCRIPT_DIR/../src/train.py" --fold 1 --model_name svm_rbf
python "$SCRIPT_DIR/../src/train.py" --fold 1 --model_name bernoulli_nb
python "$SCRIPT_DIR/../src/train.py" --fold 1 --model_name gaussian_nb


#python "$SCRIPT_DIR/../src/train.py" --fold 1 --model_name logistic_regression
#python "$SCRIPT_DIR/../src/train.py" --fold 0 --model_name logistic_regression
#python "$SCRIPT_DIR/../src/train.py" --fold 1 --model_name logistic_regression
#python "$SCRIPT_DIR/../src/train.py" --fold 2 --model_name logistic_regression
#python "$SCRIPT_DIR/../src/train.py" --fold 3 --model_name logistic_regression
#python "$SCRIPT_DIR/../src/train.py" --fold 4 --model_name logistic_regression

#python "$SCRIPT_DIR/../src/train.py" --fold 0 --model_name random_forest
#python "$SCRIPT_DIR/../src/train.py" --fold 0 --model_name decision_tree_gini
#python "$SCRIPT_DIR/../src/train.py" --fold 0 --model_name decision_tree_entropy
#python "$SCRIPT_DIR/../src/train.py" --fold 0 --model_name xgboost
#python "$SCRIPT_DIR/../src/train.py" --fold 1 --model_name lightgbm

# 警告を表示したくない時
# python -W ignore train.py --fold 0 --model_name logistic_regression