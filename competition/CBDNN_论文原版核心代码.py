# -*- coding: utf-8 -*-
"""
CBDNN: CNN + BiLSTM 入侵检测模型（核心代码）

模型：CBDNN（Convolutional Neural Network + Bidirectional LSTM）
用途：软件定义网络（SDN）环境下的入侵检测与多分类
原理：CNN 提取网络流量的空间局部特征，BiLSTM 捕获流量时序依赖

主要参考文献：
  [1] Jia Y, Ren J, He H, Li X, Xu K. CBDNN: An efficient and generalized
      intrusion detection method in SDN environments.
      Cluster Computing, 2026, 29: 263. DOI: 10.1007/s10586-026-06083-z
  [2] Elsayed M S, Le-Khac N A, Jurcut A D. InSDN: A novel SDN intrusion
      dataset. IEEE Access, 2020, 8: 165263-165284.
      DOI: 10.1109/ACCESS.2020.3022633

隐私说明：
  本文件为算法核心代码的精简与隐私处理版本，原代码中的个人本机绝对路径
  已替换为可配置的相对路径（见下方配置区），便于复现与提交。
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, Activation,
    Dropout, BatchNormalization, Reshape, LSTM, Bidirectional,
    GlobalMaxPooling1D,
)
from tensorflow.keras import callbacks

# ------------------------- 可配置参数（隐私处理：已去除个人本机路径） -------------------------
DATASET_DIR = os.path.join("dataset", "inSDN")                # 数据集目录，请按实际位置修改
MODEL_SAVE_PATH = os.path.join("model", "model_InSDN.h5")     # 模型保存路径
TEST_SIZE = 0.25
RANDOM_STATE = 2
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.2

# 待删除的无用列
USELESS_COLUMNS = ["Flow ID", "Timestamp", "Src IP", "Dst IP", "Src Port", "Dst Port"]
# 特征维度（InSDN 数据集为 77 维特征）
N_FEATURES = 77


def build_cbdnn(features, n_classes):
    """构建 CBDNN（CNN + BiLSTM）模型。"""
    input_layer = Input(shape=(features, 1))

    # ---- CNN 段：提取空间特征 ----
    x = Conv1D(filters=128, kernel_size=6, padding="same", activation="relu")(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=256, kernel_size=6, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    # ---- 维度适配：展平后重构为时序序列 ----
    x = Flatten()(x)
    x = Dense(1280)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Reshape((10, 128))(x)

    # ---- BiLSTM 段：提取时序特征 ----
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)

    # ---- 分类输出 ----
    x = Dense(256, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(192, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(DROPOUT_RATE)(x)
    output_layer = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def load_and_preprocess(dataset_dir):
    """读取数据集并完成预处理。"""
    csv_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
    df = pd.concat(
        [pd.read_csv(os.path.join(dataset_dir, f), low_memory=False) for f in csv_files],
        axis=0, ignore_index=True,
    )

    df.drop(labels=USELESS_COLUMNS, axis="columns", inplace=True, errors="ignore")
    df = df.dropna().drop_duplicates(keep="first").reset_index(drop=True)

    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])

    data_np = df.to_numpy(dtype="float32")
    data_np = data_np[~np.isinf(data_np).any(axis=1)]

    X = data_np[:, :N_FEATURES]
    y = OneHotEncoder().fit_transform(data_np[:, N_FEATURES:]).toarray()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    return X_train, X_test, Y_train, Y_test, X.shape[1], y.shape[1]


def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    X_train, X_test, Y_train, Y_test, features, n_classes = load_and_preprocess(DATASET_DIR)

    model = build_cbdnn(features, n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor="loss", patience=5)
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, min_lr=0.001
    )

    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[reduce_lr, early_stop],
        validation_data=(X_test, Y_test),
        verbose=1,
    )
    model.save(MODEL_SAVE_PATH)

    # ---- 评估 ----
    pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(Y_test, axis=1)

    acc = metrics.accuracy_score(y_true, pred)
    precision = metrics.precision_score(y_true, pred, average="weighted")
    recall = metrics.recall_score(y_true, pred, average="weighted")
    f1 = metrics.f1_score(y_true, pred, average="weighted")

    print("Accuracy : {:.4f}".format(acc))
    print("Precision: {:.4f}".format(precision))
    print("Recall   : {:.4f}".format(recall))
    print("F1-score : {:.4f}".format(f1))
    print("Confusion Matrix:\n", metrics.confusion_matrix(y_true, pred))


if __name__ == "__main__":
    main()