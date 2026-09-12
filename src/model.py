"""CNN + BiLSTM 检测模型（以及用于消融实验的三个对照模型）。

网络结构（默认 ``cnn_bilstm``，即本项目的参赛主模型）::

    输入 (T, F)  —— T 条连续流记录 × F 维流特征
      │
      ├─ CNN 分支：Conv1D(64,k=3) → BN → ReLU → MaxPool → Conv1D(128,k=3) → BN → ReLU
      │            作用：提取"相邻若干条流的局部组合模式"（突发 SYN、等间隔心跳……）
      │
      ├─ 时序分支：BiLSTM(64, return_sequences=True)
      │            作用：对窗口内的时间依赖双向建模（攻击往往有"起始—持续—结束"演化）
      │
      ├─ 注意力池化：Softmax 权重对时间步加权求和
      │            作用：① 让模型聚焦真正可疑的时间步；② 答辩时给出可解释热力图
      │
      └─ 双头输出：多分类头 softmax(K 类) + 二分类头 sigmoid(正常/异常)
                 作用：多任务联合训练，二分类头直接服务于"异常检测"这一最终目标

相比原始 demo 的修正
--------------------
* 原代码先按多分类编译，随后又用 ``BinaryCrossentropy`` 覆盖编译（多分类 one-hot 配二分类损失，
  训练目标与任务不一致）；这里多分类用 sparse categorical CE、二分类用 binary CE，各自正确。
* 原代码把 ``Dense(1280) → Reshape((10, 128))`` 写死，只对 77 维特征有效；
  这里所有维度都由输入张量推导，换数据集/换特征数不用改一行代码。
* 原代码 ``ReduceLROnPlateau(min_lr=0.001)`` 大于初始学习率 ``5e-4``，学习率永远不会下降；
  这里 min_lr 由配置给出且一定小于初始值（见 train.py）。
"""
from __future__ import annotations

from typing import Any

from .config import Config
from .utils import get_logger

LOGGER = get_logger("cida.model")

#: 可用模型变体。``cnn_bilstm`` 是本项目主模型；其余都是**对照/消融**用的单分支或经典结构：
#:   * ``cnn`` / ``bilstm`` / ``mlp`` —— 结构消融（证明双分支融合的必要性）；
#:   * ``rnn`` / ``lstm`` / ``transformer`` —— 与已发表工作对标（PLO-BiLSTM 一文的对照组就是
#:     RNN / LSTM / BiLSTM / Transformer，见 competition/02 的「与已发表方法对标」一节）。
#: 所有变体共用同一套输入、同一套训练与评估口径，因此数字之间可以直接比。
VARIANTS = ("cnn_bilstm", "cnn", "bilstm", "mlp", "rnn", "lstm", "transformer")


# --------------------------------------------------------------------------------------
# 内部构件
# --------------------------------------------------------------------------------------
def _reduce_sum(t, axis: int):
    import tensorflow as tf

    return tf.reduce_sum(t, axis=axis)


_SUM_POOL_CLS: Any = None
_POS_ENC_CLS: Any = None


def _pos_enc_class():
    """正弦位置编码层（延迟构建：不在模块导入期拉进 TensorFlow）。"""
    global _POS_ENC_CLS
    if _POS_ENC_CLS is None:
        import numpy as np
        from tensorflow.keras import initializers, layers

        class PositionalEncoding(layers.Layer):
            def __init__(self, seq_len: int = 16, d_model: int = 64, **kwargs):
                super().__init__(**kwargs)
                self.seq_len = int(seq_len)
                self.d_model = int(d_model)

            def build(self, input_shape):
                pos = np.arange(self.seq_len)[:, None]
                i = np.arange(self.d_model)[None, :]
                angle = pos / np.power(10000.0, (2 * (i // 2)) / max(self.d_model, 1))
                enc = np.where(i % 2 == 0, np.sin(angle), np.cos(angle)).astype("float32")
                self.enc = self.add_weight(name="enc", shape=(1, self.seq_len, self.d_model),
                                           initializer=initializers.Constant(enc), trainable=False)
                super().build(input_shape)

            def call(self, t):
                return t + self.enc

            def compute_output_shape(self, input_shape):
                return input_shape

            def get_config(self):
                cfg = super().get_config()
                cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
                return cfg

        PositionalEncoding.__name__ = "PositionalEncoding"
        _POS_ENC_CLS = PositionalEncoding
    return _POS_ENC_CLS


def sum_pooling_class():
    """在时间维求和的小层（用自定义层替代 Lambda，保证模型可无损保存/加载）。"""
    global _SUM_POOL_CLS
    if _SUM_POOL_CLS is None:
        from tensorflow.keras import layers

        class SumPooling1D(layers.Layer):
            def call(self, t):
                return _reduce_sum(t, axis=1)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[2])

            def get_config(self):
                return super().get_config()

        SumPooling1D.__name__ = "SumPooling1D"
        _SUM_POOL_CLS = SumPooling1D
    return _SUM_POOL_CLS


def make_sum_pooling_layer(name: str = "sum_pool"):
    return sum_pooling_class()(name=name)


def custom_objects() -> dict[str, Any]:
    """加载模型时需要传入的自定义层（延迟构建，避免无 TF 环境导入报错）。"""
    return {"SumPooling1D": sum_pooling_class(), "PositionalEncoding": _pos_enc_class()}


def _conv_block(x, filters: int, kernel: int, l2, *, pool: bool, dropout: float, idx: int,
                dilation: int = 1):
    """一层「卷积 + BN + 激活 (+池化)」。

    ``dilation``：膨胀卷积。对时序分支很重要——**池化会把时间步压没**，
    用膨胀卷积可以在不缩短序列的前提下扩大感受野。
    """
    from tensorflow.keras import layers, regularizers

    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    x = layers.Conv1D(
        filters, kernel, padding="same", dilation_rate=int(dilation),
        kernel_regularizer=reg, name=f"conv1d_{idx}",
    )(x)
    x = layers.BatchNormalization(name=f"bn_conv_{idx}")(x)
    x = layers.Activation("relu", name=f"relu_conv_{idx}")(x)
    if pool:
        x = layers.MaxPooling1D(pool_size=2, name=f"pool_{idx}")(x)
    if dropout > 0:
        x = layers.Dropout(dropout * 0.5, name=f"drop_conv_{idx}")(x)
    return x


def _positional_encoding(t, seq_len: int, d_model: int):
    """把正弦位置编码加到张量 ``t`` 上（返回张量，不是层）。

    为什么不用 Lambda 包一个常量：本项目要求模型能无损保存/加载（见 ``custom_objects``），
    而 Lambda 序列化会带上闭包。为什么不用可训练的位置 Embedding：窗口只有 T=16 步，
    可训练位置向量会和分类任务一起过拟合，而正弦编码是确定性的、更贴近原始 Transformer。
    """
    return _pos_enc_class()(int(seq_len), int(d_model), name="pos_encoding")(t)


def _attention_pool(x):
    """时间维注意力池化：返回 (池化结果, 注意力权重张量)。

    只用 Keras 内置层 + 一个可序列化的小层，因此模型能干净地保存与重新加载。
    """
    from tensorflow.keras import layers

    score = layers.Dense(1, activation="tanh", name="attn_score")(x)      # (B, T, 1)
    weights = layers.Softmax(axis=1, name="attn_weights")(score)          # (B, T, 1)
    weighted = layers.Multiply(name="attn_weighted")([x, weights])
    pooled = make_sum_pooling_layer("attn_pool")(weighted)
    return pooled, weights


def _dense_head(x, units, l2, dropout: float, batch_norm: bool, *, prefix: str = "head"):
    """全连接分类头。"""
    from tensorflow.keras import layers, regularizers

    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    for i, u in enumerate(units):
        x = layers.Dense(int(u), kernel_regularizer=reg, name=f"{prefix}_dense_{i}")(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"{prefix}_bn_{i}")(x)
        x = layers.Activation("relu", name=f"{prefix}_relu_{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"{prefix}_drop_{i}")(x)
    return x


# --------------------------------------------------------------------------------------
# 建模主入口
# --------------------------------------------------------------------------------------
def build_model(
    input_shape: tuple,
    n_classes: int,
    cfg: Config,
    *,
    variant: str | None = None,
    learning_rate: float | None = None,
    use_binary_head: bool | None = None,
    compile_model: bool = True,
):
    """构建模型。

    Parameters
    ----------
    input_shape : (T, F)
        ``sequence`` 模式为 (窗口长度, 特征数)；``sample`` 模式为 (特征数, 1)。
    n_classes : 多分类类别数
    variant   : cnn_bilstm / cnn / bilstm / mlp（后三者用于消融实验）
    """
    from tensorflow.keras import layers, models, optimizers

    variant = (variant or cfg.model_variant).lower()
    if variant not in VARIANTS:
        raise ValueError(f"未知模型变体 {variant}，可选 {VARIANTS}")
    use_binary_head = (cfg.binary_loss_weight > 0) if use_binary_head is None else use_binary_head
    lr = float(learning_rate if learning_rate is not None else cfg.learning_rate)

    seq_len = int(input_shape[0])
    n_feat = int(input_shape[1])
    inputs = layers.Input(shape=(seq_len, n_feat), name="flow_window")

    # ---------------- 特征提取主干 ----------------
    attn_weights = None
    x = inputs
    keep_sequence = bool(getattr(cfg, "keep_sequence_in_hybrid", True))

    if variant == "cnn_bilstm":
        # ============ 双分支并行混合（本项目的主模型）============
        # 分支 A：CNN —— 沿时间轴做膨胀卷积，抓"相邻若干条流"的局部组合模式；
        # 分支 B：BiLSTM —— 直接在原始序列上做双向建模，抓整段窗口的演化趋势。
        # 两路各自注意力池化后拼接，让网络自己决定更信任哪一路。
        # （比"CNN 输出再喂 LSTM"的串行做法更稳：串行时卷积的池化/压缩会损失时序信息，
        #   实测串行版本宏 F1 反而低于单独任一分支，详见 docs/02 的消融讨论）
        cnn_x = x
        for i, filters in enumerate(cfg.conv_filters):
            cnn_x = _conv_block(
                cnn_x, int(filters), int(cfg.conv_kernel), cfg.l2_reg,
                pool=False, dropout=cfg.dropout, idx=i, dilation=2 ** i,
            )
        cnn_h = layers.GlobalMaxPooling1D(name="cnn_pool")(cnn_x)

        rnn_seq = layers.Bidirectional(
            layers.LSTM(int(cfg.lstm_units), return_sequences=True), name="bi_lstm"
        )(x)
        if cfg.use_attention:
            rnn_h, attn_weights = _attention_pool(rnn_seq)
        else:
            rnn_h = layers.GlobalMaxPooling1D(name="rnn_pool")(rnn_seq)

        trunk = layers.Concatenate(name="fuse")([cnn_h, rnn_h])
    elif variant == "cnn":
        cur_len = seq_len
        for i, filters in enumerate(cfg.conv_filters):
            do_pool = (not keep_sequence) and (cur_len // 2) >= 2
            x = _conv_block(
                x, int(filters), int(cfg.conv_kernel), cfg.l2_reg,
                pool=do_pool, dropout=cfg.dropout, idx=i, dilation=1,
            )
            if do_pool:
                cur_len //= 2
        trunk = layers.GlobalMaxPooling1D(name="cnn_pool")(x)
    elif variant == "bilstm":
        trunk = layers.Bidirectional(
            layers.LSTM(int(cfg.lstm_units), return_sequences=True), name="bi_lstm"
        )(x)
        if cfg.use_attention:
            trunk, attn_weights = _attention_pool(trunk)
        else:
            trunk = layers.GlobalMaxPooling1D(name="global_max_pool")(trunk)
    elif variant == "mlp":
        flat = layers.Flatten(name="flatten")(x)
        trunk = _dense_head(
            flat, [int(cfg.dense_units[0])], cfg.l2_reg, cfg.dropout, cfg.batch_norm, prefix="mlp"
        )
    elif variant == "rnn":            # 对标用：朴素双向 RNN（无门控，长依赖能力弱）
        trunk = layers.Bidirectional(
            layers.SimpleRNN(int(cfg.lstm_units), return_sequences=True), name="bi_rnn"
        )(x)
        if cfg.use_attention:
            trunk, attn_weights = _attention_pool(trunk)
        else:
            trunk = layers.GlobalMaxPooling1D(name="global_max_pool")(trunk)
    elif variant == "lstm":           # 对标用：**单向** LSTM（论文的 LSTM 基线）
        trunk = layers.LSTM(int(cfg.lstm_units), return_sequences=True, name="uni_lstm")(x)
        if cfg.use_attention:
            trunk, attn_weights = _attention_pool(trunk)
        else:
            trunk = layers.GlobalMaxPooling1D(name="global_max_pool")(trunk)
    elif variant == "transformer":    # 对标用：2 层 Transformer 编码器（自注意力 + 位置编码）
        # 输入只有 T=16 个时间步，因此用一个小型编码器即可；d_model 取 dense_units[0] 以保证
        # 参数量与 BiLSTM 分支同量级（否则"Transformer 更强/更弱"会被参数量差异污染）。
        d_model = int(cfg.dense_units[0])
        h = layers.Dense(d_model, name="tf_proj")(x)
        h = _positional_encoding(h, seq_len, d_model)
        for i in range(2):
            attn = layers.MultiHeadAttention(
                num_heads=4, key_dim=max(d_model // 4, 1), dropout=cfg.dropout, name=f"tf_mha_{i}"
            )(h, h)
            h = layers.LayerNormalization(epsilon=1e-6, name=f"tf_ln1_{i}")(
                layers.Add(name=f"tf_res1_{i}")([h, attn])
            )
            ff = layers.Dense(d_model * 2, activation="relu", name=f"tf_ff1_{i}")(h)
            ff = layers.Dense(d_model, name=f"tf_ff2_{i}")(ff)
            h = layers.LayerNormalization(epsilon=1e-6, name=f"tf_ln2_{i}")(
                layers.Add(name=f"tf_res2_{i}")([h, ff])
            )
        trunk = layers.GlobalAveragePooling1D(name="tf_pool")(h)
    else:  # pragma: no cover
        raise ValueError(variant)

    # ---------------- 分类头 ----------------
    head = _dense_head(trunk, cfg.dense_units, cfg.l2_reg, cfg.dropout, cfg.batch_norm, prefix="clf")
    outputs: dict[str, Any] = {
        "mc": layers.Dense(n_classes, activation="softmax", name="mc")(head)
    }
    if use_binary_head:
        outputs["bin"] = layers.Dense(1, activation="sigmoid", name="bin")(head)

    model = models.Model(
        inputs=inputs,
        outputs=outputs if len(outputs) > 1 else outputs["mc"],
        name=f"cnn_bilstm_{variant}",
    )

    if compile_model:
        opt = optimizers.Adam(learning_rate=lr)
        if len(outputs) > 1:
            model.compile(
                optimizer=opt,
                loss={"mc": "sparse_categorical_crossentropy", "bin": "binary_crossentropy"},
                loss_weights={"mc": 1.0, "bin": float(cfg.binary_loss_weight)},
                metrics={"mc": ["accuracy"], "bin": ["accuracy"]},
            )
        else:
            model.compile(
                optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )

    LOGGER.info(
        "模型 %-11s 输入=%s 类别=%d 参数量=%s 注意力=%s",
        variant, input_shape, n_classes, f"{int(model.count_params()):,}",
        "开" if attn_weights is not None else "关",
    )
    model.cida_has_attention = attn_weights is not None
    return model


def build_explain_model(model):
    """构建输出注意力权重的辅助模型（用于可解释性可视化）；无注意力层时返回 None。

    注意：模型从磁盘重新加载后，构建期挂上的 ``cida_has_attention`` 属性会丢失，
    因此这里以"能不能找到 attn_weights 层"为准，而不是只看那个属性。
    """
    if not hasattr(model, "get_layer"):
        # 非 Keras 模型（例如传统 ML 基线）：直接跳过，避免在没有 TF 的环境里导入失败
        return None
    if getattr(model, "cida_has_attention", True) is False:
        return None
    try:
        from tensorflow.keras import models

        return models.Model(
            inputs=model.input,
            outputs=model.get_layer("attn_weights").output,
            name="attention_explain",
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("注意力解释模型不可用：%s", exc)
        return None


def predict_probs(model, X, batch_size: int = 512):
    """兼容单/双输出模型的概率提取，返回 (多分类概率 (N,C), 异常概率 (N,))。"""
    import numpy as np

    out = model.predict(X, batch_size=batch_size, verbose=0)
    if isinstance(out, dict):
        p_mc = np.asarray(out["mc"])
        p_bin = np.asarray(out["bin"]).reshape(-1) if "bin" in out else 1.0 - p_mc[:, 0]
    elif isinstance(out, (list, tuple)):
        p_mc = np.asarray(out[0])
        p_bin = np.asarray(out[1]).reshape(-1) if len(out) > 1 else 1.0 - p_mc[:, 0]
    else:
        p_mc = np.asarray(out)
        p_bin = 1.0 - p_mc[:, 0]
    return p_mc, np.clip(p_bin, 1e-7, 1.0 - 1e-7)


def one_hot(y, n_classes: int):
    import numpy as np

    return np.eye(n_classes, dtype="float32")[np.asarray(y, dtype="int64").reshape(-1)]


def load_model_bundle(path, compile_model: bool = False):
    """加载训练好的模型（自动带上自定义层）。"""
    from tensorflow.keras import models

    return models.load_model(path, custom_objects=custom_objects(), compile=compile_model)
