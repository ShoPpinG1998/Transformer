# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines Transformer model parameters."""

from collections import defaultdict


BASE_PARAMS = defaultdict(
    lambda: None,  # Set default value to None.

    # Input params
    default_batch_size=2048,  # Maximum number of tokens per batch of examples.每个batch的size
    default_batch_size_tpu=32768,   # TPU上每个batch的size
    max_length=256,  # Maximum number of tokens per example.每个例子（句子）的最大长度

    # Model params 模型参数
    initializer_gain=1.0,  # Used in trainable variable initialization. 初始化gain(猜测是loss步长)
    vocab_size=33708,  # Number of tokens defined in the vocabulary file. 子词词库文件的词数
    hidden_size=512,  # Model dimension in the hidden layers.   每层的隐状态维度
    num_hidden_layers=6,  # Number of layers in the encoder and decoder stacks. 模型层数
    num_heads=8,  # Number of heads to use in multi-headed attention.   multi-head attention的head数量,并行8个注意力头
    filter_size=2048,  # Inner layer dimension in the feedforward network.  feedforward层中的状态维度

    # Dropout values (only used when training)训练时随机丢弃的神经元比例
    layer_postprocess_dropout=0.1,  # 层预处理比例0.1
    attention_dropout=0.1,          # attention比例0.1
    relu_dropout=0.1,               # RELU比例0.1

    # Training params   一些训练的超参
    label_smoothing=0.1,            # 标签平滑值0.1
    learning_rate=2.0,              # 初始学习率2
    learning_rate_decay_rate=1.0,   # 学习率逐渐减少的值1
    learning_rate_warmup_steps=16000,   # 学习率衰减总次数

    # Optimizer params 优化参数
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

    # Default prediction params 默认预测参数
    extra_decode_length=50,     # ？？？额外解码长度什么东西
    beam_size=4,                # beam search 尺寸
    alpha=0.6,  # used to calculate length normalization in beam search  beam search归一化的α

    # TPU specific parameters 使用TPU的参数
    use_tpu=False,          # 是否使用TPU
    static_batch=False,     # batch大小是否固定不变
    allow_ffn_pad=True,     # 是否允许对feedforward层进行padding填充
)

BIG_PARAMS = BASE_PARAMS.copy() #Big的配置只有模型的尺寸区别
BIG_PARAMS.update(
    default_batch_size=4096,

    # default batch size is smaller than for BASE_PARAMS due to memory limits.
    default_batch_size_tpu=16384,

    hidden_size=1024,
    filter_size=4096,
    num_heads=16,
)

# Parameters for running the model in multi gpu. These should not change the
# params that modify the model shape (such as the hidden_size or num_heads).
# 使用多GPU只涉及warmup次数的不同，big还涉及dropout
BASE_MULTI_GPU_PARAMS = BASE_PARAMS.copy()
BASE_MULTI_GPU_PARAMS.update(
    learning_rate_warmup_steps=8000
)

BIG_MULTI_GPU_PARAMS = BIG_PARAMS.copy()
BIG_MULTI_GPU_PARAMS.update(
    layer_postprocess_dropout=0.3,
    learning_rate_warmup_steps=8000
)

# Parameters for testing the model
TINY_PARAMS = BASE_PARAMS.copy()
TINY_PARAMS.update(
    default_batch_size=1024,
    default_batch_size_tpu=1024,
    hidden_size=32,
    num_heads=4,
    filter_size=256,
)
