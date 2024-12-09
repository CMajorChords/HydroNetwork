# LSTM类模型的实现
from keras import ops, Model, Layer, layers, Sequential


class Encoder(Layer):
    def __init__(self,
                 hidden_units,
                 ):
        super(Encoder, self).__init__()
        self.lstm = layers.LSTM(units=hidden_units, return_state=True)

    def call(self, x):
        _, state_h, state_c = self.lstm(x)
        return state_h, state_c


class Decoder(Layer):
    def __init__(self,
                 hidden_units
                 ):
        super(Decoder, self).__init__()
        self.lstm = layers.LSTM(hidden_units, return_sequences=True)

    def call(self, x, state_h, state_c):
        return self.lstm(x, initial_state=(state_h, state_c))


class EncoderDecoderFeaturesLookbackBidirectional(Model):
    """
    Encoder-Decoder LSTM模型。其中Encoder和Decoder均为单层LSTM，通过Dense层进行状态转换和预测。
    适用于lookback与bidirectional输入的情况。
    :param lookback: 历史时间步长
    :param horizon: 预测的时间步长
    :param encoder_units: Encoder LSTM隐藏层单元数
    :param decoder_units: Decoder LSTM隐藏层单元数
    :param kwargs: 其他Model参数
    """

    def __init__(self,
                 lookback: int,
                 horizon: int,
                 encoder_units: int = 128,
                 decoder_units: int = 128,
                 output_dim: int = 1,
                 **kwargs
                 ):
        super().__init__(name="EncoderDecoderLSTM", **kwargs)
        self.lookback = lookback
        self.horizon = horizon
        self.encoder = Encoder(encoder_units)
        self.decoder = Decoder(decoder_units)
        self.dense_transform_h = Sequential([
            layers.Dense(encoder_units),
            layers.Dense(decoder_units)
        ])
        self.dense_transform_c = Sequential([
            layers.Dense(encoder_units),
            layers.Dense(decoder_units)
        ])
        dense_projection = Sequential([
            layers.Dense(int(0.5 * decoder_units), activation='leaky_relu'),
            layers.Dense(int(0.25 * decoder_units)),
            layers.Dense(output_dim)
        ])
        self.time_distributed_dense = layers.TimeDistributed(dense_projection)

    def call(self, x):
        """
        当有lookback与bidirectional输入时的前向传播
        """
        if len(x) == 2:
            # 将x根据时间步长分割为encoder_input和decoder_input，维度为(batch_size, time_steps, features)
            x_lookback, x_bidirectional = x
            encoder_input = x_bidirectional[:, :self.lookback, :]  # (batch_size, lookback, features)
            encoder_input = ops.concatenate([encoder_input, x_lookback], axis=-1)
            decoder_input = x_bidirectional[:, self.lookback:, :]  # (batch_size, horizon, features)
        elif len(x) == 3:
            # 将x根据时间步长分割为encoder_input和decoder_input，维度为(batch_size, time_steps, features)
            x_lookback, x_bidirectional, attributes = x
            attributes = ops.expand_dims(attributes, axis=1)  # [batch_size, 1, num_static_features]
            # [batch_size, lookback + horizon, num_static_features]
            attributes = ops.tile(attributes, [1, x_bidirectional.shape[1], 1])
            # [batch_size, lookback + horizon, features + num_static_features]
            x_bidirectional = ops.concatenate([x_bidirectional, attributes], axis=-1)
            # (batch_size, lookback, features)
            encoder_input = ops.concatenate([x_bidirectional[:, :self.lookback, :], x_lookback], axis=-1)
            decoder_input = x_bidirectional[:, self.lookback:, :]  # (batch_size, horizon, features)
        else:
            raise ValueError(f"len(x)={len(x)} is not supported.")
        # Encoder LSTM
        state_h, state_c = self.encoder(encoder_input)
        # Dense层转换状态
        state_h = self.dense_transform_h(state_h)
        state_c = self.dense_transform_c(state_c)
        # Decoder LSTM
        decoder_output = self.decoder(decoder_input, state_h, state_c)
        # Dense层投影
        y = self.time_distributed_dense(decoder_output)  # (batch_size, horizon, 1)
        # 丢弃最后一个为1的维度
        y = ops.squeeze(y, axis=-1)
        return y


class EncoderDecoderLSTMFeaturesBidirectional(Model):
    """
    Encoder-Decoder LSTM模型。其中Encoder和Decoder均为单层LSTM，通过Dense层进行状态转换和预测。
    适用于bidirectional输入的情况。

    :param lookback: 历史时间步长
    :param horizon: 预测的时间步长
    :param encoder_units: Encoder LSTM隐藏层单元数
    :param decoder_units: Decoder LSTM隐藏层单元数
    :param kwargs: 其他Model参数
    """

    def __init__(self,
                 lookback: int,
                 horizon: int,
                 encoder_units: int = 128,
                 decoder_units: int = 128,
                 output_dim: int = 1,
                 **kwargs
                 ):
        super().__init__(name="EncoderDecoderLSTM", **kwargs)
        self.lookback = lookback
        self.horizon = horizon
        self.encoder = Encoder(encoder_units)
        self.decoder = Decoder(decoder_units)
        self.dense_transform_h = Sequential([
            layers.Dense(encoder_units),
            layers.Dense(decoder_units)
        ])
        self.dense_transform_c = Sequential([
            layers.Dense(encoder_units),
            layers.Dense(decoder_units)
        ])
        dense_projection = Sequential([
            layers.Dense(int(0.5 * decoder_units), activation='leaky_relu'),
            layers.Dense(int(0.25 * decoder_units)),
            layers.Dense(output_dim)
        ])
        self.time_distributed_dense = layers.TimeDistributed(dense_projection)

    def call(self, x):
        """
        当只有bidirectional输入时的前向传播
        """
        if isinstance(x, tuple) or isinstance(x, list):
            (timeseries,  # [batch_size, lookback + horizon, features]
             attributes,  # [batch_size, num_static_features]
             ) = x
            attributes = ops.expand_dims(attributes, axis=1)  # [batch_size, 1, num_static_features]
            attributes = ops.tile(attributes,
                                  [1, timeseries.shape[1], 1])  # [batch_size, lookback + horizon, num_static_features]
            x = ops.concatenate([timeseries, attributes],
                                axis=-1)  # [batch_size, lookback + horizon, features + num_static_features]
        # 将x根据时间步长分割为encoder_input和decoder_input，维度为(batch_size, time_steps, features)
        encoder_input = x[:, :self.lookback, :]
        decoder_input = x[:, self.lookback:, :]
        # Encoder LSTM
        state_h, state_c = self.encoder(encoder_input)
        # Dense层转换状态
        state_h = self.dense_transform_h(state_h)
        state_c = self.dense_transform_c(state_c)
        # Decoder LSTM
        decoder_output = self.decoder(decoder_input, state_h, state_c)
        # Dense层投影
        y = self.time_distributed_dense(decoder_output)  # (batch_size, horizon, 1)
        # 丢弃最后一个为1的维度
        y = ops.squeeze(y, axis=-1)
        return y


def encoder_decoder_lstm(lookback: int,
                         horizon: int,
                         features_type,
                         units: int = 128,
                         **kwargs):
    """
    根据features_type返回对应的Encoder-Decoder LSTM模型
    :param lookback: 历史时间步长
    :param horizon: 预测的时间步长
    :param features_type: 输入模型的特征类型，支持{'bidirectional', 'lookback'}和{'bidirectional'}
    :param units: LSTM隐藏层单元数
    :param kwargs: 其他Model参数
    :return: Encoder-Decoder LSTM模型
    """
    encoder_units, decoder_units = units, units
    if set(features_type) == {'bidirectional'}:
        return EncoderDecoderLSTMFeaturesBidirectional(lookback=lookback, horizon=horizon,
                                                       encoder_units=encoder_units,
                                                       decoder_units=decoder_units, **kwargs)
    elif set(features_type) == {'lookback', 'bidirectional'}:
        return EncoderDecoderFeaturesLookbackBidirectional(lookback=lookback, horizon=horizon,
                                                           encoder_units=encoder_units,
                                                           decoder_units=decoder_units, **kwargs)
    else:
        raise ValueError(f"features_type={features_type} is not supported.")
