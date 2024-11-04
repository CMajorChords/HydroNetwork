# LSTM类模型的实现
import keras


class Encoder(keras.Model):
    def __init__(self, hidden_units):
        super(Encoder, self).__init__()
        self.lstm = keras.layers.LSTM(units=hidden_units, return_state=True)

    def call(self, x):
        _, state_h, state_c = self.lstm(x)
        return state_h, state_c


class Decoder(keras.Model):
    def __init__(self, hidden_units):
        super(Decoder, self).__init__()
        self.lstm = keras.layers.LSTM(hidden_units, return_sequences=True)

    def call(self, x, state_h, state_c):
        x = self.lstm(x, initial_state=(state_h, state_c))
        return x


class EncoderDecoderFeaturesLookbackBidirectional(keras.Model):
    """
    Encoder-Decoder LSTM模型。其中Encoder和Decoder均为单层LSTM，通过Dense层进行状态转换和预测。
    适用于lookback与bidirectional输入的情况。
    :param lookback: 历史时间步长
    :param horizon: 预测的时间步长
    :param encoder_hidden_units: Encoder LSTM隐藏层单元数
    :param decoder_hidden_units: Decoder LSTM隐藏层单元数
    :param kwargs: 其他keras.Model参数
    """

    def __init__(self,
                 lookback: int,
                 horizon: int,
                 encoder_hidden_units: int = 128,
                 decoder_hidden_units: int = 128,
                 **kwargs
                 ):
        super().__init__(name="EncoderDecoderLSTM", **kwargs)
        self.lookback = lookback
        self.horizon = horizon
        self.encoder = Encoder(encoder_hidden_units)
        self.decoder = Decoder(decoder_hidden_units)
        self.dense_transform_h = keras.Sequential([
            keras.layers.Dense(encoder_hidden_units),
            keras.layers.Dense(decoder_hidden_units)
        ])
        self.dense_transform_c = keras.Sequential([
            keras.layers.Dense(encoder_hidden_units),
            keras.layers.Dense(decoder_hidden_units)
        ])
        dense_projection = keras.Sequential([
            keras.layers.Dense(int(0.5 * decoder_hidden_units), activation='leaky_relu'),
            keras.layers.Dense(int(0.25 * decoder_hidden_units)),
            keras.layers.Dense(1)
        ])
        self.time_distributed_dense = keras.layers.TimeDistributed(dense_projection)

    def call(self, x):
        """
        当有lookback与bidirectional输入时的前向传播
        """
        # 将x根据时间步长分割为encoder_input和decoder_input，维度为(batch_size, time_steps, features)
        x_lookback, x_bidirectional = x
        x_lookback = keras.ops.expand_dims(x_lookback, axis=-1)
        encoder_input = x_bidirectional[:, :self.lookback, :]
        decoder_input = x_bidirectional[:, self.lookback:, :]
        # x_lookback的维度是(batch_size, lookback, features)
        # encoder_input的维度是(batch_size, lookback, features)
        # 将x_lookback拼接到encoder_input的最后一个维度，维度变为(batch_size, lookback，features + lookback_features)
        encoder_input = keras.layers.concatenate([encoder_input, x_lookback], axis=-1)
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
        y = keras.ops.squeeze(y, axis=-1)
        return y


class EncoderDecoderLSTMFeaturesBidirectional(keras.Model):
    """
    Encoder-Decoder LSTM模型。其中Encoder和Decoder均为单层LSTM，通过Dense层进行状态转换和预测。
    适用于bidirectional输入的情况。

    :param lookback: 历史时间步长
    :param horizon: 预测的时间步长
    :param encoder_hidden_units: Encoder LSTM隐藏层单元数
    :param decoder_hidden_units: Decoder LSTM隐藏层单元数
    :param kwargs: 其他keras.Model参数
    """

    def __init__(self,
                 lookback: int,
                 horizon: int,
                 encoder_hidden_units: int = 128,
                 decoder_hidden_units: int = 128,
                 **kwargs
                 ):
        super().__init__(name="EncoderDecoderLSTM", **kwargs)
        self.lookback = lookback
        self.horizon = horizon
        self.encoder = Encoder(encoder_hidden_units)
        self.decoder = Decoder(decoder_hidden_units)
        self.dense_transform_h = keras.Sequential([
            keras.layers.Dense(encoder_hidden_units),
            keras.layers.Dense(decoder_hidden_units)
        ])
        self.dense_transform_c = keras.Sequential([
            keras.layers.Dense(encoder_hidden_units),
            keras.layers.Dense(decoder_hidden_units)
        ])
        dense_projection = keras.Sequential([
            keras.layers.Dense(int(0.5 * decoder_hidden_units), activation='leaky_relu'),
            keras.layers.Dense(int(0.25 * decoder_hidden_units)),
            keras.layers.Dense(1)
        ])
        self.time_distributed_dense = keras.layers.TimeDistributed(dense_projection)

    def call(self, x):
        """
        当只有bidirectional输入时的前向传播
        """
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
        y = keras.ops.squeeze(y, axis=-1)
        return y


def encoder_decoder_lstm(lookback: int,
                         horizon: int,
                         features_type,
                         encoder_hidden_units: int = 128,
                         decoder_hidden_units: int = 128,
                         **kwargs):
    """
    根据features_type返回对应的Encoder-Decoder LSTM模型
    :param lookback: 历史时间步长
    :param horizon: 预测的时间步长
    :param features_type: 输入模型的特征类型，支持{'bidirectional', 'lookback_bidirectional'}和{'bidirectional'}
    :param encoder_hidden_units: Encoder LSTM隐藏层单元数
    :param decoder_hidden_units: Decoder LSTM隐藏层单元数
    :param kwargs: 其他keras.Model参数
    :return: Encoder-Decoder LSTM模型
    """
    if set(features_type) == {'bidirectional'}:
        return EncoderDecoderLSTMFeaturesBidirectional(lookback=lookback,
                                                       horizon=horizon,
                                                       encoder_hidden_units=encoder_hidden_units,
                                                       decoder_hidden_units=decoder_hidden_units,
                                                       **kwargs)
    elif set(features_type) == {'lookback', 'bidirectional'}:
        return EncoderDecoderFeaturesLookbackBidirectional(lookback=lookback,
                                                           horizon=horizon,
                                                           encoder_hidden_units=encoder_hidden_units,
                                                           decoder_hidden_units=decoder_hidden_units,
                                                           **kwargs)
    else:
        raise ValueError(f"features_type={features_type} is not supported.")
