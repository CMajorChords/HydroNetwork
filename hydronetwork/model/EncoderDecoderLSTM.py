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


class EncoderDecoderLSTM(keras.Model):
    """
    Encoder-Decoder LSTM模型。其中Encoder和Decoder均为单层LSTM，通过Dense层进行状态转换和预测。
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
        self.dense_transform_h = keras.layers.Dense(encoder_hidden_units)
        self.dense_transform_c = keras.layers.Dense(encoder_hidden_units)
        dense_projection = keras.Sequential([
            keras.layers.Dense(int(0.5 * decoder_hidden_units), activation='relu'),
            keras.layers.Dense(1)
        ])
        self.time_distributed_dense = keras.layers.TimeDistributed(dense_projection)

    def call(self, x):
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
        y = keras.ops.squeeze(y, axis=-1)
        return y
