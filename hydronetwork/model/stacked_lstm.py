from hydronetwork.model.encoder_decoder_lstm import Encoder, Decoder
from keras import ops, Model, Layer, layers, Sequential


class StackedEncoder(Layer):
    def __init__(self,
                 hidden_units,
                 ):
        super(StackedEncoder, self).__init__()
        self.lstm = layers.LSTM(units=hidden_units, return_state=True, return_sequences=True)

    def call(self, x):
        # output [batch_size, T, units]
        # state_h [batch_size, units]
        # state_c [batch_size, units]
        return self.lstm(x)


class StackedLSTM(Model):
    def __init__(self,
                 lookback: int,
                 horizon: int,
                 units: int = 128,
                 ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        # 产流模型
        self.encoder_runoff_producing = StackedEncoder(units)
        self.decoder_runoff_producing = Decoder(units)
        self.dense_transform_h_runoff_producing = Sequential([
            layers.Dense(units),
            layers.Dense(units)
        ])
        self.dense_transform_c_runoff_producing = Sequential([
            layers.Dense(units),
            layers.Dense(units)
        ])
        self.time_distributed_dense_runoff_producing = layers.TimeDistributed(Sequential([
            layers.Dense(units, activation='relu'),
            layers.Dense(units),
        ]))
        # 汇流模型
        self.encoder_confluence = Encoder(units)
        self.decoder_confluence = Decoder(units)
        self.dense_transform_h_confluence = Sequential([
            layers.Dense(units),
            layers.Dense(units)
        ])
        self.dense_transform_c_confluence = Sequential([
            layers.Dense(units),
            layers.Dense(units)
        ])
        self.time_distributed_dense_confluence = layers.TimeDistributed(Sequential([
            layers.Dense(int(0.5 * units), activation='relu'),
            layers.Dense(int(0.25 * units)),
            layers.Dense(1)
        ]))

    def call(self,
             inputs,
             ):
        lookback_features, bidirectional_features = inputs  # [batch_size, lookback], (batch_size, T, num_features)
        # 计算产流过程
        encoder_inputs = ops.concatenate([lookback_features,
                                          bidirectional_features[:, :self.lookback, :]],
                                         axis=-1)  # [batch_size, lookback, num_features + 1]
        decoder_inputs = bidirectional_features[:, self.lookback:, :]  # [batch_size, horizon, num_features]
        (encoder_output,  # [batch_size, lookback, units]
         state_h,  # [batch_size, units]
         state_c,  # [batch_size, units]
         ) = self.encoder_runoff_producing(encoder_inputs)
        state_h = self.dense_transform_h_runoff_producing(state_h)  # [batch_size, units]
        state_c = self.dense_transform_c_runoff_producing(state_c)  # [batch_size, units]
        decoder_output = self.decoder_runoff_producing(decoder_inputs, state_h, state_c)  # [batch_size, horizon, units]
        prediction_runoff = ops.concatenate([encoder_output, decoder_output],
                                            axis=-2)  # [batch_size, lookback+horizon, units]
        prediction_runoff = self.time_distributed_dense_runoff_producing(
            prediction_runoff)  # [batch_size, lookback+horizon, units]
        # 计算汇流过程
        encoder_inputs = prediction_runoff[:, :self.lookback, :]
        decoder_inputs = prediction_runoff[:, self.lookback:, :]
        (state_h,  # [batch_size, units]
         state_c,  # [batch_size, units]
         ) = self.encoder_confluence(encoder_inputs)
        state_h = self.dense_transform_h_confluence(state_h)  # [batch_size, units]
        state_c = self.dense_transform_c_confluence(state_c)  # [batch_size, units]
        decoder_output = self.decoder_confluence(decoder_inputs, state_h, state_c)  # [batch_size, horizon, units]
        prediction_confluence = self.time_distributed_dense_confluence(decoder_output)  # [batch_size, horizon, 1]
        return prediction_confluence.squeeze()

# %% 测试StackedLSTM
# import numpy as np
# inputs = np.random.rand(512, 372, 6)
# lookback_features = np.random.rand(512, 365)
# model = StackedLSTM(lookback=365, horizon=7)
# prediction = model([lookback_features, inputs])
