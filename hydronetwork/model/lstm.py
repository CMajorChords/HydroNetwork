# 自己实现的lstml模型
import keras
from keras import Model, layers, Layer, ops
from hydronetwork.model.layer.gate import Gate


class LSTMCell(Layer):
    """
    LSTM cell

    :param units: Number of hidden units
    """

    def __init__(self,
                 units: int,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.units = units
        # Input Gate
        self.input_gate = Gate(units=units, )
        # Forget Gate
        self.forget_gate = Gate(units=units, )
        # Output Gate
        self.output_gate = Gate(units=units, )
        # Cell Update
        self.cell_update = Gate(units=units, activation="tanh")

    def call(self,
             features,  # [batch_size, num_features]
             last_hidden_state,  # [batch_size, units]
             last_cell_state,  # [batch_size, units]
             ):
        # 特征拼接
        features = layers.concatenate([features, last_hidden_state], axis=-1)
        # 输入门
        input_gate = self.input_gate(features)
        # 遗忘门
        forget_gate = self.forget_gate(features)
        # 细胞状态更新
        cell_update = self.cell_update(features)
        cell_state = forget_gate * last_cell_state + input_gate * cell_update
        # 输出门
        output_gate = self.output_gate(features)
        hidden_state = output_gate * keras.activations.tanh(cell_state)
        return hidden_state, cell_state


class LSTM(Model):
    """
    LSTM model，只支持单步流量预测

    :param units: Number of hidden units
    :param return_state: Whether to return the state
    """

    def __init__(self,
                 units: int = 128,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.units = units
        self.lstm_cell = LSTMCell(units=units)
        self.dense = layers.Dense(1)

    def call(self,
             features,  # [batch_size, seq_len, num_features]
             initial_state: tuple = None,  # [last_hidden_state, last_cell_state]
             ):
        batch_size, seq_len, num_features = features.shape
        if initial_state is None:
            last_hidden_state = ops.zeros([batch_size, self.units])
            last_cell_state = ops.zeros([batch_size, self.units])
        else:
            last_hidden_state, last_cell_state = initial_state
        for i in range(seq_len):
            last_hidden_state, last_cell_state = self.lstm_cell(features[:, i],  # [batch_size, num_features]
                                                                last_hidden_state,
                                                                last_cell_state)
        return self.dense(last_hidden_state)
