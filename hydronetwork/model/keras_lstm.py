# keras官方LSTM的实现
from keras import Model, layers


class KerasLSTM(Model):
    """
    keras官方LSTM的实现
    只支持预测1个时间步的结果

    :param units: LSTM单元数量
    :param lookback: 历史时间步数
    """

    def __init__(self,
                 units: int = 128,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.lstm = layers.LSTM(units)
        self.dense = layers.Dense(1)

    def call(self,
             inputs,  # [batch_size, lookback, num_features]
             ):
        lstm_output = self.lstm(inputs)
        return self.dense(lstm_output)
