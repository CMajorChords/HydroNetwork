from hydronetwork.model.layer.gate import Gate
import keras
from keras import ops, Model, Layer, layers, Sequential
from keras.api.backend import epsilon
from keras.api.utils import normalize


@keras.saving.register_keras_serializable(package='Custom', name='MassConservingLSTMCell')
class MassConservingLSTMCell(Layer):
    """
    Mass-Conserving LSTM (MC-LSTM) cell

    :param lookback: Historical time steps
    :param horizon: Prediction time steps
    :param units: Number of hidden units
    :param return_state: Whether to return the state
    """

    def __init__(self,
                 units: int = 128,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.units = units
        # Input Gate
        self.input_gate = Gate(units=units, normalization=True)
        # Redistribution Matrix
        self.redistribution_matrix = Gate(units=units * units,
                                          activation="relu",
                                          )
        # output Gate
        self.output_gate = Gate(units=units, )

    def call(self,
             last_cell,  # [batch_size, units]
             input_precipitation,  # [batch_size, ]
             input_other_features,  # [batch_size, num_features]
             ):
        # 将last_cell除以L1范数，使其满足质量守恒
        last_cell_hat = last_cell / (
                ops.norm(last_cell, ord=1, axis=-1, keepdims=True) + epsilon())  # [batch_size, units]
        # 特征拼接
        features = ops.concatenate([input_precipitation.unsqueeze(-1),
                                    last_cell_hat,
                                    input_other_features],
                                   axis=-1)
        # 输入门
        input_gate = self.input_gate(features)  # [batch_size, units]
        # 重分配矩阵
        redistribution_matrix = self.redistribution_matrix(features)  # [batch_size, units * units]
        redistribution_matrix = redistribution_matrix.view(-1, self.units, self.units)  # [batch_size, units, units]
        redistribution_matrix = normalize(redistribution_matrix, order=1, axis=-1)  # [batch_size, units, units]
        # 输出门
        output_gate = self.output_gate(features)  # [batch_size, units]
        # 候选细胞状态
        cell_in = input_gate * input_precipitation.unsqueeze(-1) # [batch_size, units]
        cell_sys = ops.matmul(last_cell.unsqueeze(1), redistribution_matrix)  # [batch_size, 1, units]
        candidate_cell = cell_in + cell_sys.squeeze()  # [batch_size, units]
        # 输出
        output = output_gate * candidate_cell
        # 细胞状态
        cell = (1 - output_gate) * candidate_cell
        return cell, output


# %%测试MC-LSTM
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# input_precipitation = np.random.rand(32, )
# input_other_features = np.random.rand(32, 10)
# last_cell = np.zeros([32, 128])
#
# mclstm = MassConservingLSTMCell(units=128)
# hidden_state, cell = mclstm(last_cell, input_precipitation, input_other_features)
# hidden_state = tensor2numpy(hidden_state)
# cell = tensor2numpy(cell)


# %%

@keras.saving.register_keras_serializable(package='Custom', name='MassConservingLSTM')
class MassConservingLSTM(Model):
    """
    Mass-Conserving LSTM (MC-LSTM) model

    :param units: Number of hidden units
    :param return_sequences: Whether to return the sequences
    :param return_state: Whether to return the state
    """

    def __init__(self,
                 units: int = 128,
                 return_sequences: bool = False,
                 return_state: bool = False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.cell = MassConservingLSTMCell(units=units, )

    def call(self,
             inputs,  # [batch_size, time_steps, features]
             initial_cell=None,  # [batch_size, units]
             ):
        # 将inputs根据时间步长分割为input_precipitation和input_other_features，维度为(batch_size, time_steps, features)
        input_precipitation = inputs[:, :, 0]  # [batch_size, time_steps]
        input_other_features = inputs[:, :, 1:]  # [batch_size, time_steps, num_features]
        # 初始化cell
        if initial_cell is None:
            initial_cell = ops.zeros([inputs.shape[0], self.units])  # [batch_size, units]
        # 迭代时间步长
        match (self.return_sequences, self.return_state):
            case (True, True):
                hidden_states = []
                for i in range(inputs.shape[1]):
                    initial_cell, hidden_state = self.cell(initial_cell,
                                                           input_precipitation[:, i],  # [batch_size, ]
                                                           input_other_features[:, i],  # [batch_size, num_features]
                                                           )
                    hidden_states.append(hidden_state)  # [batch_size, units]
                return ops.stack(hidden_states, axis=1), initial_cell
            case (True, False):
                hidden_states = []
                for i in range(inputs.shape[1]):
                    initial_cell, hidden_state = self.cell(initial_cell,
                                                           input_precipitation[:, i],  # [batch_size, ]
                                                           input_other_features[:, i],  # [batch_size, num_features]
                                                           )
                    hidden_states.append(hidden_state)
                return ops.stack(hidden_states, axis=1)
            case (False, True):
                hidden_state = None
                for i in range(inputs.shape[1]):
                    initial_cell, hidden_state = self.cell(initial_cell,
                                                           input_precipitation[:, i],  # [batch_size, ]
                                                           input_other_features[:, i],  # [batch_size, num_features]
                                                           )
                return hidden_state, initial_cell
            case (False, False):
                hidden_state = None
                for i in range(inputs.shape[1]):
                    initial_cell, hidden_state = self.cell(initial_cell,
                                                           input_precipitation[:, i],  # [batch_size, ]
                                                           input_other_features[:, i],  # [batch_size, num_features]
                                                           )
                return hidden_state


# %%测试MC-LSTM
# import numpy as np
# from hydronetwork.utils import tensor2numpy
# inputs = np.random.rand(3, 5, 4)
# mclstm = MassConservingLSTM(units=128, return_sequences=True, return_state=True)
# hidden_states, cell = mclstm(inputs)
# hidden_states = tensor2numpy(hidden_states)
# cell = tensor2numpy(cell)
# %% MC-LSTM model
class MCLSTM(Model):
    def __init__(self,
                 lookback: int,
                 horizon: int,
                 units: int = 128,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.horizon = horizon
        self.units = units
        self.lstm = MassConservingLSTM(units=units, return_sequences=True)

    def call(self,
             features_bidirectional,  # [batch_size, lookback + horizon, features]
             ):
        output = self.lstm(inputs=features_bidirectional).sum(axis=-1)  # [batch_size, lookback + horizon]
        return output[:, -self.horizon:]  # [batch_size, horizon]

# %%测试Encoder-Decoder MC-LSTM
# import numpy as np
# from hydronetwork.utils import tensor2numpy
#
# #
# lookback = 10
# horizon = 5
# encoder_units = 128
# decoder_units = 128
# batch_size = 32
# features_lookback = np.random.rand(batch_size, lookback, 1)
# features_bidirectional = np.random.rand(batch_size, lookback + horizon, 10)
#
# encoder_decoder_mclstm = MCLSTM(lookback=lookback,
#                                 horizon=horizon,
#                                 units=128,
#                                 )
# predictions = encoder_decoder_mclstm(features_bidirectional)
# predictions = tensor2numpy(predictions)
