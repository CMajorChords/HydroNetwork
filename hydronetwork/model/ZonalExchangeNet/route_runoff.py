# 汇流模型
import keras
from keras import Layer, layers, ops
from hydronetwork.model.layer.normalization import LinearNormalization, SoftmaxNormalization


@keras.saving.register_keras_serializable(package='Custom', name='RunoffRouting')
class RunoffRouting(Layer):
    """
    汇流模型，用于计算产流量的汇流过程

    :param horizon: 模型预测未来径流的时间步数
    :param time_layer_routing: 每一层土壤的汇流神经网络参数，key为层数，value为该层土壤对应的神经网络每层的神经元数
    :param n_soil_divisions: 土壤纵向划分层数
    :param aspect_ratio: 下一层土壤汇流时间相对于上一层土壤汇流时间的比例
    """

    def __init__(self,
                 horizon: int,
                 time_soil_layer_routing: {int: [int]},
                 n_soil_divisions: int,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.n_soil_layers = len(time_soil_layer_routing)
        self.n_soil_divisions = n_soil_divisions
        soil_layer_nets = []
        for n_soil_layer, layer_routing_units in time_soil_layer_routing.items():
            assert type(n_soil_layer) is int, "time_soil_layer_routing的key必须为int"
            assert any([type(unit) is int for unit in layer_routing_units]), "time_soil_layer_routing的value必须为list[int]"
            single_layer_net = [layers.Dense(units=units, activation='relu') for units in layer_routing_units]
            soil_layer_nets.append(keras.Sequential(single_layer_net))
        self.soil_layer_nets = soil_layer_nets
        self.softmax = SoftmaxNormalization(axis=-1)
        self.linear_normalization = LinearNormalization(axis=-1)
        self.hydrograph_length = max([max(last_layer_units) for last_layer_units in time_soil_layer_routing.values()])

    def call(self,
             runoff,  # size=batch_size*T*m*n
             ):
        """
        计算产流量的汇流过程。每一层土壤的含水量被几层线性层处理，最后一层线性层的输出作为该层单位线，
        :param runoff: 产流量序列，size=batch_size*T*m*n
        :return: 汇流后的产流量序列，size=batch_size*T*(surface_routing_time * aspect_ratio^n_soil_layers)
            T表示时间步数，每个时间步的产流都会被转成一条序列，序列的长度取决于最下层土壤的汇流时间。
        """
        unit_hydrographs = []
        for n_soil_layer in range(self.n_soil_layers):
            layer_runoff = runoff[:, :, n_soil_layer].squeeze()  # size=batch_size*T*n
            unit_hydrograph = self.soil_layer_nets[n_soil_layer](layer_runoff)  # size=batch_size*T*units
            if n_soil_layer == 0:
                unit_hydrograph = self.softmax(unit_hydrograph)  # size=batch_size*T*units
            else:
                unit_hydrograph = self.linear_normalization(unit_hydrograph)  # size=batch_size*T*units
            # 将unit_hydrograph的维度转换为batch_size*T*hydrograph_length, 缺的部分用0填充
            padding = layers.ZeroPadding1D(padding=(0, self.hydrograph_length - unit_hydrograph.shape[-1]))
            unit_hydrograph = padding(unit_hydrograph)  # size=batch_size*T*hydrograph_length
            # 得到该层土壤的总产流量
            layer_runoff = ops.sum(unit_hydrograph, axis=-1)  # size=batch_size*T
            hydrograph = ops.dot(unit_hydrograph, layer_runoff)  # size=batch_size*T*hydrograph_length
            unit_hydrographs.append(hydrograph)
        # unit_hydrographs中有n_soil_layers个元素，每个元素的size为batch_size*T*hydrograph_length
        # 将每个元素都相加在一起，得到最终的产流量序列
        return ops.sum(unit_hydrographs, axis=0)  # size=batch_size*T*hydrograph_length
