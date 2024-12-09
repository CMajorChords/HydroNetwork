import keras
from keras import Layer


@keras.saving.register_keras_serializable(package='Custom', name='ExchangeWaterAttention')
class ExchangeWaterAttention(Layer):
    """
    水量混合头
    """

    def __init__(self,
                 n_mix_steps: int = 1,
                 water_mix_head_activation='softmax',
                 **kwargs
                 ):
        super(ExchangeWaterAttention, self).__init__(**kwargs)
        self.n_mix_steps = n_mix_steps
        self.water_mix_head_activation = water_mix_head_activation
        self.water_mix_head = []
        for i in range(n_mix_steps):
            self.water_mix_head.append(keras.layers.Dense(1, activation=water_mix_head_activation))

    def call(self,
             inputs,
             ):
        # inputs: [batch_size, m, n]
        water_mix = []
        for i in range(self.n_mix_steps):
            water_mix.append(self.water_mix_head[i](inputs))
