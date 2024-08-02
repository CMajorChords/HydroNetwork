import keras

class WarmupExponentialDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    WarmupExponentialDecay
    :param initial_learning_rate: 初始学习率。在warmup_steps步内学习率从0线性增加到initial_learning_rate
    :param decay_late_steps: 衰减步数.每decay_steps步衰减一次，decay_late_steps=1表示每步都衰减
    :param decay_rate: 衰减率.衰减率为0.9表示每decay_steps步学习率乘以0.9
    :param warmup_steps: warmup步数.在warmup_steps步内学习率从0线性增加到initial_learning_rate
    """
    def __init__(self,
                 initial_learning_rate: float=0.001,
                 decay_late_steps: int=1,
                 decay_rate: float=0.9,
                 warmup_steps: int=5,
                 ):
        super(WarmupExponentialDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_late_steps = decay_late_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.initial_learning_rate * (step / self.warmup_steps)
        else:
            return self.initial_learning_rate * keras.ops.exp(-self.decay_rate * ((step - self.warmup_steps) / self.decay_late_steps))

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_late_steps": self.decay_late_steps,
            "decay_rate": self.decay_rate,
            "warmup_steps": self.warmup_steps
        }
