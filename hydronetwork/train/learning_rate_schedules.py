import keras


class WarmupExponentialDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    WarmupExponentialDecay
    :param dataset_length: 每个epoch的步数
    :param initial_learning_rate: 初始学习率。在warmup_epochs内学习率从0线性增加到initial_learning_rate
    :param decay_late_epochs: 衰减步数.每decay_epochs个epoch衰减一次，decay_late_epochs=1表示每个epoch都衰减
    :param decay_rate: 衰减率.衰减率为0.9表示每decay_epochs个epoch学习率乘以0.9
    :param warmup_epochs: warmup步数。在warmup_epochs内学习率从0线性增加到initial_learning_rate
    """

    def __init__(self,
                 dataset_length: int,
                 initial_learning_rate: float = 0.0001,
                 decay_late_epochs: int = 1,
                 decay_rate: float = 0.99,
                 warmup_epochs: int = 5):
        self.initial_learning_rate = initial_learning_rate
        self.decay_late_epochs = decay_late_epochs
        self.decay_rate = decay_rate
        self.warmup_epochs = warmup_epochs
        self.dataset_length = dataset_length

    def __call__(self, step):
        # 计算当前属于第几个epoch
        current_epoch = step // self.dataset_length
        if current_epoch < self.warmup_epochs:
            return self.initial_learning_rate * (current_epoch + 1) / self.warmup_epochs
        else:
            decay_steps = current_epoch - self.warmup_epochs
            decay_factor = self.decay_rate ** (decay_steps / self.decay_late_epochs)
            return self.initial_learning_rate * decay_factor

    def get_config(self):
        return {
            "dataset_length": self.dataset_length,
            "initial_learning_rate": self.initial_learning_rate,
            "decay_late_epochs": self.decay_late_epochs,
            "decay_rate": self.decay_rate,
            "warmup_epochs": self.warmup_epochs
        }

    def from_config(self, config):
        self.__class__(**config)
