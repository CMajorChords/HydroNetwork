import keras


def callback_for_tune(model: keras.Model,
                      mode: str = "min",
                      patience: int = 10,
                      warmup: int = 0,
                      tensorboard: bool = False,
                      ):
    """
    为微调模型准备的回调函数列表

    :param model:keras模型，用于获取模型名称
    :param mode:观测指标的优化方向
    :param patience:早停的等待次数
    :param warmup:预热的轮数
    :param tensorboard:是否使用tensorboard
    :return:回调函数列表
    """
    callbacks = [keras.callbacks.EarlyStopping(patience=patience,
                                               restore_best_weights=True,
                                               monitor="val_loss",
                                               start_from_epoch=warmup,
                                               mode=mode,
                                               ),
                 keras.callbacks.ModelCheckpoint(
                     filepath="data/" + model.name + "/checkpoint/best_model.weights.h5",
                     save_best_only=True,
                     save_weights_only=True,
                     monitor="val_loss",
                     mode=mode,
                 ),
                 keras.callbacks.BackupAndRestore(backup_dir=f"data/{model.name}/backup"),
                 keras.callbacks.ProgbarLogger(),
                 keras.callbacks.CSVLogger(f"data/{model.name}/log.csv"),
                 keras.callbacks.TerminateOnNaN()
                 ]
    if tensorboard:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=f"data/{model.name}/logs"))
    return callbacks


def callback_for_features_selection(mode: str = "min",
                                    patience: int = 10,
                                    warmup: int = 5,
                                    ):
    """
    为特征选择模型准备的回调函数列表，相比于callback_for_tune，结构更简单。

    :param mode:观测指标的优化方向
    :param patience:早停的等待次数
    :param warmup:预热的轮数
    :return:回调函数列表
    """
    callbacks = [keras.callbacks.EarlyStopping(patience=patience,
                                               restore_best_weights=True,
                                               monitor="val_loss",
                                               start_from_epoch=warmup,
                                               mode=mode,
                                               ),
                 keras.callbacks.ProgbarLogger(),
                 keras.callbacks.TerminateOnNaN()
                 ]
    return callbacks
