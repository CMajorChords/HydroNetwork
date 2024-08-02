import keras


def get_call_backs(model: keras.Model,
                   mode: str = "min",
                   patience: int = 10,
                   tensorboard: bool = False,
                   ):
    """
    获取模型训练的回调函数

    :param model:keras模型，用于获取模型名称
    :param mode:观测指标的优化方向
    :param patience:早停的等待次数
    :param tensorboard:是否使用tensorboard
    :return:回调函数列表
    """
    callbacks = [keras.callbacks.EarlyStopping(patience=patience,
                                               restore_best_weights=True,
                                               monitor="val_loss",
                                               mode=mode,
                                               ),
                 keras.callbacks.ModelCheckpoint(
                     filepath="data/" + model.name + "/checkpoint/weights.{epoch:02d}-{val_loss:.6f}.weights.h5",
                     save_best_only=True,
                     save_weights_only=True,
                     monitor="val_loss",
                     mode=mode,
                 ),
                 # keras.callbacks.BackupAndRestore(backup_dir=f"data/{model.name}/backup"),
                 keras.callbacks.ProgbarLogger(),
                 keras.callbacks.CSVLogger(f"data/{model.name}/log.csv"),
                 keras.callbacks.TerminateOnNaN()
                 ]
    if tensorboard:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=f"data/{model.name}/logs"))
    return callbacks
