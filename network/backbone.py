from keras.layers import Input, Dense, Activation, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout,Flatten, Resizing
from keras.models import Model
import tensorflow as tf


def network_backbone(input_model_size):
    md = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', name='conv2d_1')(input_model_size)
    md = MaxPool2D(pool_size=(2, 2))(md)

    md = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='conv2d_2')(md)
    md = MaxPool2D(pool_size=(2, 2))(md)

    md = Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu', name='conv2d_3')(md)
    md = MaxPool2D(pool_size=(2, 2))(md)

    md = Flatten(name='flatten_1')(md)
    md = Dropout(0.5)(md)
    md = Dense(1000, activation='relu', name='dense_1')(md)
    md = Dropout(0.5)(md)
    output_md = Dense(4, activation='softmax', name='dense_2')(md)

    output_model = Model(input_model_size, output_md)

    return output_model


def opt_loss():
    # ******* added this line to fix Lagacy error
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-4,
        decay_steps=1000,
        decay_rate=0.9)
    sgd = tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    bi_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.0,
                                                         axis=-1, reduction='sum_over_batch_size',
                                                         name='categorical_crossentropy')
    return sgd, bi_loss_fn
