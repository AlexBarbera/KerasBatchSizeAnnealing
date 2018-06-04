#Original paper by Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le "Don't Decay the Learning Rate, Increase the Batch Size"

import keras
from keras import backend as K

class BatchSizeAnnealing(keras.callbacks.Callback):
    def __init__(self, callback):
        self.f = callback
        super(BatchSizeAnnealing, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        K.set_value(self.model.batch_size, self.f(epoch))

