import neptune
import keras
from keras import backend as K 
from keras.callbacks import Callback

ctx = neptune.Context()
ctx.integrate_with_keras()

EPOCH_NR = ctx.params.epoch_nr
BATCH_SIZE = ctx.params.batch_size
DROPOUT = ctx.params.dropout

mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
  keras.layers.Flatten(),
  keras.layers.Dense(512, activation=K.relu),
  keras.layers.Dropout(DROPOUT),
  keras.layers.Dense(10, activation=K.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class NeptuneMonitor(Callback):
    def on_epoch_end(self, epoch, logs={}):
        innovative_metric = logs['acc'] - 2 * logs['loss']
        ctx.channel_send('innovative_metric', epoch, innovative_metric)

model.fit(x_train, y_train, 
          epochs=EPOCH_NR, batch_size=BATCH_SIZE, 
          callbacks=[NeptuneMonitor()])
