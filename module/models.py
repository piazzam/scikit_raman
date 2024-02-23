from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, InputLayer, Conv1D, MaxPooling1D, \
    Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from scikit_raman.Classes.Keras.SingleGPU.DLModel import DLModelKeras


def create_model(n_dims, n_classes):
    learning_rate = 0.1
    optimizer = Adam(learning_rate=learning_rate)
    epochs = 300
    batch_size = 256
    es = EarlyStopping(monitor="val_categorical_accuracy", patience=100, verbose=1,
                       restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5,
                                  patience=10, min_lr=0.0001)
    callbacks = [es, reduce_lr]
    metrics = ['categorical_accuracy']
    loss = 'categorical_crossentropy'
    model = Sequential()
    model.add(InputLayer(input_shape=(n_dims,)))
    model.add(Reshape((n_dims, 1)))

    model.add(Conv1D(filters=256,
                     kernel_size=8,
                     strides=2,
                     padding='same',
                     activation='relu'))
    model.add(Flatten())

    model.add(Dense(units=128))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.5))

    model.add(Dense(units=32))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.5))

    model.add(Dense(units=16))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=n_classes, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    keras_model = DLModelKeras(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)
    return keras_model

def create_model_benchmark_cnn(n_dims, n_classes):
    learning_rate = 0.00020441990333108206
    optimizer = Adam(learning_rate=learning_rate)
    epochs = 300
    batch_size = 256
    model = Sequential()
    model.add(InputLayer(input_shape=(n_dims,)))
    model.add(Reshape((n_dims, 1)))
    loss = "categorical_crossentropy"
    metrics = ['categorical_accuracy']
    model.add(Conv1D(filters=100,
                     kernel_size=100,
                     strides=1,
                     padding='same',
                     activation='relu'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
    model.add(Conv1D(filters=100,
                     kernel_size=5,
                     strides=2,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=6,
                           strides=3,
                           padding='same'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
    model.add(Conv1D(filters=25,
                     kernel_size=9,
                     strides=5,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=3,
                           strides=2,
                           padding='same'))
    model.add(Flatten())
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def create_model_custom_uno(n_dims, n_classes):
    es = EarlyStopping(monitor="val_categorical_accuracy", patience=50, verbose=1,
                       restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor="val_categorical_accuracy", factor=0.5, verbose=4, patience=20,
                           cooldown=10)
    callbacks = [es, lr]
    batch_size = 200
    epochs = 200
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    loss = "categorical_crossentropy"
    metrics = ["categorical_accuracy"]
    model = Sequential()
    # model.add(tf.keras.layers.GaussianNoise(0.5))
    model.add(InputLayer(input_shape=(n_dims,)))
    model.add(Reshape((n_dims, 1)))

    # ----- CNN layers
    model.add(Conv1D(filters=16,
                     kernel_size=21,
                     strides=1,
                     padding='same',
                     activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=11,
                     strides=1,
                     padding='same',
                     activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(filters=64,
                     kernel_size=5,
                     padding='same',
                     activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    # ----- Dense layers
    model.add(Dense(units=256))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.5))

    model.add(Dense(units=128))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.25))

    model.add(Dense(units=64))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.5))

    # ----- Classification layer
    model.add(Dense(units=n_classes, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    keras_model = DLModelKeras(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)
    return keras_model

def create_model_custom_uno_gaussian(n_dims=991, n_classes=2):
    es = EarlyStopping(monitor="val_categorical_accuracy", patience=50, verbose=1,
                       restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor="val_categorical_accuracy", factor=0.5, verbose=4, patience=20,
                           cooldown=10)
    callbacks = [es, lr]
    batch_size = 200
    epochs = 200
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    loss = "categorical_crossentropy"
    metrics = ["categorical_accuracy"]
    model = Sequential()
    model.add(tf.keras.layers.GaussianNoise(0.5))
    model.add(InputLayer(input_shape=(n_dims,)))
    model.add(Reshape((n_dims, 1)))

    # ----- CNN layers
    model.add(Conv1D(filters=16,
                     kernel_size=21,
                     strides=1,
                     padding='same',
                     activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2,
                           strides=2,
                           padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=11,
                     strides=1,
                     padding='same',
                     activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(filters=64,
                     kernel_size=5,
                     padding='same',
                     activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    # ----- Dense layers
    model.add(Dense(units=256))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.5))

    model.add(Dense(units=128))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.25))

    model.add(Dense(units=64))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.5))

    # ----- Classification layer
    model.add(Dense(units=n_classes, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    keras_model = DLModelKeras(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)
    return keras_model

def create_model_resnet(n_dims=991, n_classes=2):
  es = EarlyStopping(monitor="val_loss", patience=50, verbose=1,
                                           restore_best_weights=True)
  lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, verbose=4, patience=15,
                                               cooldown=10)
  callbacks = [es, lr]
  batch_size = 200
  epochs= 200
  learning_rate = 0.0001
  optimizer = Adam(learning_rate=learning_rate)
  loss = "categorical_crossentropy"
  metrics = ["categorical_accuracy"]
  model = ResNet34()
  model.compile(optimizer=optimizer, loss = loss, metrics = metrics)
  keras_model = DLModelKeras(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)
  return keras_model


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv1D(filter, 3, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv1D(filter, 3, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv1D(filter, 3, padding = 'same', strides = 2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv1D(filter, 3, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv1D(filter, 1, strides = 2)(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNet34(shape = (991, 1), classes = 2):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    #gaussian = tf.keras.layers.GaussianNoise(0.1)
    x = tf.keras.layers.ZeroPadding1D(3)(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling1D(2, padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
    return model

def create_model_simple_net(n_dims, n_classes):
    es = EarlyStopping(monitor="val_categorical_accuracy", patience=50, verbose=1,
                       restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor="val_categorical_accuracy", factor=0.5, verbose=4, patience=20,
                           cooldown=10)
    callbacks = [es, lr]
    model = Sequential()
    model.add(InputLayer(input_shape=(991,)))
    model.add(Reshape((n_dims, 1)))
    model.add(Conv1D(filters=128,
                     kernel_size=1,
                     strides=1,
                     padding='same',
                     activation='relu'))
    model.add(Conv1D(filters=32,
                     kernel_size=1,
                     strides=1,
                     padding='same',
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=8, activation="relu"))
    model.add(Dense(units=n_classes, activation="sigmoid"))

    learning_rate = 0.001
    opt = Adam(learning_rate=learning_rate)
    batch_size = 256
    epochs = 300
    metrics = ['categorical_accuracy']
    loss = 'categorical_crossentropy'
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    keras_model = DLModelKeras(model, batch_size, epochs, callbacks, opt, loss, metrics, learning_rate)
    return keras_model