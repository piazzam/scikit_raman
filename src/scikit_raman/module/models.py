from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, InputLayer, Conv1D, MaxPooling1D, \
    Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from scikit_raman.Classes.Keras.Model.DLModel import DLModelKeras
from tensorflow.keras.initializers import HeUniform


def create_model_benchmark_weight_initialization(n_dims, number_classes):
    initializer = HeUniform()

    # ----- init model
    model = Sequential()
    model.add(InputLayer(shape=(n_dims,)))
    model.add(Reshape((n_dims, 1)))

    # ----- CNN layers
    model.add(Conv1D(filters=100,
                     kernel_size=100,
                     strides=1,
                     padding='same',
                     activation='relu',
                     kernel_initializer=initializer))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
    model.add(Conv1D(filters=100,
                     kernel_size=5,
                     strides=2,
                     padding='same',
                     activation='relu',
                     kernel_initializer=initializer))
    model.add(MaxPooling1D(pool_size=6,
                           strides=3,
                           padding='same'))
    model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
    model.add(Conv1D(filters=25,
                     kernel_size=9,
                     strides=5,
                     padding='same',
                     activation='relu',
                     kernel_initializer=initializer))
    model.add(MaxPooling1D(pool_size=3,
                           strides=2,
                           padding='same'))

    # ----- Flatten layer between CNN and Dense layers
    model.add(Flatten())
    model.add(Dropout(rate=0.1))
    # ----- Dense layers
    model.add(Dense(units=732))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.7))

    model.add(Dense(units=189))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.25))

    model.add(Dense(units=152))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.1))

    # ----- Classification layer
    model.add(Dense(units=number_classes, activation='softmax'))
    return model

def create_model_benchmark(n_dims, number_classes):

    # ----- init model
    model = Sequential()
    model.add(InputLayer(shape=(n_dims,)))
    model.add(Reshape((n_dims, 1)))

    # ----- CNN layers
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

    # ----- Flatten layer between CNN and Dense layers
    model.add(Flatten())
    model.add(Dropout(rate=0.1))
    # ----- Dense layers
    model.add(Dense(units=732))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.7))

    model.add(Dense(units=189))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.25))

    model.add(Dense(units=152))
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.1))

    # ----- Classification layer
    model.add(Dense(units=number_classes, activation='softmax'))
    return model

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