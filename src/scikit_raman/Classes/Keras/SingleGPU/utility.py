from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD

def get_optimizer(optimizer, learning_rate):
        optimizers_dict = {
            'adagrad': Adagrad(learning_rate=learning_rate),
            'adam': Adam(learning_rate=learning_rate),
            'rmsprop': RMSprop(learning_rate=learning_rate),
            'sgd': SGD(learning_rate=learning_rate),
        }

        optimizer_fun = optimizers_dict.get(optimizer.lower())
        if optimizer_fun is None:
            raise ValueError('Provide a proper optimizer. Available options are \'adagrad\', \'adam\', \'rmsprop\', \'sgd\'')

        return optimizer_fun