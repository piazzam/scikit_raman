import tensorflow as tf
import os

class EpochCheckpointSaver(tf.keras.callbacks.Callback):
    def __init__(self, save_interval, folder_path, model_name):
        self.save_interval = save_interval
        self.folder_path = folder_path
        self.model_name = model_name
    
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_interval == 0:
            file_path = os.path.join(self.folder_path, f'{self.model_name}_epoch_{epoch+1}.keras')
            self.model.save(file_path)
            print(f'\nModel saved to {file_path} at epoch {epoch + 1}')