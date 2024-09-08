import tensorflow as tf
import os

class EpochCheckpointSaver(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to save the model at regular epoch intervals.

    This callback saves the model after every specified number of epochs. 
    The model will be saved in the specified folder with a filename that includes 
    the fold, model name, and epoch number.

    Parameters
    ----------
    save_interval: int
        The number of epochs between each model checkpoint save.
    folder_path: str
        Path to the folder where the model checkpoints will be saved.
    model_name: str
        The name of the model, used in the filename of the saved checkpoints.
    fold: str
        Identifier for the model fold, used in the filename of the saved checkpoints.

    Methods
    -------
    on_epoch_end(epoch, logs=None)
        Called at the end of each epoch. Saves the model at specified intervals.
    """

    def __init__(self, save_interval, folder_path, model_name, fold):
        """
        Initialize the EpochCheckpointSaver.

        Parameters
        ----------
        save_interval: int
            Number of epochs between each checkpoint save.
        folder_path: str
            Directory path where checkpoints will be stored.
        model_name: str
            Name of the model, used in the saved file name.
        fold: str
            Identifier for the model fold, used in the saved file name.
        """
        self.save_interval = save_interval
        self.folder_path = folder_path
        self.model_name = model_name
        self.fold = fold
        
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def on_epoch_end(self, epoch, logs=None):
        """
        Save the model at the end of every specified number of epochs.

        Parameters
        ----------
        epoch: int
            The current epoch number.
        logs: dict, optional
            Currently not used but may contain information about the epoch.
        """
        if (epoch + 1) % self.save_interval == 0:
            file_path = os.path.join(self.folder_path, f'{self.fold}_{self.model_name}_epoch_{epoch+1}.keras')
            self.model.save(file_path)
            print(f'\nModel saved to {file_path} at epoch {epoch + 1}')
