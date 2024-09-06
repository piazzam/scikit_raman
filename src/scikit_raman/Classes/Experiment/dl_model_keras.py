import tensorflow as tf

def load_model(path):
	"""
	Load a keras model from a path.

	Parameters
	----------
	path : string
		Path for loading the model.

	Returns
	-------
	tf.keras
		Deep model
	"""
	return tf.keras.models.load_model(path)
