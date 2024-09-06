import sklearn.ensemble as ensemble
import sklearn.svm as svm
import sklearn.discriminant_analysis as da
import scikit_raman.Classes.MLModel as MLModel

def name_to_object(name, seed=42):
	"""
	Read a string and return the corresponding scikit-learn object.

	Parameters
	----------
	name : string
		Name of the Machine Learning Model.
	seed : int, optional
		Seed for creation of scikit-learn objects.

	Returns
	-------

	"""
	if name == 'xgboost':
		return MLModel.MLModel(ensemble.GradientBoostingClassifier(random_state=seed))
	elif name == 'svm':
		return MLModel.MLModel(svm.SVC(probability=True, random_state=seed))
	elif name == 'lda':
		return MLModel.MLModel(da.LinearDiscriminantAnalysis())