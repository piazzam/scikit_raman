import pandas as pd

class TSNE_result:
    """
    A class that represent a result obtained from the application of TSNE.
    ...

    Attributes
    ----------
    result : pandas.Dataframe
        Results obtained from TSNE.

    Methods
    -------
    load_from_results(cls, result, dataset)
        Create object from the results obtained from TSNE.
    """

    def __init__(self, df):
        """
        Constructor of class scikit_raman.Preprocessing.TSNE_result.

        Parameters
        ----------
        df : pandas.Dataframe
            Dataframe containing TSNE results.
        """
        self.result = df

    @classmethod
    def load_from_results(cls, result, dataset):
        """
        Load TSNEResults from result and dataset.

        Parameters
        ----------
        result : np.array
            Results obtained from TSNE Object in scikit-learn.
        dataset : scikit_raman.Dataset
            Original dataset.

        Returns
        -------
        scikit_raman.Preprocessing.TSNE_result
            Object obtained from TSNE results.
        """
        df_base = pd.DataFrame(data = [], columns=["user", "category", "labels"])
        df_base['user'] = dataset.user
        df_base['category'] = dataset.category
        df_base['labels'] = dataset.labels
        df = pd.concat([pd.DataFrame(result), df_base], axis=1)
        return cls(df)