import pandas as pd

class PCA_result:
    """
    A class that represent a result obtained from the application of PCA.
    ...

    Attributes
    ----------
    result : pandas.Dataframe
        Results obtained from PCA.
    """

    def __init__(self, result, dataset):
        """
        Constructor of class scikit_raman.Preprocessing.PCA_result.

        Parameters
        ----------
        df : pandas.Dataframe
            Dataframe containing PCA results.
        """
        df_base = pd.DataFrame(data = [], columns=["user", "category", "labels"])
        df_base['user'] = dataset.user
        df_base['category'] = dataset.category
        df_base['labels'] = dataset.labels
        df = pd.concat([pd.DataFrame(result), df_base], axis=1)
        self.result = df