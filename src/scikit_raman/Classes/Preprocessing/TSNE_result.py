import pandas as pd

class TSNE_result:

    def __init__(self, df):
        self.result = df

    @classmethod
    def load_from_results(cls, result, dataset):
        df_base = pd.DataFrame(data = [], columns=["user", "category", "labels"])
        df_base['user'] = dataset.user
        df_base['category'] = dataset.category
        df_base['labels'] = dataset.labels
        df = pd.concat([pd.DataFrame(result), df_base], axis=1)
        return cls(df)