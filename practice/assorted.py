import numpy as np
from copy import deepcopy

def outlier_treatment(col):

    sorted(col)
    q1, q3 = np.percentile(col, [25, 75])
    iqr = q3 - q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)
    return lower_range, upper_range


class Rim:

    def __init__(self, df, y=None):
        self.df = df
        self.y = y

    def show_outliers(self, col):

        lower_range, upper_range = outlier_treatment(self.df[col])
        lower_df, upper_df = self.df[self.df[col] < lower_range], self.df[self.df[col] > upper_range]
        lower_outlier_counts = lower_df.value_counts().sum(axis=0)
        upper_outlier_counts = upper_df.value_counts().sum(axis=0)
        total_outlier_counts = lower_outlier_counts + upper_outlier_counts
        print('lower_outlier_counts', lower_outlier_counts)
        print('upper_outlier_counts', upper_outlier_counts)
        print('total_outlier_counts', total_outlier_counts)

        return total_outlier_counts

    def get_outliers_indices(self, col):

        lower_range, upper_range = outlier_treatment(self.df[col])
        lower_indices, upper_indices = self.df[self.df[col] < lower_range].index, self.df[self.df[col] > upper_range].index
        indices = list(lower_indices) + list(upper_indices)
        print('outlier indices:', indices)
        return indices

    def drop_outliers(self, col):
        print('Shape Before Dropping outliers:', self.df.shape)
        self.df.drop(self.get_outliers_indices(col), inplace=True)
        print('Shape After Dropping outliers:', self.df.shape)

    def correlation_treatment(self, method='pearson', lra=True):
        print('correlation treatment')
        pearson_corr = self.df.corr(method=method)
        corr_target = abs(pearson_corr[self.y])
        relevant_features = corr_target[corr_target > .5]
        relevant_cols = list({cols for cols in relevant_features.index if cols != self.y})
        if lra:
            col_pairs, corr_pairs, drop_cols = [], [], []
            for n in range(len(relevant_cols)):
                col_pairs += [[relevant_cols[n], cols] for cols in relevant_cols[n+1:]]
            print('col_pairs: ', col_pairs)
            for pairs in col_pairs:
                corr = self.df[pairs].corr()
                if abs(corr[pairs[0]][pairs[1]]) > .5:
                    corr_pairs.append(pairs)

            print('corr pairs: ', corr_pairs)
            for pairs in corr_pairs:
                #corr = self.df[pairs[0], self.y].corr()
                x = abs(pearson_corr[pairs[0]][self.y])
                #corr = self.df[pairs[1], self.y].corr()
                y = abs(pearson_corr[pairs[1]][self.y])

                if x > y:
                    drop_cols.append(pairs[1])
                else:
                    drop_cols.append(pairs[0])
            print('drop columns: ', drop_cols)
            print('shape before dropping: ', self.df.shape)
            self.df.drop(columns=drop_cols, inplace=True)
            print('shape after dropping: ', self.df.shape)
        #print(relevant_features, 123, type(relevant_features))


    # def outlier_treatment(self):