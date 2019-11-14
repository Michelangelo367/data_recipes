import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bokeh.palettes import Dark2_5 as palette # that's for getting a color palette

PATH = './example.csv'
COLUMNS = ['Source string',
           'Translation', 'Validator', 'Quantity']

class BrokenPipelineException(Exception):
    """That's only the exception that
    """
    pass

class DataProcessor(object):
    """DataProcessor is an analysis tool. It provides a pipeline tp analyze
    a source CSV. On each step, the source data remain untouched while
    the target data are incrementally updated. In the end of the pipeline, user
    is able to visualize target data by invoking the corresponding methods.
    """
    def __init__(self, path=None):
        """A python data processing pipline. It is meant to provide an easy
        way to process csv structured data.
        :param string path: the path to the source csv
        :return: the pipeline processor
        :rtype: DataProcessor
        """
        self._path = path or PATH
        self.source_data = pd.read_csv(self._path, sep=',', header=0)
        self.df = self.source_data.copy()
        self.__stuck = []
        self._kmeans = None

    def assert_columns_exist(self, columns=COLUMNS, tdf=None):
        """Validates if the columns do exist.

        :param list columns: the columns that we ask for <optional>
        :param pandas.DataFrame df: the data frame to check against <optional>
        :raises : BrokenPipelineException
        :return : a pipeline entrypoint
        :rtype : DataProcessor
        """
        if tdf is not None:
            df = tdf
        else:
            df = self.df

        column_names = df.columns.values
        missing_columns = set(columns) - set(column_names)
        if len(missing_columns)!=0:
            raise BrokenPipelineException(
                'Missing columns {}'.format(
                    list(missing_columns)
                )
            )
        return self

    def keep_columns(self, columns):
        """Given a the input dataframe, it it prunes all the unnecessary columns

        :param list columns: the columns we need to maintain
        :return: a pipeline entrypoint
        :rtype: DataProcessor
        """
        self.assert_columns_exist(columns)
        self.df = self.df[columns]
        return self

    def exclude_columns(self, columns):
        """Given a the input dataframe, it it prunes all the unnecessary columns

        :param list columns: the columns we need to remove
        :return: a pipeline entrypoint
        :rtype: DataProcessor
        """
        self.assert_columns_exist(columns)
        columns_to_keep = list(set(self.df.columns.values) - set(columns))
        self.df = self.df[columns_to_keep]
        return self

    def clean_nan(self, columns):
        """Cleans out the null dataFrameLines

        :param list columns: the columns we keep to clean up
        :return: a pipeline entrypoint
        :rtype: DataProcessor
        """
        self.assert_columns_exist(columns)
        self.df = self.df.dropna(axis=0, subset=columns)
        return self

    def one_hot_encode(self, hot_columns, keep_columns=[]):
        """Returns the one-hot encoded version of a specific column.

        Hint the column should afford distinct values.

        :param str hot_column: the column we need to one hot encode
        :return: a pipeline entrypoint
        :rtype: DataProcessor
        """
        self.assert_columns_exist(hot_columns)
        self.assert_columns_exist(keep_columns)

        one_hotted_list = [
            pd.get_dummies(self.df[hot_column])
            for hot_column in hot_columns
        ]
        # should join all one_hotted data frames to a single one
        # based on the main index
        one_hotted = one_hotted_list[0]
        for i in range(1, len(one_hotted_list)):
            one_hotted = one_hotted.set_index(self.df.index).join(
                one_hotted_list[i].set_index(self.df.index)
            )
        # in case of keep_columns we should provide the these columns as well
        self.df = one_hotted.join(self.df[keep_columns])
        return self

    def pivot_values(self, index_column, values_columns, fillValue=0):
        """Groups rows based on some particular column value.

        :param list columns: the columns we need to group and pivot
        :param int fillValue: the value to fill if missing
        :return: a pipeline entrypoint
        :rtype: DataProcessor
        """
        # TODO: this is a quick fix, should be generalized even better
        pivoted = pd.pivot_table(self.df,
                                 index=[index_column],
                                 columns=[values_columns],
                                fill_value=fillValue)
        # let's get back the original names
        column_names = list(map(lambda name: name[1], pivoted.columns.values))
        pivoted.columns = column_names
        # flatten the pivoted table
        self.df = pd.DataFrame(pivoted.to_records())
        return self

    def standarize(self, columns=None):
        """Provide data frame standarizarion.

        Hint: this is gonna raise multiple errors in case of multinested
        structures

        :param pandas.dataFrame src_frame: the source frame to standarise
        :param list columns: the columns we need to standarise
        :return: a pipeline entrypoint
        :rtype: DataProcessor

        """
        if not columns:
            columns = self.df.columns.values
        else:
            self.assert_columns_exist(columns)

        # get the columns to standarise
        frame_to_standarise = self.df.loc[:, columns]
        # assert there is non nan at the specific subframe

        standarised_values = StandardScaler().fit_transform(
            frame_to_standarise.values
        )
        # the standiezed array
        standarised_frame = pd.DataFrame(standarised_values, columns=columns)
        self.df.update(standarised_frame)
        return self

    def apply_pca(self, target_dim, notify_on_lt=0.85):
        """Principal component analysis.
        Projects the samples geometry to a spaces of <target_dim> dimensions.

        :param int target_dim: the number of dimensions of targer space
        :notify_on_lt: the threshold of the information  to keep
        :return: a pipeline entrypoint
        :rtype: DataProcessor

        Hint: PCA should be always following standarising step
        """
        # apply PCA
        pca = PCA(n_components=target_dim)
        principalComponents = pca.fit_transform(self.df)
        self.df = pd.DataFrame(
            data=principalComponents,
            columns=['pc{}'.format(i) for i in range(1, target_dim + 1)])

        # let the user be aware when the applied PCA absorbs infromation
        if pca.explained_variance_ratio_.sum() < notify_on_lt:
            print("PCA reduced data expression power into {}".format(
                pca.explained_variance_ratio_.sum()
            ))
        return self

    def fillna(self, columns, value):
        """Fills all nan with a particular value.

        :param list columns: the list of the columns to fill nan of
        :param <python primitive type> value: the const value to fill
        :return: a pipeline entrypoint
        :rtype: DataProcessor
        """
        self.df.update(self.df[columns].fillna(0))
        return self

    def kmeans_train(self, n_clusters, random_state=0):
        """Given a number of groups we the best known clustering.

        :param int n_clusters: the groups we aspire to have as a result
        :return: a pipeline entrypoint
        :rtype: DataProcessor
        """
        self._kmeans = KMeans(n_clusters=n_clusters,
            random_state=random_state).fit(self.df.to_numpy())
        return self

    def push(self):
        df = self.df.copy()
        self.__stuck.append(df)
        return self

    def pop(self):
        self.df = self.__stuck.pop(-1)
        return self

    def get_kmeans_prediction(self, instances):
        """Predicts the cluster that the instance should belong to.

        :param list instance: the list of np.1d.array instances we need to
            predict the clusters of
        :return: the list of labels of the provided instances
        :rtype: list

        Use as: get_kmeans_prediction([[0, 0], [12, 3]])
        """
        if not self._kmeans:
            raise BrokenPipelineException(
                "Please run 'kmeans_train' to train first"
            )
        return self._kmeans.predict(instances)

    def get_cluster_labels_by_columns(self, columns):
        """Retruns a mapping of sample attributes, cluster labels.

        Hint; Use pop to get the load the correct df from the pipeline

        :param list columns : list of the columns we need to project the labels to
        :return: a dataFrame of columns: [columns, cluster_labels
        :rtype: pandas.dataFrame
        """
        self.assert_columns_exist(columns=columns, tdf=self.source_data)
        labels = self.source_data[columns]
        labels = pd.DataFrame(
            self._kmeans.labels_,
            columns=["cluster_labels"],
            index=self.df.index)
        # FIXME: missing rows issue
        return pd.merge(self.df, self.source_data, how='inner', on=[self.df.index])

    def get_source_path(self):
        """Retruns the CSV path that the data originally loaded from.

        :retrun: the source csv path
        :rtype: str
        """
        return self._path

    def get_data(self):
        """Alias of the data that the pipline is build on top of.

        :returns: the data at each stage of the pipeline
        :rtype: pandas.DataFrame
        """
        data = self.df.copy()
        return data

    def visualize_2d_projection(self):
        cluster_labels = self._kmeans.labels_
        pca_cluster_labels_df = pd.concat([
            self.df,
            pd.DataFrame(
                cluster_labels,
                index=self.df.index,
                columns=['cluster'])
        ], axis = 1)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2D projection', fontsize=20)
        clusters = list(range(0, self._kmeans.n_clusters))
        colors = itertools.cycle(palette)
        for cluster, color in zip(clusters, colors):
            ax.scatter(
                pca_cluster_labels_df.loc[pca_cluster_labels_df['cluster']
                                       == cluster, 'pc1'],
                pca_cluster_labels_df.loc[pca_cluster_labels_df['cluster']
                                       == cluster, 'pc2'],
                 c=color,
                 s=50)
        ax.legend(clusters)
        ax.grid()

        return