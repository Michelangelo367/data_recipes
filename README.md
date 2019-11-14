# data_recipes
A repo about tools and recipes for data analysis or management.


# csv_Processor:
That's a lean pipeline to manage large CSVs. Example:
```
k = DataProcessor()
    k.keep_columns(['Col1', 'Col2', 'Col3']
        ).clean_nan(['Col2', 'Col1']
        ).fillna(['Col3'], 0
        ).pivot_values( 'Col1', 'Col2', 0
        ).exclude_columns(['Col1']
        ).standarize(
        ).kmeans_train(3
        ).apply_pca(2
        ).visualize_2d_projection()
```
