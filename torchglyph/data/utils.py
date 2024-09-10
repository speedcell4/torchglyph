from typing import List, Tuple, Union


def iter_columns(*datasets, columns: Union[List[str], Tuple[str, ...]]):
    for dataset in datasets:
        for row in dataset:
            for column in columns:
                yield row[column]
