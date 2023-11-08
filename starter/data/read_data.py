import pandas as pd


if __name__ == '__main__':
    # read the data
    df = pd.read_csv('census_clean.csv')
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.shape)
    print(df.columns)
    print(df.dtypes)
