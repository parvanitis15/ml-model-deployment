from starter.pipeline import pipeline

if __name__ == '__main__':
    data_file = 'data/census_clean.csv'

    # run the pipeline
    pipeline(data_file)
