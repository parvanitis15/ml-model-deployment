from starter.train_model_pipeline import train_model_pipeline

if __name__ == '__main__':
    data_file = 'data/census_clean.csv'

    # run the pipeline
    train_model_pipeline(data_file)
