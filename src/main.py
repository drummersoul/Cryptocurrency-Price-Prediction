from data_understanding import DataUnderstanding

def run(file_name : str = 'crypto_data_info.csv'):
    du = DataUnderstanding()
    print("******************** Start ********************")
    raw_data = du.load_and_filter_data(file_name)
    cleaned_data = du.clean_data(raw_data)
    cleaned_data = du.handle_missing_data(cleaned_data)
    data_with_new_features = du.create_features(cleaned_data)
    du.data_understanding(data_with_new_features)
    du.train_models(data_with_new_features)
    # evaluate
    print("******************** End ********************")


if(__name__ == "__main__"):
    run()