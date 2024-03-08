from Data_understanding import DataUnderstanding

def run(file_name : str = 'Crypto_data_info.csv'):
    du = DataUnderstanding()
    print("******************** Start ********************")
    du.data_understanding(file_name)
    print("******************** End ********************")


if(__name__ == "__main__"):
    run()