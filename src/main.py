from data_understanding import DataUnderstanding

def run(file_name : str = 'crypto_data_info.csv'):
    du = DataUnderstanding()
    print("******************** Start ********************")
    du.data_understanding(file_name)
    print("******************** End ********************")


if(__name__ == "__main__"):
    run()