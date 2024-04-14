### Setting Virtual Environment

```bash
$ pip install virtualenv
$ python -m venv venv # for windows
$ source venv/Scripts/activate # to activate virtualenv windows
$ source venv/bin/activate # to activate virtualenv mac
$ pip install -r requirements.txt # to install all the packages
$ pip freeze > requirements.txt # to update requirements.txt
```

### Running the Dashboard
```bash
$ streamlit run src/main.py

or

$ bash run_dashboard.sh
```

#Dashboard Link
#https://cryptocurrency-price-prediction-whrfmwvuvjhakbdqusrzet.streamlit.app/