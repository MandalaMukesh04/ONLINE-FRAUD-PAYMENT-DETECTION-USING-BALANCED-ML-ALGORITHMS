# ONLINE-FRAUD-PAYMENT-DETECTION-USING-BALANCED-ML-ALGORITHMS
This Django-based web application detects fraudulent financial transactions using machine learning. It provides an end-to-end interface from data loading, preprocessing, training models, and evaluating results.

---

## Features

- User registration and login  
- Upload transaction test files  
- Detect fraud using trained ML models  
- Balanced dataset handling with SMOTE  
- Train and compare models (Random Forest & Naive Bayes)  
- Display metrics: Accuracy, Precision, Recall, F1-Score  
- Visualizations: Confusion Matrix & Comparison Graphs  

---

## Dataset

Original dataset: `Dataset/PS_20174392719_1491204439457_log.csv`  
- Contains financial transaction records  
- Binary label `isFraud` for classification  

---

## Requirements

Create a `requirements.txt` file with the following content:

django==2.1.7
scikit-learn
imbalanced-learn
matplotlib
seaborn
pandas
numpy
pymysql

---

Install dependencies with:
```bash
pip install -r requirements.txt
Setup Instructions
1. Install Python 3.7.2
Download from python.org

During installation, check Add Python to PATH

```
```bash
2. Install Django 2.1.7

pip install django==2.1.7
```
```bash
3. Set Up WAMP Server
Download and install from https://www.wampserver.com/en/

Start WAMP and make sure MySQL is running

Open http://localhost/phpmyadmin in browser

Create a database named fraud
```
```
Use the SQL in database.txt to create the table:

CREATE TABLE register (
    username VARCHAR(50) PRIMARY KEY,
    password VARCHAR(50),
    contact VARCHAR(12),
    email VARCHAR(50),
    address VARCHAR(80)
);
```
```
4. Configure Django to Use MySQL
Edit your settings.py:

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'fraud',
        'USER': 'root',
        'PASSWORD': '',  # default WAMP setup has empty password
        'HOST': '127.0.0.1',
        'PORT': '3306',
    }
}
```
```
import pymysql
pymysql.install_as_MySQLdb()
5. Run the App

python manage.py makemigrations
python manage.py migrate
python manage.py runserver 8000
Access it at:

http://127.0.0.1:8000/
URL Routes Summary
```
```
Route	Description
/Home page
/UserLogin.html	Login form
/UserLoginAction	Handle login
/Register.html	Register form
/RegisterAction	Handle registration
/LoadDataset	Show dataset info & visualization
/TrainML	Train models and show comparison
/BalancedData	SMOTE balancing details
/Predict.html	Upload prediction file form
/PredictAction	Run fraud detection on uploaded file
```
```

License
This project is for educational use. Modify and reuse with attribution
```
