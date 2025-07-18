from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pymysql
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB

global username
global X_train, X_test, y_train, y_test, X, Y, train_size
labels = ['Non-Fraud', 'Fraud']
accuracy = []
precision = []
recall = [] 
fscore = []

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    a = (accuracy_score(y_test,predict)*100)
    p = (precision_score(y_test, predict,average='macro') * 100)
    r = (recall_score(y_test, predict,average='macro') * 100)
    f = (f1_score(y_test, predict,average='macro') * 100)
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm

dataset = pd.read_csv("Dataset/PS_20174392719_1491204439457_log.csv")
Y = dataset['isFraud'].ravel()
unique, count = np.unique(Y, return_counts=True)
dataset.drop(['step', 'type', 'isFraud', 'isFlaggedFraud'], axis = 1,inplace=True)

label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for j in range(len(types)):
    name = types[j]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[j], le])
dataset.fillna(dataset.mean(), inplace = True)

X = dataset.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data
train_size = X_train.shape[0]
rf = RandomForestClassifier(n_estimators=1)
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
calculateMetrics("RF", y_test, predict)

nb = GaussianNB()
nb.fit(X_train, y_train)
predict = nb.predict(X_test)
calculateMetrics("nb", y_test, predict)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
calculateMetrics("rf", y_test, predict)
conf_matrix = confusion_matrix(y_test, predict)

nb = GaussianNB()
nb.fit(X_train, y_train)
predict = nb.predict(X_test)
calculateMetrics("nb", y_test, predict)

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global rf, scaler, labels, dataset
        myfile = request.FILES['t1'].read()
        filename = request.FILES['t1'].name
        if os.path.exists('FraudApp/static/'+filename):
            os.remove('FraudApp/static/'+filename)
        with open('FraudApp/static/'+filename, "wb") as file:
            file.write(myfile)
        file.close()
        testData = pd.read_csv('FraudApp/static/'+filename)
        data = testData.values
        testData.drop(['step', 'type'], axis = 1,inplace=True)
        for i in range(len(label_encoder)):
            le = label_encoder[i]
            testData[le[0]] = pd.Series(le[1].transform(testData[le[0]].astype(str)))#encode all str columns to numeric
        testData.fillna(dataset.mean(), inplace = True)
        testData = scaler.transform(testData)
        predict = rf.predict(testData)
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Test Data</th><th><font size="3" color="black">Detection Status</th></tr>'
        for i in range(len(predict)):
            if predict[i] == 0:
                output += '<tr><td><font size="3" color="black">'+str(data[i])+'</td><td><font size="4" color="green">Normal Transaction</td></tr>'
            else:
                output += '<tr><td><font size="3" color="black">'+str(data[i])+'</td><td><font size="4" color="red">Fraud Transaction</td></tr>'
        output+= "</table></br></br></br></br>"       
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def TrainML(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, labels
        global accuracy, precision, recall, fscore, conf_matrix
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Algorithm Name</th><th><font size="3" color="black">Accuracy</th>'
        output += '<th><font size="3" color="black">Precision</th><th><font size="3" color="black">Recall</th><th><font size="3" color="black">FSCORE</th></tr>'
        algorithms = ['Random Forest', 'Naive Bayes', 'Random Forest with Smote', 'Naive Bayes with Smote']
        for i in range(len(algorithms)):
            output += '<tr><td><font size="3" color="black">'+algorithms[i]+'</td><td><font size="3" color="black">'+str(accuracy[i])+'</td><td><font size="3" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(recall[i])+'</td><td><font size="3" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("All Algorithms Comparison Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(labels)])
        df = pd.DataFrame([['Random Forest','Accuracy',accuracy[0]],['Random Forest','Precision',precision[0]],['Random Forest','Recall',recall[0]],['Random Forest','FSCORE',fscore[0]],
                           ['Naive Bayes','Accuracy',accuracy[1]],['Naive Bayes','Precision',precision[1]],['Naive Bayes','Recall',recall[1]],['Naive Bayes','FSCORE',fscore[1]],
                           ['Random Forest with Smote','Accuracy',accuracy[2]],['Random Forest with Smote','Precision',precision[2]],['Random Forest with Smote','Recall',recall[2]],['Random Forest with Smote','FSCORE',fscore[2]],
                           ['Naive Bayes with Smote','Accuracy',accuracy[3]],['Naive Bayes with Smote','Precision',precision[3]],['Naive Bayes with Smote','Recall',recall[3]],['Naive Bayes with Smote','FSCORE',fscore[3]],
                          ],columns=['Parameters','Algorithms','Value'])
        df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', ax=axis[1])        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):    
    if request.method == 'GET':
        global unique, count, labels, dataset
        output = '<font size="3" color="black">Online Fraud Payment Detection Dataset Loaded</font><br/>'
        output += '<font size="3" color="blue">Total records found in Dataset = '+str(dataset.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">Different Class Labels found in Dataset = '+str(labels)+'</font><br/><br/>'
        #visualizing class labels count found in dataset
        height = count
        bars = labels
        y_pos = np.arange(len(bars))
        plt.figure(figsize = (4, 3)) 
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.xlabel("Imbalanced Dataset Class Label Graph")
        plt.ylabel("Count")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def BalancedData(request):    
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, X, Y, train_size
        output = '<font size="3" color="black">Smote Balancing Dataset Details</font><br/>'
        output += '<font size="3" color="blue">Training Size Before Applying SMOTE = '+str(train_size)+'</font><br/>'
        output += '<font size="3" color="blue">Training Size after Applying SMOTE = '+str(X_train.shape[0])+'</font><br/><br/>'
        unique, count = np.unique(y_train)
        height = count
        bars = labels
        y_pos = np.arange(len(bars))
        plt.figure(figsize = (4, 3)) 
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.xlabel("Balanced Dataset After Applying SMOTE Graph")
        plt.ylabel("Count")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)    

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'fraud',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'fraud',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform online fraud Detection"
        context= {'data':output}
        return render(request, 'Register.html', context)    

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'fraud',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})
