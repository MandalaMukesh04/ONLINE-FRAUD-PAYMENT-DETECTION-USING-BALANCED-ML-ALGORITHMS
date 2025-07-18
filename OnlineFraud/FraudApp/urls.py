from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	             path("UserLogin.html", views.UserLogin, name="UserLogin"),
		     path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
		     path("Predict.html", views.Predict, name="Predict"),
		     path("PredictAction", views.PredictAction, name="PredictAction"),
		     path("LoadDataset", views.LoadDataset, name="LoadDataset"),
		     path("TrainML", views.TrainML, name="TrainML"),
		     path("BalancedData", views.BalancedData, name="BalancedData"),
		     path("Register.html", views.Register, name="Register"),
		     path("RegisterAction", views.RegisterAction, name="RegisterAction"),
		     ]