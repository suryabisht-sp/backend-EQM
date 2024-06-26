from .views import addData, predict_expense, PredictExpenses, predict_expense_regression
from django.urls import path



urlpatterns = [
    path('addData/',addData.as_view),
    path('predict_expense/', predict_expense.as_view()),
    path("data", PredictExpenses.as_view()),
    path("predict_expense_regression/", predict_expense_regression.as_view())
]
