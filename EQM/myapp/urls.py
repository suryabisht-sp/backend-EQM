from .views import addData, predict_expense, PredictExpenses
from django.urls import path



urlpatterns = [
    path('addData/',addData.as_view),
    path('predict_expense/', predict_expense.as_view()),
    path("data", PredictExpenses.as_view()),
]
