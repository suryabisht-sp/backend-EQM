import json
from django.shortcuts import render
from rest_framework.response import Response
from .serializer import ExpenseSerializer
from rest_framework.decorators import api_view
from rest_framework import status
from .models import Expense
import openai
from django.http import JsonResponse
from rest_framework.views import APIView
import os
from dotenv import load_dotenv
load_dotenv()
from .utility import dataTrain, dataScikit

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
openai.api_key = OPEN_AI_KEY

data = [
    {"id": 1, "date": "2023-01-01", "category": "Rent", "description": "Monthly rent", "amount": 50.00},
    {"id": 2, "date": "2023-01-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 3, "date": "2023-01-10", "category": "Payroll", "description": "Chef salary", "amount": 900.00},
    {"id": 4, "date": "2023-01-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 800.00},
    {"id": 5, "date": "2023-01-15", "category": "Inventory", "description": "Food supplies", "amount": 800.00},
    {"id": 6, "date": "2023-01-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 7, "date": "2023-02-01", "category": "Rent", "description": "Monthly rent", "amount": 800.00},
    {"id": 8, "date": "2023-02-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 9, "date": "2023-02-10", "category": "Payroll", "description": "Chef salary", "amount": 800.00},
    {"id": 10, "date": "2023-02-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 800.00},
    {"id": 11, "date": "2023-02-15", "category": "Inventory", "description": "Food supplies", "amount": 800.00},
    {"id": 12, "date": "2023-02-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 13, "date": "2023-03-01", "category": "Rent", "description": "Monthly rent", "amount": 800.00},
    {"id": 14, "date": "2023-03-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 15, "date": "2023-03-10", "category": "Payroll", "description": "Chef salary", "amount": 800.00},
    {"id": 16, "date": "2023-03-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 800.00},
    {"id": 17, "date": "2023-03-15", "category": "Inventory", "description": "Food supplies", "amount": 800.00},
    {"id": 18, "date": "2023-03-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 19, "date": "2023-04-01", "category": "Rent", "description": "Monthly rent", "amount": 800.00},
    {"id": 20, "date": "2023-04-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 21, "date": "2023-04-10", "category": "Payroll", "description": "Chef salary", "amount": 120.00},
    {"id": 22, "date": "2023-04-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 120.00},
    {"id": 23, "date": "2023-04-15", "category": "Inventory", "description": "Food supplies", "amount": 120.00},
    {"id": 24, "date": "2023-04-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 25, "date": "2023-05-01", "category": "Rent", "description": "Monthly rent", "amount": 120.00},
    {"id": 26, "date": "2023-05-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 27, "date": "2023-05-10", "category": "Payroll", "description": "Chef salary", "amount": 120.00},
    {"id": 28, "date": "2023-05-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 120.00},
    {"id": 29, "date": "2023-05-15", "category": "Inventory", "description": "Food supplies", "amount": 120.00},
    {"id": 30, "date": "2023-05-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 31, "date": "2023-06-01", "category": "Rent", "description": "Monthly rent", "amount": 800.00},
    {"id": 32, "date": "2023-06-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 33, "date": "2023-06-10", "category": "Payroll", "description": "Chef salary", "amount": 900.00},
    {"id": 34, "date": "2023-06-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 900.00},
    {"id": 35, "date": "2023-06-15", "category": "Inventory", "description": "Food supplies", "amount": 900.00},
    {"id": 36, "date": "2023-06-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 37, "date": "2023-07-01", "category": "Rent", "description": "Monthly rent", "amount": 900.00},
    {"id": 38, "date": "2023-07-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 39, "date": "2023-07-10", "category": "Payroll", "description": "Chef salary", "amount": 900.00},
    {"id": 40, "date": "2023-07-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 500.00},
    {"id": 41, "date": "2023-07-15", "category": "Inventory", "description": "Food supplies", "amount": 500.00},
    {"id": 42, "date": "2023-07-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 43, "date": "2023-08-01", "category": "Rent", "description": "Monthly rent", "amount": 500.00},
    {"id": 44, "date": "2023-08-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 45, "date": "2023-08-10", "category": "Payroll", "description": "Chef salary", "amount": 500.00},
    {"id": 46, "date": "2023-08-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 500.00},
    {"id": 47, "date": "2023-08-15", "category": "Inventory", "description": "Food supplies", "amount": 500.00},
    {"id": 48, "date": "2023-08-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 49, "date": "2023-09-01", "category": "Rent", "description": "Monthly rent", "amount": 500.00},
    {"id": 50, "date": "2023-09-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 51, "date": "2023-09-10", "category": "Payroll", "description": "Chef salary", "amount": 500.00},
    {"id": 52, "date": "2023-09-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 500.00},
    {"id": 53, "date": "2023-09-15", "category": "Inventory", "description": "Food supplies", "amount": 500.00},
    {"id": 54, "date": "2023-09-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 55, "date": "2023-10-01", "category": "Rent", "description": "Monthly rent", "amount": 500.00},
    {"id": 56, "date": "2023-10-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 57, "date": "2023-10-10", "category": "Payroll", "description": "Chef salary", "amount": 500.00},
    {"id": 58, "date": "2023-10-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 1500.00},
    {"id": 59, "date": "2023-10-15", "category": "Inventory", "description": "Food supplies", "amount": 1500.00},
    {"id": 60, "date": "2023-10-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 61, "date": "2023-11-01", "category": "Rent", "description": "Monthly rent", "amount": 1500.00},
    {"id": 62, "date": "2023-11-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 63, "date": "2023-11-10", "category": "Payroll", "description": "Chef salary", "amount": 1500.00},
    {"id": 64, "date": "2023-11-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 1500.00},
    {"id": 65, "date": "2023-11-15", "category": "Inventory", "description": "Food supplies", "amount": 1500.00},
    {"id": 66, "date": "2023-11-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 67, "date": "2023-12-01", "category": "Rent", "description": "Monthly rent", "amount": 1500.00},
    {"id": 68, "date": "2023-12-05", "category": "Utilities", "description": "Electricity bill", "amount": 50.00},
    {"id": 69, "date": "2023-12-10", "category": "Payroll", "description": "Chef salary", "amount": 900.00},
    {"id": 70, "date": "2023-12-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 900.00},
    {"id": 71, "date": "2023-12-15", "category": "Inventory", "description": "Food supplies", "amount": 900.00},
    {"id": 72, "date": "2023-12-20", "category": "Marketing", "description": "Social media ads", "amount": 50.00},
    {"id": 73, "date": "2024-01-01", "category": "Rent", "description": "Monthly rent", "amount": 90.00},
    {"id": 74, "date": "2024-01-05", "category": "Utilities", "description": "Electricity bill", "amount": 90.00},
    {"id": 75, "date": "2024-01-10", "category": "Payroll", "description": "Chef salary", "amount": 540.00},
    {"id": 76, "date": "2024-01-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 40.00},
    {"id": 77, "date": "2024-01-15", "category": "Inventory", "description": "Food supplies", "amount": 40.00},
    {"id": 78, "date": "2024-01-20", "category": "Marketing", "description": "Social media ads", "amount": 40.00},
    {"id": 79, "date": "2024-02-01", "category": "Rent", "description": "Monthly rent", "amount": 40.00},
    {"id": 80, "date": "2024-02-05", "category": "Utilities", "description": "Electricity bill", "amount": 40.00},
    {"id": 81, "date": "2024-02-10", "category": "Payroll", "description": "Chef salary", "amount": 40.00},
    {"id": 82, "date": "2024-02-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 40.00},
    {"id": 83, "date": "2024-02-15", "category": "Inventory", "description": "Food supplies", "amount": 90.00},
    {"id": 84, "date": "2024-02-20", "category": "Marketing", "description": "Social media ads", "amount": 90.00},
    {"id": 85, "date": "2024-03-01", "category": "Rent", "description": "Monthly rent", "amount": 90.00},
    {"id": 86, "date": "2024-03-05", "category": "Utilities", "description": "Electricity bill", "amount": 90.00},
    {"id": 87, "date": "2024-03-10", "category": "Payroll", "description": "Chef salary", "amount": 90.00},
    {"id": 88, "date": "2024-03-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 90.00},
    {"id": 89, "date": "2024-03-15", "category": "Inventory", "description": "Food supplies", "amount": 90.00},
    {"id": 90, "date": "2024-03-20", "category": "Marketing", "description": "Social media ads", "amount": 90.00},
    {"id": 91, "date": "2024-04-01", "category": "Rent", "description": "Monthly rent", "amount": 800.00},
    {"id": 92, "date": "2024-04-05", "category": "Utilities", "description": "Electricity bill", "amount": 800.00},
    {"id": 93, "date": "2024-04-10", "category": "Payroll", "description": "Chef salary", "amount": 120.00},
    {"id": 94, "date": "2024-04-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 120.00},
    {"id": 95, "date": "2024-04-15", "category": "Inventory", "description": "Food supplies", "amount": 120.00},
    {"id": 96, "date": "2024-04-20", "category": "Marketing", "description": "Social media ads", "amount": 800.00},
    {"id": 97, "date": "2024-05-01", "category": "Rent", "description": "Monthly rent", "amount": 120.00},
    {"id": 98, "date": "2024-05-05", "category": "Utilities", "description": "Electricity bill", "amount": 800.00},
    {"id": 99, "date": "2024-05-10", "category": "Payroll", "description": "Chef salary", "amount": 120.00},
    {"id": 100, "date": "2024-05-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 120.00},
    {"id": 101, "date": "2024-05-15", "category": "Inventory", "description": "Food supplies", "amount": 120.00},
    {"id": 102, "date": "2024-05-20", "category": "Marketing", "description": "Social media ads", "amount": 800.00},
    {"id": 103, "date": "2024-06-01", "category": "Rent", "description": "Monthly rent", "amount": 800.00},
    {"id": 104, "date": "2024-06-05", "category": "Utilities", "description": "Electricity bill", "amount": 800.00},
    {"id": 105, "date": "2024-06-10", "category": "Payroll", "description": "Chef salary", "amount": 540.00},
    {"id": 106, "date": "2024-06-10", "category": "Payroll", "description": "Waitstaff wages", "amount": 540.00},
]


# data from Db
class PredictExpenses(APIView):
    def get(self, request):
        data = []
        objects = Expense.objects.all()
        for object in objects:
            data_dict = {"date":object.date, "category":object.category, "amount":object.amount}
            data.append(data_dict)
        print(f"data = {data}")
        return Response({"data":data})

# prediction using regression
class predict_expense_regression(APIView):
    def get(self, request):
        return Response("Get wala")
    def post(self, request):
        payload={
            "data":data,  
            "year":request.data['year'],
            "month":request.data['month'],
            "category":request.data['category']
        }
        # objects = Expense.objects.all()
        # for object in objects:
        #     data_dict = {"date":object.date, "category":object.category, "amount":object.amount}
        #     payload['data'].append(data_dict)
        expense = dataTrain(payload)
        # expense = dataScikit(payload)
        return JsonResponse({'generated_text': expense})
    

# predicit expenses using openi
class predict_expense(APIView):
    def post(self, request):
        # data = []
        # objects = Expense.objects.all()
        # for object in objects:
        #     data_dict = {"date":object.date, "category":object.category, "amount":object.amount}
        #     data.append(data_dict)
        month = request.data['month']
        year = request.data['year']
        category= request.data['category']
        prompt_text=''

        if category == "All":
            prompt_text = f"""Predict the expense for {category} for the {month} {year} based on the expense data provided: {str(data)}
        I want only price for all the five category in the below format in response \n,
        category_name = price and please remove all the other context from response.
        """
        else:
            prompt_text = f"""Predict the expense for {category} for the {month} {year} based on the expense data provided: {str(data)}
        I want the price for this category in response and dont give me the detailed explanation, just give me only the average price for this category instead of range
        and the response will be like : category_name = $average price''
        """
            
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system",   "content": "You are a financial analyst."},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=500,  
                temperature=0.5,
        )
        prediction1 = response['choices'][0]['message']['content'].strip()
        # Format response
        def extract_relevant_part(text):
            marker = "format:"
            if marker in text:
                return text.split(marker, 1)[1].strip()
            return text
        
        filtered_prediction = extract_relevant_part(prediction1)
        print("predicion is here:- ", filtered_prediction)

        return JsonResponse({'generated_text': filtered_prediction})


# adding data to DB
class addData(APIView):
    def get(self, request):
            adding()
            return Response("Added", status=status.HTTP_201_CREATED)

# for adding data to db
def adding():
    for item in data:
        print("process", item["id"])
        serializer = ExpenseSerializer(data=item)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        else:
            print(f"Error in data: {serializer.errors}")



