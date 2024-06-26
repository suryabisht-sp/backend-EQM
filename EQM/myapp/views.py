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
from .utility import dataTrain

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
openai.api_key = OPEN_AI_KEY

class PredictExpenses(APIView):
    def get(self, request):
        data = []
        objects = Expense.objects.all()
        for object in objects:
            data_dict = {"date":object.date, "category":object.category, "amount":object.amount}
            data.append(data_dict)
        print(f"data = {data}")
        return Response({"data":data})

class predict_expense_regression(APIView):
    def get(self, request):
        return Response("Get wala")
    def post(self, request):
        payload={
            "data":[],  
            "year":request.data['year'],
            "month":request.data['month'],
            "category":request.data['category']
        }
        objects = Expense.objects.all()
        for object in objects:
            data_dict = {"date":object.date, "category":object.category, "amount":object.amount}
            payload['data'].append(data_dict)
        expense = dataTrain(payload)
        return JsonResponse({'generated_text': expense})

class predict_expense(APIView):
    def post(self, request):
        data = []
        objects = Expense.objects.all()
        for object in objects:
            data_dict = {"date":object.date, "category":object.category, "amount":object.amount}
            data.append(data_dict)
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
    def addDataToDb(self, request):
            # adding()
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



