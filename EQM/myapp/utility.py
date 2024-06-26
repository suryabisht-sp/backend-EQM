# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error
# import pandas as pd
# # from sklearn.externals.
import joblib

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

# df = pd.DataFrame(data)

# # Extract month and year
# df['date'] = pd.to_datetime(df['date'])
# df['month'] = df['date'].dt.month
# df['year'] = df['date'].dt.year

# # Encoding categorical data
# df = pd.get_dummies(df, columns=['category'])

# # Features and target variable
# X = df.drop(["amount", "date", "id", "description"], axis=1)
# y = df["amount"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model initialization and training
# # model = GradientBoostingRegressor()
# # model.fit(X_train, y_train)
# model = joblib.load('expenses_EQ.pkl')

# # Evaluation on test data
# y_pred = model.predict(X_test)
# acc=model.score(X_test,y_test)
# print("========",f"{acc}")
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)

# # joblib.dump(model, 'expenses_EQ.pkl')

# # Function to generate dummy variables for all categories
# def generate_dummy_variables(input_data):
#     all_categories = ["category_Inventory", "category_Marketing", "category_Payroll", "category_Rent", "category_Utilities"]
#     input_dummy = {col: 0 for col in all_categories}
#     if input_data['category'] in all_categories:
#         input_dummy[input_data['category']] = 1
#     return input_dummy

# # Input future expense details
# print("Enter the expense details:")
# input_data = {
#     "month": int(input("Enter the month (1-12): ")),
#     "year": int(input("Enter the year: ")),
#     "category": input("Enter the expense category: ")
# }

# # Generate dummy variables for input category
# input_dummy = generate_dummy_variables(input_data)

# # Create DataFrame for user input5
# user_input = pd.DataFrame({
#     "month": [input_data["month"]],
#     "year": [input_data["year"]]
# })

# print(222,input_data)
# print(f"user_input = {user_input}")
# # Concatenate dummy variables to user input
# user_input = pd.concat([user_input, pd.DataFrame(input_dummy, index=[0])], axis=1)

# # Predict the expense
# prediction = model.predict(user_input)
# print("Predicted expense is:", prediction[0])





# ndw one
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.metrics import r2_score

# #Data Preparation--------------------------------------------
# Convert data to DataFrame


def dataTrain(data):

    # print(data,"identifier")
    # return "TEsting"
        print(data)
    
        print(f"Data is :- {data['data']}")
        df = pd.DataFrame(data['data'])

        # Convert date string to datetime object and extract month and year
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year

        # Encode categorical variables
        encoder = LabelEncoder()
        #Encodes the categorical variable 'category' using LabelEncoder and stores the encoded values in 'category_encoded
        df['category_encoded'] = encoder.fit_transform(df['category'])

        # Define Features (X) and Target (y)-------------------------------------------------
        X = df[['year', 'month', 'category_encoded']].copy() 
        y = df['amount']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize Gradient Boosting Regressor
        gbr = GradientBoostingRegressor()

        # Train the model
        # gbr.fit(X, y)
        gbr.fit(X_train, y_train)

        # Predict on the test set
        y_pred = gbr.predict(X_test)
        acc=gbr.score(X_test,y_test)
        print("========",f"{acc}")

        # Calculate R-squared score
        r2 = r2_score(y_test, y_pred)
        print(f"R-squared score: {r2:.2f}")

        joblib.dump(gbr, 'expenses_EQ.pkl')
        # Function to predict expenses
        def predict_expenses(year, month, category):
            category_encoded = encoder.transform([category])[0]
            prediction = gbr.predict(pd.DataFrame([[year, month, category_encoded]], columns=['year', 'month', 'category_encoded']))
            return prediction[0]

        # Interactive user input and prediction
        def main():
            while True:
                try:
                    # user_year = int(input("Enter the year (e.g., 2023): "))
                    # user_month = int(input("Enter the month (1-12): "))
                    # user_category = input("Enter the category (e.g., Rent, Utilities, Payroll, Inventory, Marketing): ")
                    user_year = int(data['year'])
                    user_month = int(data['month'])
                    user_category = data['category']
                    if user_month < 1 or user_month > 12:
                        print("Month should be between 1 and 12.")
                        continue
                    predicted_amount = predict_expenses(user_year, user_month, user_category)
                    print(f"Predicted expense amount for {user_category} in {datetime(year=user_year, month=user_month, day=1).strftime('%B %Y')}: ${predicted_amount:.2f}")
                    return f"Predicted expense amount for {user_category} in {datetime(year=user_year, month=user_month, day=1).strftime('%B %Y')}: ${predicted_amount:.2f}" 
                except ValueError:
                    print("Invalid input. Please enter a valid year and month.")

        return main()


# {
# "year":"2025",
# "month":"6",
# "category":"Rent"
# }