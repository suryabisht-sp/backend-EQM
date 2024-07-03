import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.metrics import r2_score

# Data Preparation
# Convert data to DataFrame

def dataTrain(data):  
                
        # print(f"Data is :- {data['data']}")
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
    
        # Calculate R-squared score
        r2 = r2_score(y_test, y_pred)
        print(f"R-squared score: {r2:.2f}")

        # Save the model and the encoder
        # joblib.dump(gbr, 'expenses_EQ.pkl')
        # joblib.dump(encoder, 'encoder.pkl')
        
        def load_model():
         gbr1 = joblib.load('expenses_EQ.pkl')
         encoder1 = joblib.load('encoder.pkl')
         return gbr1, encoder1

        # Function to predict expenses
        def predict_expenses(year, month, category):
            gbr1, encoder1 = load_model()
            category_encoded = encoder1.transform([category])[0]
            prediction = gbr1.predict(pd.DataFrame([[year, month, category_encoded]], columns=['year', 'month', 'category_encoded']))
            return prediction[0]

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
