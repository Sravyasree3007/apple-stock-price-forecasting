# Apple Stock Price Prediction

## Project Overview
This project focuses on forecasting Apple Inc. (AAPL) stock prices using historical stock market data from 2012 to 2019.

The main objective is to analyze historical stock trends and predict future closing prices using time series forecasting techniques.

## Objectives
- Analyze historical Apple stock price data.
- Perform data preprocessing and exploratory data analysis (EDA).
- Identify trends, seasonality, and volatility.
- Apply time series forecasting techniques.
- Forecast the next 30 business days of Apple stock prices.
- Visualize historical data and predicted stock prices through a Streamlit application.

## Dataset
The project uses historical Apple stock market data.

**Dataset file:** `P721 DATASET.csv`

The dataset includes:
- Date
- Open
- High
- Low
- Close
- Volume

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Streamlit

## Project Workflow

### 1. Data Preprocessing
- Load the historical stock dataset.
- Check for missing values and duplicate records.
- Convert date columns into datetime format.
- Prepare the data for analysis and forecasting.

### 2. Exploratory Data Analysis (EDA)
- Analyze historical stock prices.
- Visualize closing price trends.
- Study seasonality and volatility.
- Examine trading volume patterns.
- Use charts and correlation analysis to understand the data.

### 3. Feature Engineering
- Calculate moving averages.
- Calculate daily returns.
- Analyze price ranges and volatility.
- Examine changes in trading volume.

### 4. Model Development
The project explores time series forecasting and machine learning approaches for stock price prediction.

The SARIMA model is used in the Streamlit application to generate the saved forecast.

### 5. Model Evaluation
Forecasting models can be evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

### 6. Deployment
A Streamlit application provides an interface to:
- Select a forecast date range.
- View predicted stock prices.
- Visualize forecast results.
- Download prediction results as a CSV file.

## Project Structure

```text
apple-stock-price-forecasting/
│
├── P721 DATASET.csv
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── apple_sarima_model.pkl
```

**Note:** The trained model file must be available in the project directory when running the application. It may be excluded from GitHub through `.gitignore`.

## How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/Sravyasree3007/apple-stock-price-forecasting.git
```

### Step 2: Open the project folder

```bash
cd apple-stock-price-forecasting
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit application

Make sure the trained model file `apple_sarima_model.pkl` is present in the project directory.

```bash
streamlit run app.py
```

### Step 5: Open the application

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## Expected Output
The application displays:
- Apple stock price forecasts.
- Prediction visualizations.
- Historical and forecast comparisons (if enabled in the application).
- A downloadable CSV file containing forecast results.

## Future Enhancements
- Include external financial indicators.
- Improve forecasting accuracy through model tuning.
- Compare multiple forecasting models.
- Deploy the application online.

## Disclaimer
This project is developed for educational purposes only. Stock price predictions are estimates and should not be considered financial or investment advice.

## Author
**Sravyasree**  
GitHub: [Sravyasree3007](https://github.com/Sravyasree3007)
