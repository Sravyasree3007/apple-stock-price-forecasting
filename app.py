

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Apple Stock Price Prediction",
    page_icon="🍎",
    layout="wide"
)

# Get application folder
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "apple_sarima_model.pkl"
DATA_PATH = BASE_DIR / "P721 DATASET.csv"

# ==========================================
# 2. APPLICATION TITLE
# ==========================================

st.title("🍎 Apple Stock Price Prediction")

st.write(
    "Forecast Apple stock closing prices "
    "using the SARIMA model."
)

st.divider()

# ==========================================
# 3. LOAD TRAINED SARIMA MODEL
# ==========================================

try:
    model_bundle = joblib.load(MODEL_PATH)

    model = model_bundle["model"]

    last_date = pd.Timestamp(
        model_bundle["last_date"]
    )

except Exception as e:

    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================================
# 4. LOAD HISTORICAL DATASET
# ==========================================

try:

    df = pd.read_csv(DATA_PATH)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Find Date and Close columns
    column_map = {
        col.lower().strip(): col
        for col in df.columns
    }

    date_col = column_map.get("date")

    close_col = column_map.get("close")

    if date_col is None or close_col is None:

        st.error(
            "Dataset must contain Date and Close columns."
        )

        st.stop()

    # Convert Date column
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    # Convert Close column
    df[close_col] = pd.to_numeric(
        df[close_col].astype(str).str.replace(",", ""),
        errors="coerce"
    )

    # Remove missing values
    historical_df = df[
        [date_col, close_col]
    ].dropna().copy()

    # Rename columns
    historical_df.columns = [
        "Date",
        "Closing Price"
    ]

    # Sort by date
    historical_df = historical_df.sort_values(
        "Date"
    )

except Exception as e:

    st.error(f"Error loading dataset: {e}")
    st.stop()

# ==========================================
# 5. DISPLAY HISTORICAL DATA
# ==========================================

st.header("📊 Historical Apple Stock Prices")

st.write(
    "Historical Apple stock closing prices "
    "from your original dataset."
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Total Historical Records",
        len(historical_df)
    )

with col2:

    st.metric(
        "Last Model Training Date",
        last_date.strftime("%d-%m-%Y")
    )

# Historical stock price graph

fig1, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(
    historical_df["Date"],
    historical_df["Closing Price"],
    label="Historical Closing Price",
    linewidth=1.5
)

ax1.set_title(
    "Apple Historical Stock Closing Prices"
)

ax1.set_xlabel("Date")

ax1.set_ylabel("Closing Price (USD)")

ax1.legend()

ax1.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

st.pyplot(fig1)

plt.close(fig1)

st.divider()

# ==========================================
# 6. GENERATE NEXT 30 BUSINESS DAYS
# ==========================================

future_dates = pd.bdate_range(
    start=last_date + pd.offsets.BDay(1),
    periods=30
)

# ==========================================
# 7. DATE RANGE SELECTION
# ==========================================

st.header("📅 Select Forecast Date Range")

st.write(
    "Choose dates within the next 30 business days."
)

start_date = st.date_input(
    "Start Date",
    value=future_dates[0].date(),
    min_value=future_dates[0].date(),
    max_value=future_dates[-1].date()
)

end_date = st.date_input(
    "End Date",
    value=future_dates[-1].date(),
    min_value=future_dates[0].date(),
    max_value=future_dates[-1].date()
)

# ==========================================
# 8. PREDICTION BUTTON
# ==========================================

if st.button("🔮 Predict Stock Prices"):

    if start_date > end_date:

        st.error(
            "End date must be after start date."
        )

    else:

        try:

            # Generate 30-day forecast
            forecast = model.get_forecast(
                steps=30
            )

            predicted_prices = (
                forecast.predicted_mean.values
            )

            # Create forecast dataframe
            forecast_df = pd.DataFrame({

                "Date": future_dates,

                "Predicted Closing Price":
                    predicted_prices

            })

            # Filter selected dates
            forecast_df = forecast_df[
                (forecast_df["Date"].dt.date >= start_date) &
                (forecast_df["Date"].dt.date <= end_date)
            ].copy()

            # ==================================
            # 9. DISPLAY PREDICTION RESULTS
            # ==================================

            st.success(
                "Stock price prediction completed successfully!"
            )

            st.subheader("📋 Predicted Stock Prices")

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            # ==================================
            # 10. PREDICTED STOCK PRICE GRAPH
            # ==================================

            st.subheader(
                "📈 Predicted Stock Price Graph"
            )

            fig2, ax2 = plt.subplots(
                figsize=(12, 5)
            )

            ax2.plot(
                forecast_df["Date"],
                forecast_df["Predicted Closing Price"],
                marker="o",
                label="Predicted Price"
            )

            ax2.set_title(
                "Apple Stock Price Forecast"
            )

            ax2.set_xlabel("Date")

            ax2.set_ylabel("Closing Price (USD)")

            ax2.legend()

            ax2.grid(True)

            plt.xticks(rotation=45)

            plt.tight_layout()

            st.pyplot(fig2)

            plt.close(fig2)

            # ==================================
            # 11. COMBINED HISTORICAL + FORECAST
            # ==================================

            st.subheader(
                "📊 Historical vs Predicted Prices"
            )

            fig3, ax3 = plt.subplots(
                figsize=(12, 5)
            )

            # Historical prices
            ax3.plot(
                historical_df["Date"],
                historical_df["Closing Price"],
                label="Historical Price",
                linewidth=1.5
            )

            # Predicted prices
            ax3.plot(
                forecast_df["Date"],
                forecast_df["Predicted Closing Price"],
                marker="o",
                label="Predicted Price",
                linewidth=2
            )

            ax3.set_title(
                "Apple Historical and Predicted Stock Prices"
            )

            ax3.set_xlabel("Date")

            ax3.set_ylabel("Closing Price (USD)")

            ax3.legend()

            ax3.grid(True)

            plt.xticks(rotation=45)

            plt.tight_layout()

            st.pyplot(fig3)

            plt.close(fig3)

            # ==================================
            # 12. DOWNLOAD FORECAST CSV
            # ==================================

            st.subheader(
                "📥 Download Prediction Results"
            )

            csv = forecast_df.to_csv(
                index=False
            )

            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name="Apple_Stock_Forecast.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )

# ==========================================
# 13. FOOTER
# ==========================================

st.divider()

st.caption(
    "Apple Stock Price Prediction | "
    "Time Series Forecasting using SARIMA"
)
