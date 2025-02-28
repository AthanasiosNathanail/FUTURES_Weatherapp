import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import plotly.express as px
from math import sqrt
import datetime

# -------------- #
# Session State Setup
# -------------- #
if "model" not in st.session_state:
    st.session_state.model = None
if "scaling_info" not in st.session_state:
    st.session_state.scaling_info = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "sequence_length" not in st.session_state:
    st.session_state.sequence_length = 12
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "norm_data" not in st.session_state:
    st.session_state.norm_data = None

# -------------- #
# Helper Functions
# -------------- #

def load_data(uploaded_file):
    """Load CSV or Excel file into a pandas DataFrame."""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type.")
        return None

def preprocess_data(df, date_col, temp_col):
    """Convert date column to datetime, drop missing values and sort by date."""
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col, temp_col], inplace=True)
    df.sort_values(date_col, inplace=True)
    return df

def generate_humidity(df):
    """Generate synthetic humidity values (around 50% ± 10%)."""
    return 50.0 + 10 * np.random.randn(len(df))

def generate_wind(df):
    """Generate synthetic wind speed values (around 5 m/s ± 2)."""
    return 5.0 + 2 * np.random.randn(len(df))

def generate_pressure(df):
    """Generate synthetic pressure values (around 1010 hPa ± 5)."""
    return 1010.0 + 5 * np.random.randn(len(df))

def compute_scaling_info(df, feature_cols):
    """Compute min and max for each feature (for normalization)."""
    scaling_info = {}
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val == min_val:
            max_val = min_val + 1e-6  # avoid zero range
        scaling_info[col] = (min_val, max_val)
    return scaling_info

def normalize_features(data, scaling_info, feature_cols):
    """
    data: np.array of shape (N, len(feature_cols))
    scaling_info: dict {col: (min_val, max_val)}
    feature_cols: list of column names in the same order as data columns
    """
    norm_data = data.copy().astype(np.float32)
    for i, col in enumerate(feature_cols):
        min_val, max_val = scaling_info[col]
        norm_data[:, i] = (norm_data[:, i] - min_val) / (max_val - min_val)
    return norm_data

def inverse_normalize(value, scaling_info, col):
    """Invert normalization for a single scalar or array for the given column."""
    min_val, max_val = scaling_info[col]
    return value * (max_val - min_val) + min_val

def create_sequences_multi(data, seq_len):
    """
    Create sliding window sequences for multi-output time-series forecasting.
    data: shape (N, num_features)
    Returns X_seq (N - seq_len, seq_len, num_features),
            y_seq (N - seq_len, num_features).
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(data)):
        X_seq.append(data[i-seq_len:i])
        y_seq.append(data[i])  # we want to predict all features at time i
    return np.array(X_seq), np.array(y_seq)

# -------------- #
# Model Definition
# -------------- #

class WeatherTransformer(nn.Module):
    """
    A simple Transformer model for multi-feature weather forecasting.
    It embeds the input features, adds a learnable positional encoding,
    passes the sequence through Transformer encoder layers, and outputs predictions
    for all features (temperature, humidity, wind, pressure).
    """
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2,
                 dim_feedforward=128, dropout=0.1, seq_len=12, output_dim=4):
        super(WeatherTransformer, self).__init__()
        self.d_model = d_model
        self.feature_embed = nn.Linear(input_dim, d_model)
        # Learnable positional encoding for a fixed sequence length
        self.pos_enc = nn.Parameter(torch.zeros(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output: (batch_size, output_dim) for each time step's final representation
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: shape (batch, seq_len, input_dim)
        x_emb = self.feature_embed(x) + self.pos_enc.unsqueeze(0)
        # Permute to shape (seq_len, batch, d_model) as required by TransformerEncoder
        x_emb = x_emb.permute(1, 0, 2)
        enc_output = self.transformer_encoder(x_emb)  # shape (seq_len, batch, d_model)
        # Use the last time step
        last_enc = enc_output[-1, :, :]  # shape (batch, d_model)
        out = self.fc_out(last_enc)      # shape (batch, output_dim)
        return out

# -------------- #
# Training Function
# -------------- #

def train_transformer(df, feature_cols, sequence_length, epochs=30):
    """
    Trains a multi-output Transformer model on all features (temp, humidity, wind, pressure).
    Returns the trained model, scaling_info, and normalized data array.
    """
    # Compute scaling info
    scaling_info = compute_scaling_info(df, feature_cols)
    data_values = df[feature_cols].values.astype(np.float32)
    # Normalize
    norm_data = normalize_features(data_values, scaling_info, feature_cols)

    # Create sequences
    X_seq, y_seq = create_sequences_multi(norm_data, sequence_length)
    if len(X_seq) == 0:
        raise ValueError("Insufficient data to form sequences. Provide more data or reduce window size.")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    # Convert to torch tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    # Initialize model
    input_dim = len(feature_cols)
    model = WeatherTransformer(input_dim=input_dim, output_dim=input_dim,
                               seq_len=sequence_length, d_model=64, nhead=8,
                               num_layers=2, dim_feedforward=128, dropout=0.1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        pred = model(X_train_tensor)  # shape (batch, input_dim)
        loss = criterion(pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_tensor)
            val_loss = criterion(val_pred, y_test_tensor).item()
        
        # Print progress
        st.text(f"Epoch {epoch}/{epochs} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss:.4f}")

    return model, scaling_info, norm_data

# -------------- #
# Forecast Function
# -------------- #

def forecast_multi(model, norm_data, scaling_info, feature_cols, sequence_length, forecast_steps):
    """
    Autoregressive multi-feature forecasting.
    Returns dict of predicted arrays for each feature + lists for rain/snow/fog probabilities.
    """
    # Start with the last sequence_length rows
    current_seq = norm_data[-sequence_length:].copy()  # shape (sequence_length, len(feature_cols))

    # Collect normalized predictions for each step
    predictions_norm = []

    model.eval()
    for _ in range(forecast_steps):
        inp = torch.tensor(current_seq).unsqueeze(0)  # shape (1, seq_len, feature_dim)
        with torch.no_grad():
            out = model(inp).squeeze(0).numpy()  # shape (feature_dim,)
        predictions_norm.append(out)

        # Shift the window: drop oldest row, append new row
        current_seq = np.vstack([current_seq[1:], out])

    # Convert to numpy array shape (forecast_steps, feature_dim)
    predictions_norm = np.array(predictions_norm)

    # Inverse normalize each feature
    predictions = {}
    for i, col in enumerate(feature_cols):
        predictions[col] = inverse_normalize(predictions_norm[:, i], scaling_info, col)

    # ------------- #
    # Weather Probabilities
    # ------------- #
    # (UPDATED SECTION ONLY)
    temp_name = feature_cols[0]   # e.g. temperature
    hum_name = "Humidity"         # second col is named "Humidity"
    
    temps = predictions[temp_name]
    humids = predictions[hum_name]

    rain_probs = []
    snow_probs = []
    fog_probs = []

    # Base probabilities + threshold-based increments
    for t, h in zip(temps, humids):
        # Start each condition with a small base
        rain_prob = 0.05
        snow_prob = 0.05
        fog_prob = 0.05

        # Increase RAIN probability with humidity and temp above freezing
        if h > 40:
            rain_prob = 0.2
        if h > 60:
            rain_prob = 0.4
        if h > 75:
            rain_prob = 0.6
        if h > 85 and t > 0:
            rain_prob = 0.8

        # Increase SNOW probability if humid + cold
        if h > 40 and t < 10:
            snow_prob = 0.2
        if h > 60 and t < 5:
            snow_prob = 0.4
        if h > 75 and t < 2:
            snow_prob = 0.6
        if h > 85 and t < 0:
            snow_prob = 0.9

        # Increase FOG probability if very humid + not hot
        if h > 80:
            fog_prob = 0.3
        if h > 90:
            fog_prob = 0.6
        if h > 95 and t < 15:
            fog_prob = 0.85

        # Append
        rain_probs.append(rain_prob)
        snow_probs.append(snow_prob)
        fog_probs.append(fog_prob)

    return predictions, rain_probs, snow_probs, fog_probs

# -------------- #
# Main Streamlit App
# -------------- #

def main():
    st.title("Transformer-based Weather Forecasting (Improved)")
    st.write("""
    1. Upload historical weather data (CSV or XLSX).
    2. Optionally filter by Country/City if columns are available.
    3. Train a Transformer to predict future values of temperature, humidity, wind, and pressure.
    4. View predicted temperature and weather probabilities for up to 100 years ahead.
    5. Adjust the climate factor without re-triggering training.
    """)

    # -------------------- #
    # Data Upload
    # -------------------- #
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is None:
            return

        st.subheader("Raw Data Preview")
        st.write(df.head())

        # Column selectors
        all_cols = list(df.columns)
        date_col = st.selectbox("Date/Time column:", all_cols, index=0)
        temp_col = st.selectbox("Temperature column:", all_cols, index=1)
        hum_col = st.selectbox("Humidity column (optional):", ["None"] + all_cols, index=0)
        wind_col = st.selectbox("Wind Speed column (optional):", ["None"] + all_cols, index=0)
        press_col = st.selectbox("Pressure column (optional):", ["None"] + all_cols, index=0)

        # Country/City columns (optional)
        country_col = st.selectbox("Country column (optional):", ["None"] + all_cols, index=0)
        city_col = st.selectbox("City column (optional):", ["None"] + all_cols, index=0)

        # Preprocess
        df = preprocess_data(df, date_col, temp_col)

        # Filter by country if requested
        if country_col != "None" and country_col in df.columns:
            countries = df[country_col].dropna().unique()
            if len(countries) > 0:
                choice_country = st.selectbox("Filter by country:", ["All"] + list(countries))
                if choice_country != "All":
                    df = df[df[country_col] == choice_country]

        # Filter by city if requested
        if city_col != "None" and city_col in df.columns:
            cities = df[city_col].dropna().unique()
            if len(cities) > 0:
                choice_city = st.selectbox("Filter by city:", ["All"] + list(cities))
                if choice_city != "All":
                    df = df[df[city_col] == choice_city]

        # If any optional columns not provided, generate synthetic
        if hum_col == "None" or hum_col not in df.columns:
            df["Humidity"] = generate_humidity(df)
            st.info("Synthetic humidity generated.")
            hum_col = "Humidity"
        if wind_col == "None" or wind_col not in df.columns:
            df["WindSpeed"] = generate_wind(df)
            st.info("Synthetic wind speed generated.")
            wind_col = "WindSpeed"
        if press_col == "None" or press_col not in df.columns:
            df["Pressure"] = generate_pressure(df)
            st.info("Synthetic pressure generated.")
            press_col = "Pressure"

        # Show after preprocessing
        st.subheader("Preprocessed Data Preview")
        st.write(df.head())

        # Plot historical temperature
        st.subheader("Historical Temperature Trend")
        fig_hist = px.line(df, x=date_col, y=temp_col, title="Historical Temperature")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Feature columns: keep consistent ordering
        feature_cols = [temp_col, hum_col, wind_col, press_col]

        # Sequence length
        seq_len = st.slider("Sequence Window Length (days/weeks/etc.):", 5, 60, 12)
        st.session_state.sequence_length = seq_len

        # Forecast horizon
        st.subheader("Forecast Horizon")
        horizon_unit = st.selectbox("Horizon unit:", ["Days", "Months", "Years"])
        
        # Allow up to 100 years
        if horizon_unit == "Days":
            forecast_steps = st.slider("Number of days to forecast:", 1, 36500, 7)  # ~100 years
            freq_offset = "D"
        elif horizon_unit == "Months":
            forecast_steps = st.slider("Number of months to forecast:", 1, 1200, 6)  # 100 years
            freq_offset = "M"
        else:
            forecast_steps = st.slider("Number of years to forecast:", 1, 100, 3)
            freq_offset = "Y"

        # -------------------- #
        # Train & Forecast
        # -------------------- #
        if st.button("Train & Forecast"):
            # Train
            with st.spinner("Training Transformer..."):
                try:
                    model, scaling_info, norm_data = train_transformer(
                        df, feature_cols, seq_len, epochs=30
                    )
                    st.session_state.model = model
                    st.session_state.scaling_info = scaling_info
                    st.session_state.feature_cols = feature_cols
                    st.session_state.norm_data = norm_data
                    st.session_state.trained = True
                except Exception as e:
                    st.error(f"Training error: {e}")
                    return

            # Forecast
            with st.spinner("Generating forecast..."):
                try:
                    preds, rain_probs, snow_probs, fog_probs = forecast_multi(
                        model, norm_data, scaling_info, feature_cols, seq_len, forecast_steps
                    )
                except Exception as e:
                    st.error(f"Forecast error: {e}")
                    return

            # Create future dates
            last_date = df[date_col].max()
            if horizon_unit == "Days":
                future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_steps, freq="D")
            elif horizon_unit == "Months":
                future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_steps, freq="M")
            else:
                future_dates = pd.date_range(last_date + pd.DateOffset(years=1), periods=forecast_steps, freq="Y")

            forecast_df = pd.DataFrame({
                date_col: future_dates,
                "Predicted_Temperature": preds[temp_col],
                "Predicted_Humidity": preds[hum_col],
                "Predicted_WindSpeed": preds[wind_col],
                "Predicted_Pressure": preds[press_col],
                "Rain_Prob": rain_probs,
                "Snow_Prob": snow_probs,
                "Fog_Prob": fog_probs
            })

            st.session_state.forecast_df = forecast_df

        # -------------------- #
        # Display Results
        # -------------------- #
        if st.session_state.trained and st.session_state.forecast_df is not None:
            forecast_df = st.session_state.forecast_df.copy()

            # Climate factor
            st.subheader("Climate Scenario Adjustment")
            climate_factor = st.slider("Warming factor (multiplier on temperature):",
                                       0.8, 1.5, 1.0, step=0.05)
            # Create a new column for climate-adjusted temp
            forecast_df["Climate_Adjusted_Temp"] = forecast_df["Predicted_Temperature"] * climate_factor

            # Show forecast data
            st.subheader("Forecasted Weather Data")
            st.write(forecast_df)

            # Plot predicted temperature
            st.subheader("Forecasted Temperature (Line Chart)")
            fig_temp = px.line(
                forecast_df, 
                x=date_col, 
                y="Climate_Adjusted_Temp", 
                title="Climate-Adjusted Temperature Forecast"
            )
            st.plotly_chart(fig_temp, use_container_width=True)

            # Plot predicted humidity
            st.subheader("Forecasted Humidity (Line Chart)")
            fig_hum = px.line(
                forecast_df, 
                x=date_col, 
                y="Predicted_Humidity", 
                title="Forecasted Humidity"
            )
            st.plotly_chart(fig_hum, use_container_width=True)

            # Plot probabilities
            st.subheader("Predicted Weather Condition Probabilities")
            fig_probs = px.line(
                forecast_df, 
                x=date_col, 
                y=["Rain_Prob", "Snow_Prob", "Fog_Prob"], 
                title="Rain/Snow/Fog Probabilities Over Time"
            )
            st.plotly_chart(fig_probs, use_container_width=True)

            # Distribution of forecasted temperature
            st.subheader("Distribution of Forecasted Temperatures")
            fig_dist = px.histogram(
                forecast_df, 
                x="Climate_Adjusted_Temp", 
                nbins=10, 
                title="Histogram of Climate-Adjusted Forecasted Temperatures"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # -------------------- #
            # Combined Historical + Forecasted Plot
            # -------------------- #
            st.subheader("Historical + Forecasted Temperature")
            # Build two dataframes: historical & forecast
            hist_df = df[[date_col, temp_col]].copy()
            hist_df["Type"] = "Historical"
            hist_df.rename(columns={temp_col: "Temperature"}, inplace=True)

            fcst_df = forecast_df[[date_col, "Climate_Adjusted_Temp"]].copy()
            fcst_df.rename(columns={"Climate_Adjusted_Temp": "Temperature"}, inplace=True)
            fcst_df["Type"] = "Forecast"

            combined_df = pd.concat([hist_df, fcst_df], ignore_index=True)
            fig_combined = px.line(
                combined_df, 
                x=date_col, 
                y="Temperature", 
                color="Type", 
                title="Historical & Forecasted Temperature"
            )
            st.plotly_chart(fig_combined, use_container_width=True)

    else:
        st.info("Please upload a data file to proceed.")

if __name__ == '__main__':
    main()
