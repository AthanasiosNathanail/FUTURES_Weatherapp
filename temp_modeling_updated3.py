import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def load_data(file):
    """Load .csv or .xlsx file into a pandas DataFrame."""
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    return df

def preprocess_data(df: pd.DataFrame, date_col: str, temp_col: str):
    """
    Preprocess the data by:
      - Ensuring the date col is datetime
      - Sorting by date
      - Dropping rows with missing essential values
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, temp_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    return df

def create_trend_model(df: pd.DataFrame, date_col: str, temp_col: str):
    """
    Fit a linear regression model using Year as predictor and Temperature as response.
    """
    df['Year'] = df[date_col].dt.year
    X = df[['Year']].values
    y = df[temp_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_future(model, start_year, end_year):
    """
    Generate predictions from start_year to end_year inclusive.
    Returns a DataFrame with Year and Predicted Temperature.
    """
    future_years = np.arange(start_year, end_year + 1)
    future_X = future_years.reshape(-1, 1)
    future_preds = model.predict(future_X)
    
    return pd.DataFrame({
        'Year': future_years,
        'Predicted_Temperature': future_preds
    })

def make_climate_prediction(model, baseline_year=2020, horizon=2050, climate_factor=1.02):
    """
    Simple placeholder to scale predictions by a climate_factor.
    """
    future_df = predict_future(model, baseline_year, horizon)
    future_df['Climate_Adjusted_Temperature'] = future_df['Predicted_Temperature'] * climate_factor
    return future_df

# -----------------------------------------------------------
# Main Streamlit App
# -----------------------------------------------------------

def main():
    st.title("Temperature Trends with Flexible Plotting & Aggregation")
    
    st.markdown(
        """
        **Features**:
        - Upload CSV or XLSX
        - Optional columns for *uncertainty*, *country*, *city*
        - Filter by country/city
        - **Choose**: Resampling (monthly/yearly/decadal), Rolling Average, Downsampling, 
          Different plot types (line, scatter, box), and separate vs. single subplots.
        """
    )
    
    # -----------------------------
    # 1. File Upload
    # -----------------------------
    uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])
    
    if not uploaded_file:
        st.info("Please upload a file to begin.")
        return

    # Load Data
    df = load_data(uploaded_file)
    
    # Column Selection
    st.subheader("Select Key Columns")
    all_columns = list(df.columns)
    date_col = st.selectbox("Select the date/time column", all_columns)
    temp_col = st.selectbox("Select the temperature column", all_columns)
    uncertainty_col = st.selectbox("Select the uncertainty column (optional)", ["None"] + all_columns, index=0)
    country_col = st.selectbox("Select the country column (optional)", ["None"] + all_columns, index=0)
    city_col = st.selectbox("Select the city column (optional)", ["None"] + all_columns, index=0)
    
    # Basic Preprocessing
    df = preprocess_data(df, date_col, temp_col)
    
    # Optional Filtering by country/city
    if country_col != "None":
        st.subheader("Filter by Country (Optional)")
        unique_countries = df[country_col].dropna().unique()
        chosen_country = st.selectbox("Select a country or 'All'", ["All"] + list(unique_countries))
        if chosen_country != "All":
            df = df[df[country_col] == chosen_country]
    
    if city_col != "None":
        st.subheader("Filter by City (Optional)")
        unique_cities = df[city_col].dropna().unique()
        chosen_city = st.selectbox("Select a city or 'All'", ["All"] + list(unique_cities))
        if chosen_city != "All":
            df = df[df[city_col] == chosen_city]

    if len(df) == 0:
        st.warning("No data left after filtering. Please adjust your filters.")
        return
    
    st.subheader("Data Preview")
    st.write(df.head())
    
    # -----------------------------
    # 2. Plotting Transformations
    # -----------------------------
    st.subheader("Plotting Options")
    
    # Let user pick transformations via multiselect
    transformations = st.multiselect(
        "Select transformations to apply *before* plotting:",
        ["Resample", "Rolling Average", "Downsample"]
    )
    
    # We’ll work on a copy so we don’t lose the original
    df_to_plot = df.copy()
    
    # If user picks Resample, let them choose frequency
    if "Resample" in transformations:
        st.markdown("**Resampling Options**")
        freq = st.selectbox("Select resample frequency", ["M", "Y", "10AS"], 
                            help="M=Monthly, Y=Yearly, 10AS=Decadal start")
        # Convert date_col to datetime index if not already
        df_to_plot.set_index(date_col, inplace=True, drop=False)  # keep original col too
        # Resample & take mean
        df_resampled = df_to_plot.resample(freq, on=date_col).mean(numeric_only=True)
        # Drop rows with all NaN
        df_resampled.dropna(subset=[temp_col], inplace=True)
        df_to_plot = df_resampled.reset_index(drop=False)  # move index back to a column
    
    # If user picks Rolling Average, let them choose window size
    if "Rolling Average" in transformations:
        st.markdown("**Rolling Average Options**")
        window_size = st.number_input("Rolling window size (in time periods)", min_value=1, value=12)
        # If we have resampled monthly, for instance, 12 => 12 months
        # We'll do a simple rolling on the selected temperature column
        df_to_plot[temp_col + "_rolling"] = df_to_plot[temp_col].rolling(window_size).mean()
        # For clarity, we might choose to plot the rolling column instead of original
        # The user can pick which to plot. For simplicity, let's rename & rely on it:
        temp_col_for_plot = temp_col + "_rolling"
    else:
        temp_col_for_plot = temp_col
    
    # If user picks Downsample, let them choose the stride
    if "Downsample" in transformations:
        st.markdown("**Downsample Options**")
        stride = st.number_input("Take every nth row", min_value=2, value=10)
        df_to_plot = df_to_plot.iloc[::stride].copy()
    
    # -----------------------------
    # 3. Plot Type & Interactivity
    # -----------------------------
    st.subheader("Choose Plot Type")
    chart_type = st.selectbox("Chart type", ["line", "scatter", "box"])
    
    # We’ll handle these chart types in a simple if-else
    # Some chart types don’t use x=, y= in the same way (e.g., box plots typically have a categorical x)
    # But we can adapt to keep date on the x-axis in box for demonstration.
    
    if chart_type == "line":
        # We can optionally add error bars if uncertainty is present
        if uncertainty_col != "None" and uncertainty_col in df_to_plot.columns:
            fig = px.line(
                df_to_plot, 
                x=date_col, 
                y=temp_col_for_plot, 
                error_y=uncertainty_col,
                title="Temperature (Line) with Transformations"
            )
        else:
            fig = px.line(
                df_to_plot, 
                x=date_col, 
                y=temp_col_for_plot,
                title="Temperature (Line) with Transformations"
            )
    
    elif chart_type == "scatter":
        # We can set alpha < 1.0 to reduce overplotting
        if uncertainty_col != "None" and uncertainty_col in df_to_plot.columns:
            fig = px.scatter(
                df_to_plot, 
                x=date_col, 
                y=temp_col_for_plot, 
                error_y=uncertainty_col,
                opacity=0.7,
                title="Temperature (Scatter) with Transformations"
            )
        else:
            fig = px.scatter(
                df_to_plot, 
                x=date_col, 
                y=temp_col_for_plot,
                opacity=0.7,
                title="Temperature (Scatter) with Transformations"
            )
    
    else:  # "box"
        # In Plotly, a box plot typically uses y as numeric data, x as a grouping category
        # But we can group by year or month. Let's group by year.
        df_to_plot["Year"] = df_to_plot[date_col].dt.year
        fig = px.box(
            df_to_plot, 
            x="Year", 
            y=temp_col_for_plot,
            title="Temperature (Box) by Year"
        )
        fig.update_xaxes(type='category')  # so it treats years as discrete categories
    
    st.plotly_chart(fig, use_container_width=True)
    
    # -----------------------------
    # 4. Historical vs. Predicted
    # -----------------------------
    st.subheader("Historical vs. Predicted Temperature (Separate or Single Plot)")
    # We’ll build a model on the *full* df (not df_to_plot) for maximum data usage
    if len(df) < 2:
        st.warning("Not enough data to build a predictive model.")
        return
    
    model = create_trend_model(df, date_col, temp_col)
    last_year = df[date_col].dt.year.max()
    
    pred_end_year = st.slider("Forecast end year", min_value=int(last_year + 1), max_value=2100, value=min(int(last_year + 10), 2100))
    pred_df = predict_future(model, int(last_year + 1), pred_end_year)
    
    # Merge with historical
    hist_data = df.copy()
    hist_data["Year"] = hist_data[date_col].dt.year
    hist_data["Type"] = "Historical"
    
    pred_data = pred_df.rename(columns={"Predicted_Temperature": temp_col})
    pred_data["Type"] = "Predicted"
    
    combined = pd.concat([
        hist_data[["Year", temp_col, "Type"]], 
        pred_data[["Year", temp_col, "Type"]]
    ]).reset_index(drop=True)
    
    # Let user choose single or separate subplots
    subplot_option = st.radio("Plot Historical & Predicted in:", ["Single Chart", "Separate Subplots"])
    
    if subplot_option == "Single Chart":
        fig_hp = px.line(
            combined, 
            x="Year", 
            y=temp_col, 
            color="Type",
            title="Historical vs. Predicted (Single Chart)"
        )
    else:
        fig_hp = px.line(
            combined, 
            x="Year", 
            y=temp_col, 
            color="Type",
            facet_col="Type", 
            facet_col_wrap=1,  # one column, stacked plots
            title="Historical vs. Predicted (Separate Subplots)"
        )
    
    st.plotly_chart(fig_hp, use_container_width=True)
    
    # -----------------------------
    # 5. Simple Climate Adjustment
    # -----------------------------
    st.subheader("Simple Climate Adjustment")
    climate_factor = st.slider(
        "Climate factor multiplier", 
        min_value=1.0, max_value=1.1, 
        value=1.02, step=0.001
    )
    climate_end_year = st.slider("Climate prediction horizon", min_value=int(last_year+1), max_value=2100, value=min(int(last_year+30), 2100))
    
    climate_df = make_climate_prediction(model, baseline_year=int(last_year+1), horizon=climate_end_year, climate_factor=climate_factor)
    st.write("Sample of climate projection data:", climate_df.head(10))
    
    fig_climate = px.line(
        climate_df, 
        x="Year", 
        y=["Predicted_Temperature", "Climate_Adjusted_Temperature"], 
        labels={"value": "Temperature (°C or K)", "variable": "Series"},
        title="Climate-Adjusted Temperature Projections"
    )
    st.plotly_chart(fig_climate, use_container_width=True)

if __name__ == "__main__":
    main()
