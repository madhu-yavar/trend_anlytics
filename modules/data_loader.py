import os
import pickle
import pandas as pd
import streamlit as st

# Set cache directory for period-specific data
CACHE_DIR = "cached_period_data"
os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def load_data():
    """
    Loads the main dataset from a Parquet file and formats the Period column.
    This is cached to avoid reloading on every run.
    """
    file_path = "/Users/yavar/Desktop/HDFC_Loan_Poc/Analytics_bot/New_processed_odr_data_cleaned.parquet"
    df = pd.read_parquet(file_path)
    
    # ✅ Ensure Period is consistently formatted for reliable filtering
    df["Period"] = pd.to_datetime(df["Period"], errors='coerce').dt.strftime("%b %y")
    
    # ✅ Cache period-specific data on the first load
    cache_period_data(df)
    
    return df


def cache_period_data(df):
    """
    Caches data for each unique period as separate pickle files
    for faster filtering and retrieval in queries.
    """
    for period in df["Period"].unique():
        file_path = os.path.join(CACHE_DIR, f"{period}.pkl")
        
        # ✅ Only cache if not already cached to avoid redundant writes
        if not os.path.exists(file_path):
            period_df = df[df["Period"] == period]
            with open(file_path, "wb") as f:
                pickle.dump(period_df, f)


def load_cached_period_data(periods):
    """
    Loads data for the specified periods from cached pickle files.
    Returns an empty DataFrame if no matching periods are found.
    """
    dfs = []
    for period in periods:
        file_path = os.path.join(CACHE_DIR, f"{period}.pkl")
        
        # ✅ Check if cached file exists before loading
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                dfs.append(pickle.load(f))
        else:
            st.warning(f"⚠️ No cached data found for period: {period}")
    
    return pd.concat(dfs) if dfs else pd.DataFrame()
