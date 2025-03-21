import streamlit as st
import re
import json
import pandas as pd
from models.hybrid_llm import HybridLLM

llm = HybridLLM()

def extract_filters(query: str, df):
    """Accurately extracts Region, Pools, and Period using structured JSON formatting."""
    prompt = f"""
    You are an AI data analyst. Some insights about the data:
    - The data has ODR (Overall Defaulter Rate), Bad_Count (BC), Total_Count (TC) and time periods (Sep 18, Dec 18, Mar 19, Jun 19, Sep 19), Regions like A, AC, unknown, Z, etc., and Pools like P50, P26, etc.
    - The data is divided by time periods and stored in separate pickle files.
    - Understand the query and extract the filters, which are critical for the analysis.

    Extract the following filters from the query:
    - **Region** (e.g., "D", "Z", "NA", or "All" if missing)
    - **Pools** (e.g., "P50", "P26", or "All" if missing)
    - **Period** (e.g., "Dec 18", "Sep 19", or "All" if missing).
      Sometimes the periods may not be straightforward; for example, "last 2 quarters" or "next 2 quarters". 
      In that case, identify the right time period according to the data distribution and provide the exact periods like "Sep 19, Jun 19" etc.
      Do not give vague terms like "last 2 quarters" or "next 2 quarters".
    -Important: If the query's intent is prediction for example "what is the ODR trend for next 2 quarters", consider "All" the available time period data for the prediction.
    **User Query**: "{query}"

    **Return JSON ONLY in this format (without extra text)**:
    {{"Region": "value", "Pools": "value", "Period": "value"}}
    """
    
    response = llm._call(prompt)
    
    # ✅ Ensure response is valid JSON
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            filters = json.loads(json_str)
        except json.JSONDecodeError:
            filters = {}
    else:
        filters = {}

    def split_values(value):
        """Helper function to split comma-separated values and return a list"""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",")]
        return value

    # Convert all filters to lists for consistency
    return {
        "Region": split_values(filters.get("Region", "All")),
        "Pools": split_values(filters.get("Pools", "All")),
        "Period": split_values(filters.get("Period", "All")),
    }

def apply_advanced_filtering(df, query=None):
    """Apply advanced filtering using extracted filters and user selections."""
    #st.sidebar.subheader("🔎 Advanced Filters")

    # Extract filters from query if provided
    filters = extract_filters(query, df) if query else {}

    st.write(" Extracted Filters 🔍", filters)  # Debugging

    # Clean and normalize data
    df['Region'] = df['Region'].astype(str).str.strip().str.upper()
    df['Pools'] = df['Pools'].astype(str).str.strip().str.upper()

    # Apply Region Filter
    selected_regions = filters.get("Region", ['All'])
    if 'All' not in selected_regions:
        df = df[df['Region'].isin(selected_regions)]

    # Apply Pool Filter
    selected_pools = filters.get("Pools", ['All'])
    if 'All' not in selected_pools:
        df = df[df['Pools'].isin(selected_pools)]

    # Preserve original period strings for display and matching
    df['Period_Str'] = df['Period'].astype(str).str.strip()

    # Apply Period Filter Directly
    selected_periods = filters.get("Period", ['All'])
    if 'All' not in selected_periods:
        df = df[df['Period_Str'].isin(selected_periods)]

    # Collect applied filters
    applied_filters = {
        "Region": selected_regions,
        "Pools": selected_pools,
        "Period": selected_periods
    }

    #st.write("🔍 Debug: Applied Filters:", applied_filters)
    st.write("🔍 Debug: Filtered DataFrame Shape:", df.shape)
    st.dataframe(df.head())

    return df, applied_filters
