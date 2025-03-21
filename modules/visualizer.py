import streamlit as st
import altair as alt
import pandas as pd

def plot_odr_trends(trend_df: pd.DataFrame):
    """Plot ODR trends over time."""
    line_chart = alt.Chart(trend_df).mark_line(point=True).encode(
        x='Period:O',
        y='ODR:Q',
        tooltip=['Period', 'ODR']
    ).properties(title="ODR Trends Over Time")

    st.altair_chart(line_chart, use_container_width=True)

def plot_region_comparison(comparison_df: pd.DataFrame):
    """Plot ODR comparison across regions."""
    bar_chart = alt.Chart(comparison_df).mark_bar().encode(
        x='Region:O',
        y='ODR:Q',
        tooltip=['Region', 'ODR']
    ).properties(title="ODR Comparison Across Regions")

    st.altair_chart(bar_chart, use_container_width=True)

def plot_distribution(df: pd.DataFrame, column: str):
    """Plot distribution of a selected column."""
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(column, bin=True),
        y='count()',
        tooltip=[column, 'count()']
    ).properties(title=f"Distribution of {column}")

    st.altair_chart(hist, use_container_width=True)
