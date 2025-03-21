# Import necessary modules

from pandasai import SmartDataframe
from models.hybrid_llm import HybridLLM
import streamlit as st

# Import new functions from odr_calculator
from odr_calculator import (
    calculate_odr,
    calculate_weighted_odr,
    calculate_trends,
    compare_regions,
    calculate_odr_ranks,
    compare_product_level_odr,
    top_odr_regions,
    bottom_odr_regions,
    rank_change_regions,
    segment_regions,                  # ✅ NEW: For segmentation
    group_pools_by_count,
    sourcing_trend,
    correlate_with_macroeconomic,
    predict_future_odr,
    region_with_no_defaults,
    highest_bad_count_pool,
    highest_odr_pool_region
)

llm = HybridLLM()

def execute_steps(categorized_steps, query, df):
    """Dynamically execute steps identified and categorized."""
    
    if categorized_steps["filtering"]:
        st.write("🔄 **Executing Filtering Steps...**")
        df = apply_filtering(categorized_steps["filtering"], df)
    
    if categorized_steps["aggregation"]:
        st.write("🔄 **Executing Aggregation Steps...**")
        df = apply_aggregation(categorized_steps["aggregation"], df)
    
    if categorized_steps["comparison"]:
        st.write("🔄 **Executing Comparison Steps...**")
        df = apply_comparison(categorized_steps["comparison"], df)
    
    if categorized_steps["trend_analysis"]:
        st.write("🔄 **Executing Trend Analysis Steps...**")
        df = apply_trend_analysis(categorized_steps["trend_analysis"], df)
    
    if categorized_steps["segmentation"]:  # ✅ NEW: Added segmentation execution
        st.write("🔄 **Executing Segmentation Steps...**")
        df = apply_segmentation(categorized_steps["segmentation"], df)

    if categorized_steps["visualization"]:
        st.write("🔄 **Executing Visualization Steps...**")
        fig = generate_visualization(query, df)
        if fig:
            st.pyplot(fig)
    
    if categorized_steps["prediction"]:
        st.write("🔄 **Executing Prediction Steps...**")
        prediction = apply_prediction(categorized_steps["prediction"], df)
        st.write("🔮 **Prediction Result:**", prediction)

    return df

def apply_filtering(filtering_steps, df):
    """Apply filtering steps on the dataframe."""
    for step in filtering_steps:
        st.write("🔍 **Applying Filter:**", step)
        # Implement dynamic filtering logic
    return df

def apply_aggregation(aggregation_steps, df):
    """Apply aggregation steps on the dataframe."""
    for step in aggregation_steps:
        st.write("🔍 **Applying Aggregation:**", step)
        # Implement aggregation logic
    return df

def apply_comparison(comparison_steps, df):
    """Apply comparison steps on the dataframe."""
    for step in comparison_steps:
        st.write("🔍 **Applying Comparison:**", step)
        if "rank-ordering" in step:
            df = compare_regions(df)  # ✅ Updated to use compare_regions
        elif "rank change" in step:
            df = calculate_odr_ranks(df)  # ✅ Updated to use calculate_odr_ranks
    return df

def apply_trend_analysis(trend_analysis_steps, df):
    """Apply trend analysis steps on the dataframe."""
    for step in trend_analysis_steps:
        st.write("🔍 **Applying Trend Analysis:**", step)
        if "product-level comparison" in step:
            df = compare_product_level_odr(df)  # ✅ Updated to compare product-level ODR
        elif "rank change" in step:
            df = rank_change_regions(df)  # ✅ Updated to identify rank changes
        else:
            df = calculate_trends(df)  # ✅ Updated to calculate ODR trends
    return df

def apply_segmentation(segmentation_steps, df):
    """Apply segmentation steps on the dataframe."""  # ✅ NEW: Added segmentation handler
    for step in segmentation_steps:
        st.write("🔍 **Applying Segmentation:**", step)
        df = segment_regions(df)
    return df

def apply_prediction(prediction_steps, df):
    """Apply prediction steps on the dataframe using HybridLLM."""
    for step in prediction_steps:
        st.write("🔍 **Applying Prediction:**", step)
        prediction = predict_future_odr(df)  # ✅ Updated to use predict_future_odr
        st.write("🔮 **Prediction Result:**", prediction)
    return prediction

def generate_visualization(query, df):
    """Generate visualization using PandasAI."""
    try:
        smart_df = SmartDataframe(df)
        fig = smart_df.chat(query, visualize=True, llm=llm)
        return fig
    except Exception as e:
        st.write(f"❌ PandasAI Visualization Error: {e}")
        return None
