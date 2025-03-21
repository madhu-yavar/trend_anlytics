import streamlit as st
import pandas as pd
import json
import time
import os
from modules.data_loader import load_data
from modules.data_filter import apply_advanced_filtering
from modules.maker_agent import maker_agent
from modules.checker_agent import checker_agent
from st_aggrid import AgGrid, GridOptionsBuilder

# Function to log feedback
def log_feedback(query, analysis, rating, reason=None):
    """Log user feedback for analysis quality"""
    feedback = {
        "query": query,
        "analysis": analysis,
        "rating": rating,
        "reason": reason,  # Capture reason for negative feedback
        "timestamp": pd.Timestamp.now().isoformat()
    }
    # Save feedback to logs folder
    os.makedirs("logs", exist_ok=True)
    log_file_path = "logs/feedback_log.json"
    with open(log_file_path, "a") as f:
        f.write(json.dumps(feedback) + "\n")
    st.success("Thank you for your feedback!")

def main():
    # Logo and title
    logo_path = "/Users/yavar/Desktop/HDFC_Loan_Poc/Analytics_bot/yavarlogo.png"
    st.image(logo_path, width=100)  
    st.title(" Trend Analytics Agent")

    # Initialize progress bar and processing times
    overall_progress = st.progress(0)
    processing_times = {}

    # Load data
    df = load_data()
    overall_progress.progress(10)

    # # ✅ Load and display the Excel file
    excel_path = "Copy of Test_Data_LLM_backend.xlsx"  # Update with actual path if needed
    if not excel_path:
        st.error("No file found!")
    else:
        # Read all sheets
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names  # Get all sheet names
        
        st.write("### Data Preview")

        # Create tabs for each sheet
        tabs = st.tabs(sheet_names)

        for tab, sheet in zip(tabs, sheet_names):
            with tab:
                dfe = pd.read_excel(xls, sheet_name=sheet)
                st.dataframe(dfe, use_container_width=True)

    # User query input
    query = st.text_input("🔍 Enter your query for advanced analysis:")
    
    if st.button("🔍 Ask") and query:
        try:
            # Stage 1: Filtering
            with st.spinner("🔍 Applying filters..."):
                start_time = time.time()
                filtered_df, _ = apply_advanced_filtering(df.copy(), query)
                processing_times['filtering'] = time.time() - start_time
                overall_progress.progress(30)

            # Stage 2: Maker Agent
            maker_progress = st.progress(0)
            with st.spinner("🧠 Processing with Maker Agent..."):
                start_time = time.time()
                analysis_result = maker_agent(query, filtered_df)
                processing_times['maker'] = time.time() - start_time
                maker_progress.progress(100)
                overall_progress.progress(60)

            # Stage 3: Checker Agent
            checker_progress = st.progress(0)
            with st.spinner("✅ Validating with Checker Agent..."):
                start_time = time.time()
                summary = checker_agent(query, analysis_result, filtered_df)
                processing_times['checker'] = time.time() - start_time
                checker_progress.progress(100)
                overall_progress.progress(90)

            # Final display
            st.success(f"✅ Analysis completed in {sum(processing_times.values()):.1f}s")
            overall_progress.progress(100)

            # Show timing breakdown
            with st.expander("⏱️ Performance Metrics"):
                st.write(f"🔍 Filtering: {processing_times['filtering']:.1f}s")
                st.write(f"🧠 Maker Agent: {processing_times['maker']:.1f}s")
                st.write(f"✅ Checker Agent: {processing_times['checker']:.1f}s")

            # Feedback Section
            st.write("### Was this analysis helpful?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Good Analysis", key="good_analysis_button"):
                    log_feedback(query, analysis_result, rating=5)
            with col2:
                if st.button("👎 Needs Improvement", key="needs_improvement_button"):
                    reason = st.text_area("Please provide a reason or scope for improvement:")
                    if reason:
                        log_feedback(query, analysis_result, rating=1, reason=reason)

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            overall_progress.progress(0)

if __name__ == "__main__":
    main()
