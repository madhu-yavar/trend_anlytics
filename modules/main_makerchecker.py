
import streamlit as st
import pandas as pd
from modules.data_loader import load_data
from modules.data_filter import apply_advanced_filtering
from modules.makerChecker import maker_agent

import json
import pandas as pd
import time
import seaborn as sns


def log_feedback(query, analysis, rating):
    """Log user feedback for analysis quality"""
    feedback = {
        "query": query,
        "analysis": analysis,
        "rating": rating,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    # Save feedback to a file or database (example: JSON file)
    with open("feedback_log.json", "a") as f:
        f.write(json.dumps(feedback) + "\n")
    st.success("Thank you for your feedback!")



def main():
    st.title("📊 ODR Analytics Dashboard")
    
    # Initialize progress bar
    overall_progress = st.progress(0)
    processing_times = {}

    df = load_data()
    overall_progress.progress(10)
    
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
                #st.write("🧠 Maker Agent Analysis Result:",analysis_result)
                processing_times['maker'] = time.time() - start_time
                maker_progress.progress(100)
                overall_progress.progress(60)

            # # Stage 3: Checker Agent
            # checker_progress = st.progress(0)
            # with st.spinner("✅ Validating with Checker Agent..."):
            #     start_time = time.time()
            #     summary = checker_agent(query,analysis_result, filtered_df)
            #     processing_times['checker'] = time.time() - start_time
            #     checker_progress.progress(100)
            #     overall_progress.progress(90)

            # Final display
            st.success(f"✅ Analysis completed in {sum(processing_times.values()):.1f}s")
            overall_progress.progress(100)
            
            # Show timing breakdown
            with st.expander("⏱️ Performance Metrics"):
                st.write(f"🔍 Filtering: {processing_times['filtering']:.1f}s")
                st.write(f"🧠 Maker Agent: {processing_times['maker']:.1f}s")
                st.write(f"✅ Checker Agent: {processing_times['checker']:.1f}s")

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            overall_progress.progress(0)
# Feedback Buttons (Moved inside the try block to ensure analysis_result is defined)
        st.write("### Was this analysis helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Good Analysis", key="good_analysis_button"):
                log_feedback(query, analysis_result, rating=5)
            with col2:
                if st.button("👎 Needs Improvement", key="needs_improvement_button"):
                    log_feedback(query, analysis_result, rating=1)
if __name__ == "__main__":
    main()