from models.hybrid_llm import HybridLLM
import streamlit as st
import re  # Import regex for better text processing

llm = HybridLLM()

def identify_steps(query, filters, unique_periods):
    """Identify steps needed for analysis using the LLM."""
    planning_prompt = f"""
    You are an expert data analyst. 

    About the data:
    - Data i am providing to you is already filtered and cleaned. Means it has only the relevent data for analysis.
    - ODR (Overall Defaulter Rate) is already calculated.
    - Time periods: September 2018, December 2018, March 2019, June 2019, September 2019
    - Region: A, AC, unknown, Z, etc.
    - Pools: P50, P26, etc.
    
    You have received the following query: "{query}".
    The extracted filters are:
    - Region: {filters["Region"]}
    - Pools: {filters["Pools"]}
    - Period: {unique_periods}

    Identify the steps required to perform the analysis, considering the available data. 
    Break down the task into clear, actionable steps categorized as:
    1. Data Filtering
    2. Data Aggregation
    3. Comparison
    4. Trend Analysis
    5. Visualization
    6. Rank-Ordering Analysis (only if explicitly requested)
    7. Prediction (if needed)

    Clearly indicate if rank-ordering checks are NOT needed.
    Present the steps in a structured format for further execution.
    """

    steps = llm._call(planning_prompt)
    
    if not steps or steps.strip() == "":
        st.error("⚠️ No steps identified. Please check the query or try again.")
        return []

    #st.write("📝 **Identified Steps:**")
    #for step in steps.split('\n'):
        #st.markdown(f"- {step.strip()}")
    
    return steps

def normalize_text(text):
    """Normalize text for consistent processing."""
    text = text.lower().strip()  # Lowercase and trim whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def match_keywords(step, keywords):
    """Check if any keyword is present in the step."""
    return any(keyword in step for keyword in keywords)

def categorize_steps(steps):
    """Categorize steps into filtering, aggregation, comparison, trend analysis, rank-ordering, and visualization."""
    categorized_steps = {
        "filtering": [],
        "aggregation": [],
        "comparison": [],
        "trend_analysis": [],
        "rank_ordering": [],
        "visualization": [],
        "prediction": []
    }
    
    # Define keyword sets for each category
    keyword_map = {
        "filtering": ["filter", "select", "subset", "group by"],
        "aggregation": ["aggregate", "sum", "average", "calculate", "count", "total", "mean", "median"],
        "comparison": ["compare", "difference", "versus", "relative to", "against"],
        "trend_analysis": ["trend", "time series", "change over time", "increase", "decrease", "growth", "decline"],
        "rank_ordering": ["rank", "order", "position", "ranking", "rank-ordering", "shift in rank"],
        "visualization": ["visualize", "chart", "graph", "plot", "display", "histogram", "heatmap", "bar chart"],
        "prediction": ["predict", "forecast", "future", "projection", "estimate", "model", "regression"]
    }
    
    for step in steps.split('\n'):
        step = normalize_text(step)
        categorized = False
        
        # Match step against keywords for each category
        for category, keywords in keyword_map.items():
            if match_keywords(step, keywords):
                # Special Condition for Rank-Ordering:
                # Only categorize as rank-ordering if 'change' or 'shift' is mentioned in the context
                if category == "rank_ordering" and ("change" in step or "shift" in step):
                    categorized_steps["rank_ordering"].append(step)
                    categorized = True
                    break
                elif category != "rank_ordering":
                    categorized_steps[category].append(step)
                    categorized = True
                    break
        

    return categorized_steps

