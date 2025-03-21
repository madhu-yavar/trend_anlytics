from models.hybrid_llm import HybridLLM
import pandas as pd

from pandasai import SmartDataframe  # Only SmartDataframe is needed
#from pandasai.llm.ollama import Ollama

#llm = Ollama(model="deepseek-r1:14b")
#pandas_ai = PandasAI(llm)

llm = HybridLLM()

def analyze_correlation(df: pd.DataFrame, target='ODR'):
    """Analyze correlation of ODR with other variables using SmartDataframe."""
    sdf = SmartDataframe(df,config={"llm": llm})
    query = f"Find the correlation of {target} with other variables."
    result = sdf.chat(query)
    return result

def predict_future_odr(df: pd.DataFrame, period_column='Period', target='ODR'):
    """Predict future ODR trends using HybridLLM."""
    trend_data = df[[period_column, target]].dropna().to_dict(orient='records')
    prompt = f"""
    Given the ODR trend data: {trend_data},
    predict the ODR for the next 2 quarters.
    """
    prediction = llm._call(prompt)
    return prediction

def advanced_query(df: pd.DataFrame, query: str):
    """Handle advanced queries using SmartDataframe with improved error handling and validation."""
    try:
        # ✅ Initialize SmartDataframe
        sdf = SmartDataframe(df, config={"llm": llm})
        
        # ✅ Perform the query
        result = sdf.chat(query)
        
        # ✅ Validate Result
        if result is None or (isinstance(result, str) and not result.strip()):
            st.warning("⚠️ No meaningful result returned. The query might be too vague or unsupported.")
            return "No relevant information found."
        
        return result
    
    except Exception as e:
        # ✅ Catch and log any errors
        st.error(f"❌ An error occurred during the advanced query: {e}")
        return f"Error: {e}"


def calculate_weighted_odr(df: pd.DataFrame, weight_column='Total_Count', target='ODR'):
    """Calculate weighted ODR."""
    df['Weighted_ODR'] = (df[target] * df[weight_column]) / df[weight_column].sum()
    weighted_odr = df['Weighted_ODR'].sum()
    return weighted_odr

def compare_regions(df: pd.DataFrame, regions, target='ODR'):
    """Compare ODR trends between selected regions."""
    comparison = {}
    for region in regions:
        region_df = df[df['Region'] == region]
        sdf = SmartDataframe(region_df, config={"llm": llm})
        query = f"Show the trend of {target} over time."
        comparison[region] = sdf.chat(query)
    return comparison

def segment_analysis(df: pd.DataFrame, target='ODR'):
    """Segment data into groups with common ODR trends."""
    sdf = SmartDataframe(df, config={"llm": llm})
    query = f"Segment the data into groups with similar {target} trends."
    segments = sdf.chat(query)
    return segments

def macro_variable_correlation(df: pd.DataFrame, macro_var='GDP', target='ODR'):
    """Check ODR correlation with macroeconomic variables."""
    sdf = SmartDataframe(df, config={"llm": llm})
    query = f"Find the correlation between {target} and {macro_var}."
    correlation = sdf.chat(query)
    return correlation
