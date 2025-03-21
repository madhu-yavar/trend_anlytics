import sys
import os

# Add root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import ODR calculator with absolute path
from Analytics_bot.modules.odr_calculator import (
    calculate_odr, calculate_trends, compare_regions,
    calculate_odr_ranks, top_odr_regions, predict_future_odr
)

# Remove the try/except import block

from langchain.agents import AgentExecutor, initialize_agent, Tool
from models.hybrid_llm import HybridLLM
from pandasai import SmartDataframe
import streamlit as st
import logging
import pandas as pd
import json
import re
#import modules.odr_calculator as odr_calculator
import json
#/Users/yavar/Desktop/HDFC_Loan_Poc/Analytics_bot/modules/odr_calculator.py
# from modules.odr_calculator import (
#     calculate_odr, 
#     calculate_trends,
#     compare_regions,
#     calculate_odr_ranks,
#     top_odr_regions,
#     predict_future_odr
# )

# Initialize HybridLLM with strict parameters
#llm = HybridLLM()

class HybridLLMWrapper(HybridLLM):
    def ask(self, prompt, context=None):
        return self.generate(prompt=prompt, context=context)

llm = HybridLLMWrapper(temperature=0, max_length=1000)
logging.basicConfig(filename='maker_errors.log', level=logging.WARNING)

# Constants
VALID_QUERY_TYPES = ["filtering", "aggregation", "comparison", 
                    "trend_analysis", "rank_ordering", "visualization", "prediction"]
MAX_ITERATIONS = 7  # Increased for complex operations

# Enhanced Data Validation
def validate_data_context(df):
    """Strict validation of dataframe structure and values"""
    required_columns = {"Pools", "Region", "Bad_Count", "Total_Count", "Period", "ODR"}
    #Pools	Region	Total_Count	Bad_Count	Period	ODR
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing critical columns: {missing}")
    
    if df.empty:
        raise ValueError("Empty dataframe provided")
    
    if df['Total_Count'].eq(0).any():
        raise ZeroDivisionError("Total_Count contains zero values")

# Enhanced PandasAI Tool with Strict Type Handling
def pandasai_tool(query, df):
    """Executes PandasAI operations with strict validation"""
    try:
        # Type sanitization and validation
        df = df.copy()
        numeric_cols = ['Total_Count', 'Bad_Count', 'ODR']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        validate_data_context(df)
        
        # Configure SmartDataframe with strict rules
        sdf = SmartDataframe(
            df,
            config={
                "llm": llm,
                "enable_cache": False,
                "max_retries": 2,
                "custom_instructions": f"""
                    STRICT RULES:
                    1. Use only these columns: {list(df.columns)}
                    2. Never invent new columns or aggregations
                    3. Validate filters against: 
                       Region: {df['Region'].unique()}
                       Pools: {df['Pools'].unique()}
                    4. Always generate executable code
                """
            }
        )
        
        response = sdf.chat(query)
        
        # Code validation
        if not hasattr(sdf, 'last_code') or not sdf.last_code:
            raise ValueError("No executable code generated")
        
        if any(kw in sdf.last_code.lower() for kw in ["lambda", "apply", "eval"]):
            raise ValueError("Unsafe code pattern detected")
            
        st.write("🔍 Generated Code:", sdf.last_code)
        return response
        
    except Exception as e:
        st.error(f"❌ PandasAI Failure: {str(e)}")
        log_error(query, df, str(e))
        raise  # Trigger fallback

# Structured HybridLLM Analysis with ODR Integration
def hybrid_analysis(query, df):
    """Executes structured analysis with ODR validation"""
    try:
        # Get structured response
        response = llm.generate(prompt=template, context=df.to_json())
        
        # Parse and validate
        parsed = json.loads(response)
        if not validate_odr_results(parsed, df):
            return execute_odr_fallback(query, df)
            
        return parsed
        
    except json.JSONDecodeError:
        return execute_odr_fallback(query, df)

def classify_query(query):
    """Classifies query type using regex patterns"""
    patterns = {
        "filtering": r"filter|where|select",
        "aggregation": r"sum|average|total",
        "comparison": r"compare|versus|vs",
        "trend_analysis": r"trend|over time",
        "rank_ordering": r"rank|order|top|bottom",
        "visualization": r"chart|graph|plot",
        "prediction": r"predict|forecast"
    }
    
    for qtype, pattern in patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            return qtype
    return "unknown"

def build_analysis_template(query_type, df):
    """Generates strict output templates based on query type"""
    base_template = f"""
    About the data:
    - Data is already filtered and cleaned with relevant data for analysis.
    - ODR (Overall Defaulter Rate) is calculated in the dataset.
    - Time periods: September 2018, December 2018, March 2019, June 2019, September 2019
    - Region: A, AC, unknown, Z, etc.
    - Pools: P50, P26, etc.

    You MUST respond in JSON format using ONLY these columns: {list(df.columns)}
    All values MUST exist in the data:
    - Region: {df['Region'].unique()}
    - Pools: {df['Pools'].unique()}
    - Period: {df['Period'].unique()}
    """
    
    templates = {
        "filtering": f"""{base_template}
        {{"operation": "filter", "criteria": [], "result_samples": []}}""",
        
        "aggregation": f"""{base_template}
        {{"operation": "aggregate", "groups": [], "metrics": {{}}}}""",
        
        "comparison": f"""{base_template}
        {{"comparison": {{"base": "", "target": ""}}, "metrics": []}}""",
        
        "trend_analysis": f"""{base_template}
        {{"trend_periods": [], "metrics": {{}}, "change_percentage": ""}}""",
        
        "rank_ordering": f"""{base_template}
        {{"ranking_by": "", "top_n": "", "bottom_n": ""}}""",
    }
    
    return templates.get(query_type, base_template)

# Enhanced Maker Agent Core

def maker_agent(query, df):
    """Orchestrates analysis with robust error handling"""
    tools = [
        Tool(
            name="PandasAI_Analysis",
            func=lambda q: pandasai_tool(q, df),
            description="For data operations requiring code generation"
        ),
        Tool(
            name="HybridLLM_Analysis",
            func=lambda q: hybrid_analysis(q, df),
            description="For business analysis requiring ODR validation"
        )
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent="structured-chat-zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=MAX_ITERATIONS,
        early_stopping_method="generate"
    )

    try:
        response = agent.run(input={
            "input": query,
            "data_schema": str(df.dtypes.to_dict()),
            "sample_data": str(df.head(2).to_dict())
        })
        
        return standardize_output(response)
        
    except Exception as e:
        st.error(f"🔥 Critical Failure: {str(e)}")
        return emergency_fallback(query, df)

def maker_agent_old(query, df):
    """Orchestrates analysis with robust error handling"""
    tools = [
        Tool(
            name="PandasAI_Analysis",
            func=lambda q: pandasai_tool(q, df),
            description="For data operations requiring code generation"
        ),
        Tool(
            name="HybridLLM_Analysis",
            func=lambda q: hybrid_analysis(q, df),
            description="For business analysis requiring ODR validation"
        )
    ]

    agent = AgentExecutor.from_agent_and_tools(
        agent = initialize_agent(
            tools,
            llm,
            agent="structured-chat-zero-shot-react-description",  # Not agent_type
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=MAX_ITERATIONS,
            early_stopping_method="generate"
        ),
        tools=tools,
        max_iterations=MAX_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )

    try:
        response = agent.run({
            "input": query,
            "data_schema": str(df.dtypes.to_dict()),
            "sample_data": str(df.head(2).to_dict())
        })
        
        validated_response = standardize_output(response)
        
        if validate_final_output(validated_response, df):
            return validated_response
            
        return execute_odr_fallback(query, df, classify_query(query))
        
    except Exception as e:
        st.error(f"🔥 Critical Failure: {str(e)}")
        return emergency_fallback(query, df)  # Now properly defined
    


def validate_final_output(response, df):
    """Final output validation layer"""
    if isinstance(response, dict):
        return all([
            check_data_presence(response, df),
            check_value_ranges(response, df),
            check_column_references(response, df)
        ])
    return False

# Emergency Fallback System
def execute_odr_fallback(query, df, query_type):
    """Executes ODR calculator-based fallback"""
    from odr_calculator import (
        calculate_odr_ranks, compare_regions, 
        calculate_trends, group_pools_by_count
    )
    
    handlers = {
        "rank_ordering": calculate_odr_ranks,
        "comparison": compare_regions,
        "trend_analysis": calculate_trends,
        "filtering": group_pools_by_count
    }
    
    handler = handlers.get(query_type, calculate_odr)
    result = handler(df)
    
    log_error(query, df, f"Fallback to {handler.__name__}")
    return {
        "result": result.to_dict() if hasattr(result, 'to_dict') else result,
        "metadata": {
            "fallback_used": True,
            "calculation_method": handler.__name__
        }
    }

# Logging System
def log_error(query, df, error_msg):
    """Centralized error logging"""
    logging.warning(f"""
    ERROR DETAILS:
    Query: {query}
    DataFrame Shape: {df.shape}
    Columns: {list(df.columns)}
    Error: {error_msg}
    """)

def emergency_fallback(query, df):
    """Final safety net for complete failures"""
    from odr_calculator import calculate_odr
    return {
        "result": calculate_odr(df).to_dict(),
        "metadata": {
            "emergency": True,
            "warning": "Complete analysis failure. Raw ODR data returned"
        }
    }

def check_data_presence(response, df):
    """Verify all mentioned data points exist in DataFrame"""
    checks = []
    if 'numerical_results' in response:
        results = response['numerical_results']
        checks.append(all(pool in df['Pools'].unique() for pool in results.get('pool_names', [])))
        checks.append(all(region in df['Region'].unique() for region in results.get('regions', [])))
    return all(checks)

def check_value_ranges(response, df):
    """Validate numerical values against DataFrame ranges"""
    if 'numerical_results' in response:
        results = response['numerical_results']
        max_odr = df['ODR'].max()
        return all(0 <= val <= max_odr for val in results.get('odr_values', []))
    return True

def check_column_references(response, df):
    """Ensure only valid columns are referenced"""
    valid_columns = set(df.columns)
    mentioned = set()
    if 'calculation_steps' in response:
        for step in response['calculation_steps']:
            mentioned.update(re.findall(r'\b[A-Za-z_]+\b', step))
    return mentioned.issubset(valid_columns)

def standardize_output(response):
    """Ensure consistent output structure"""
    if isinstance(response, dict):
        return response
    return {
        "analysis": str(response),
        "metadata": {
            "format_warning": "Non-structured response",
            "original_type": type(response).__name__
        }
    }

def validate_odr_results(response, df):
    """Enhanced validation using multiple checks"""
    valid = True
    
    # 1. Check numerical values against ODR calculator
    if 'odr_values' in response:
        calculated = calculate_odr(df)['ODR'].tolist()
        valid &= np.allclose(response['odr_values'], calculated, atol=0.01)
        
    # 2. Check region/pool existence
    if 'regions' in response:
        valid &= set(response['regions']).issubset(df['Region'].unique())
        
    if 'pools' in response:
        valid &= set(response['pools']).issubset(df['Pools'].unique())
        
    # 3. Check temporal consistency
    if 'periods' in response:
        valid &= set(response['periods']).issubset(df['Period'].unique())
        
    return valid