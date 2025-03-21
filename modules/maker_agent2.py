from langchain.agents import initialize_agent, Tool
from models.hybrid_llm import HybridLLM
from pandasai import SmartDataframe
import streamlit as st
import logging

# Initialize HybridLLM
llm = HybridLLM()

# Modified Section: Enhanced PandasAI Tool with Type Handling
# -----------------------------------------
def pandasai_tool(input, df):
    if df.empty:
        return "Error: The DataFrame is empty. Please check your filters."

    # NEW: Type sanitization
    df = df.copy()
    numeric_cols = ['Total_Count', 'Bad_Count', 'ODR']
    for col in numeric_cols:
        if col in df.columns:
            # Convert to float and handle NaNs
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    st.write("🔍 Filtered DataFrame Shape:", df.shape)
    st.dataframe(df.head())
    
    try:
        validate_data_context(df)
        sdf = SmartDataframe(df, config={
            "llm": llm,
            "enable_cache": False,
            "custom_instructions": f"""
                You MUST:
                1. Use explicit column names from: {list(df.columns)}
                2. Handle numeric comparisons using pandas query syntax
                3. Never invent new columns or aggregations
            """
        })
        
        response = sdf.chat(input)
        
        # NEW: Code validation
        if hasattr(sdf, 'last_code'):
            if "lambda" in sdf.last_code or "apply" in sdf.last_code:
                raise ValueError("Unsafe code generated")
            st.write("Generated Code:", sdf.last_code)
            return response
        else:
            raise ValueError("No executable code generated")
            
    except Exception as e:
        st.error(f"PandasAI Error: {str(e)}")
        raise  # Trigger fallback
# -----------------------------------------

# Enhanced HybridLLM Analysis with Structured Output
# -----------------------------------------
def ask_hybridllm(input, df):
    structured_prompt = f"""You MUST respond in JSON format:
    {{
        "analysis": "Summary of findings",
        "calculation_steps": ["list", "of", "actual", "operations"],
        "numerical_results": {{
            "pool_names": [],
            "total_counts": [],
            "odr_values": []
        }},
        "data_source": "columns: {list(df.columns)}"
    }}

    Query: {input}
    
    Available Data Filters:
    - Region must be: {df['Region'].unique()}
    - Period must be: {df['Period'].unique()}
    - Total Count < 100
    
    Example Valid Response:
    {{
        "analysis": "3 pools in region AC meet criteria",
        "calculation_steps": [
            "Filter Region == 'AC'",
            "Filter Total_Count < 100",
            "Group by Pool",
            "Calculate ODR mean"
        ],
        "numerical_results": {{
            "pool_names": ["P26", "P45"],
            "total_counts": [85, 92],
            "odr_values": [4.2, 3.8]
        }}
    }}
    """
    
    try:
        response = llm.ask(structured_prompt, context=df.to_json())
        return json.loads(response)  # Force JSON parse
    except json.JSONDecodeError:
        return {"error": "Invalid response format"}
# -----------------------------------------

# Enhanced Maker Agent with Validation Layer
# -----------------------------------------
def maker_agent(query, df):
    tools = [
        Tool(
            name="PandasAI", 
            func=lambda input: pandasai_tool(input, df),
            description="For numerical operations and dataframe manipulations"
        ),
        Tool(
            name="HybridLLM_Analysis", 
            func=lambda input: ask_hybridllm(input, df),
            description="For structured business analysis with JSON output"
        )
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent_type="structured-chat-zero-shot-react-description",  # Better for structured data
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

    try:
        response = agent.run({
            "input": query,
            "data_columns": list(df.columns),
            "data_sample": df.head(2).to_dict()
        })
        
        # Validation Layer
        if not validate_response(response, df):
            raise ValueError("Validation failed")
            
        return response
        
    except Exception as e:
        st.error(f"Critical Error: {str(e)}")
        return odr_calculator.calculate(query, df)  # Final fallback

def validate_response(response, df):
    """Programmatic validation of results"""
    if isinstance(response, dict):
        if 'numerical_results' in response:
            # Check pool names exist
            invalid_pools = set(response['numerical_results']['pool_names']) - set(df['Pool'])
            if invalid_pools:
                raise ValueError(f"Invalid pools: {invalid_pools}")
            
            # Check values match data
            for pool in response['numerical_results']['pool_names']:
                pool_data = df[df['Pool'] == pool]
                if pool_data.empty:
                    raise ValueError(f"Pool {pool} not found")
                    
    return True
# -----------------------------------------


# NEW: Logging Hallucinations
# -----------------------------------------
def log_hallucination(query, erroneous_response, df):
    logging.error(f"""
    Hallucination Detected!
    Query: {query}
    Bad Response: {erroneous_response}
    DataFrame Context: {df.head()}
    """)
# -----------------------------------------




