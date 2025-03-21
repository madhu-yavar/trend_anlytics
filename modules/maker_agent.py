
from langchain.agents import initialize_agent, Tool
from models.hybrid_llm import HybridLLM
from pandasai import SmartDataframe
import streamlit as st
import json

# Initialize HybridLLM
llm = HybridLLM()

# Function for PandasAI Analysis using SmartDataframe
def pandasai_tool(input, df):
    if df.empty:
        return "Error: The DataFrame is empty. Please check your filters."

    st.write("🔍 Filtered DataFrame Shape:", df.shape)
    st.dataframe(df.head())
    
    try:
        sdf = SmartDataframe(df.copy(), config={"llm": llm})
        response = sdf.chat(input)
        
    except Exception as e:
        st.error(f"Error in PandasAI processing: {e}")
        response = f"PandasAI Error: {str(e)}"
    
    return response

# Function for HybridLLM Analysis
def ask_hybridllm_debug(input, df):
    if df.empty:
        return "Error: The DataFrame is empty. Please check your filters."

    # ===== DEBUG 1: Verify DF reaches this function =====
    st.write("🛠️ DEBUG - DF received by HybridLLM tool:")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(3))
    
    # ===== DEBUG 2: Verify serialization works =====
    try:
        df_sample = df.head(3).to_dict(orient="records")
        df_sample_str = json.dumps(df_sample, indent=2)
        st.write("🛠️ DEBUG - Serialized DataFrame sample:", df_sample_str)
    except Exception as e:
        st.error(f"Serialization error: {str(e)}")
        return "Data conversion failed"

    # ===== DEBUG 3: Show full prompt being sent to LLM =====
    prompt = f"""You are an expert data analyst working with this dataset:
    {df_sample_str}
    
    Time periods are labeled as month-year (e.g., 'Sep 18' = Q3 2018).
    Provide a detailed analysis of: '{input}'
    
    1. First, calculate required metrics
    2. Then, compare metrics and draw conclusions
    3. Clearly separate technical steps from final insights
    """
    st.write("🛠️ DEBUG - Full prompt sent to HybridLLM:", prompt)

    # ===== DEBUG 4: Verify LLM receives the prompt ===== 
    try:
        response = llm.generate(prompt)
        st.write("🛠️ DEBUG - Raw LLM response:", response)
        return response
    except Exception as e:
        st.error(f"LLM processing error: {str(e)}")
        return "LLM analysis failed"

# Function for HybridLLM Analysis (Updated)
def ask_hybridllm(input, df):
    if df.empty:
        return "Error: The DataFrame is empty. Please check your filters."

    # 1. Convert DataFrame to list of dictionaries
    df_sample = df.to_dict(orient="records")
    df_sample_str = json.dumps(df_sample, indent=2)
    

    # 2. Wrap prompt in a list to match LLM's expected input format
    prompt = [
        f"""You are an expert data analyst working with this dataset:
        {df_sample_str}
        
        Time periods are labeled as month-year (e.g., 'Sep 18' = Q3 2018).
        Provide a detailed analysis of: '{input}'
        
        1. First, calculate required metrics
        2. Then, compare metrics and draw conclusions
        3. Clearly separate technical steps from final insights
        4. Do not provide code snippets"""
    ]

    try:
        # 3. Pass as list and extract first response
        response = llm.generate(prompt)
        return response.generations[0][0].text  # Adapt this based on your HybridLLM's response format
        
    except Exception as e:
        st.error(f"LLM processing error: {str(e)}")
        return "Analysis failed"
    
def maker_agent(query, df):
    # Define tools inside the agent function to capture current df
    tools = [
        Tool(
            name="PandasAI", 
            func=lambda input: pandasai_tool(input, df),  # Captures df
            description="Use for data processing, filtering, and visualization. Handles DataFrame operations."
        ),
        Tool(
            name="HybridLLM", 
            func=lambda input: ask_hybridllm(input, df),  # Captures df
            description="Use for complex analysis, trend explanations, and business insights. Handles natural language queries."
        ),
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent_type="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True
    )

    try:
        # First try PandasAI directly
        st.write("🧠 Attempting LLM Analysis...")
        LLM_response = ask_hybridllm(query, df)
        
        # Check for failure indicators
        if any(error in LLM_response for error in ["Error:", "I am sorry", "No code found"]):
            st.warning("⚠️ LLM couldn't handle query, switching to PandasAI...")
            return agent.run(query)  # Fallback to PandasAI
        return LLM_response
        
    except Exception as e:
        st.error(f"Critical LLM Error: {str(e)}")
        st.write("🚨 Falling back to PandasAI Analysis...")
        return agent.run(query)  # Fallback to PandasAI