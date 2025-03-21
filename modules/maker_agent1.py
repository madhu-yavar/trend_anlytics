from langchain.agents import initialize_agent, Tool
from models.hybrid_llm import HybridLLM
from pandasai import SmartDataframe
import streamlit as st

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
        
        if hasattr(sdf, 'last_code') and sdf.last_code:
            st.write("Generated Code from PandasAI:", sdf.last_code)
        else:
            st.write("No generated code available.")
        
    except Exception as e:
        st.error(f"Error in PandasAI processing: {e}")
        response = f"PandasAI Error: {str(e)}"
    
    return response

# Function for HybridLLM Analysis
def ask_hybridllm(input, df):
    if df.empty:
        return "Error: The DataFrame is empty. Please check your filters."

    prompt = f"""You are an expert data analyst working with this dataset:
    {df.head()}
    Time periods are labeled as month-year (e.g., 'Sep 18' = Q3 2018).
    Provide a detailed analysis of: '{input}'
    
    1. First, calculate required metrics
    2. Then, compare metrics and draw conclusions
    3. Clearly separate technical steps from final insights
    """
    return llm.ask(prompt, context=df)

def maker_agent(query, df):
    # Define tools inside the agent function to capture current df
    tools = [
        Tool(
            name="PandasAI", 
            func=lambda input: pandasai_tool(input, df),  # Now properly captures df
            description="Use for data processing, filtering, and visualization. Handles DataFrame operations."
        ),
        Tool(
            name="HybridLLM", 
            func=lambda input: ask_hybridllm(input, df),  # Now properly captures df
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
        st.write("🧠 Attempting PandasAI Analysis...")
        pandasai_response = pandasai_tool(query, df)
        
        # Check for failure indicators
        if any(error in pandasai_response for error in ["Error:", "I am sorry", "No code found"]):
            st.warning("⚠️ PandasAI couldn't handle query, switching to HybridLLM...")
            return agent.run(query)  # Now passing string input
        return pandasai_response
        
    except Exception as e:
        st.error(f"Critical PandasAI Error: {str(e)}")
        st.write("🚨 Falling back to HybridLLM Analysis...")
        return agent.run(query)  # Now passing string input