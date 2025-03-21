from langchain.agents import initialize_agent, Tool
from models.hybrid_llm import HybridLLM
from pandasai import SmartDataframe
import streamlit as st
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
def ask_hybridllm(input, df):
    if df.empty:
        return "Error: The DataFrame is empty. Please check your filters."

    # Convert DataFrame to list of dictionaries
    df_sample = df.to_dict(orient="records")
    df_sample_str = json.dumps(df_sample, indent=2)
    
    # Wrap prompt in a list to match LLM's expected input format
    prompt = [
        f"""You are an expert data analyst working with this dataset:
        {df_sample_str}
        
        Time periods are labeled as month-year (e.g., 'Sep 18' = Q3 2018).
        Provide a detailed analysis of: '{input}'
        
        1. First, calculate required metrics
        2. Then, compare metrics and draw conclusions
        3. Clearly separate technical steps from final insights"""
    ]

    try:
        # Pass as list and extract first response
        response = llm.generate(prompt)
        return response.generations[0][0].text  # Adapt this based on your HybridLLM's response format
        
    except Exception as e:
        st.error(f"LLM processing error: {str(e)}")
        return "Analysis failed"

# Agentic Checker/Summarizer
def checker_agent(query, analysis_text, df):
    """Agent to validate and summarize the analysis"""
    st.subheader("✅ Checker Agent Evaluation")
    
    # Create summary structure with query context
    summary_prompt = f"""You are a quality assurance analyst. Review this analysis for the query: "{query}"
    
    Maker's Analysis:
    {analysis_text}
    
    Available Data Metrics:
    - Time Range: {df['Period'].min()} to {df['Period'].max()}
    - Total Records: {len(df):,}
    - Average ODR: {df['ODR'].mean():.4f}
    - Total Bad Count: {df['Bad_Count'].sum():,}
    
    Your Task:
    1. Verify alignment between query and analysis
    2. Highlight 3 key numerical insights
    3. Identify any data limitations
    4. Format clearly with emojis
    
    Required Format:
    ✅ **Query Alignment**  
    [Assessment of how well analysis addresses query]
    
    🔍 **Key Metrics**  
    - [Metric 1 with context]
    - [Metric 2 with comparison]
    - [Metric 3 with trend]
    
    ⚠️ **Data Considerations**  
    - [Any data limitations]"""
    
    # Generate structured summary
    with st.spinner("🔍 Quality Checking Analysis..."):
        structured_summary = llm.generate([summary_prompt]).generations[0][0].text
    
    st.write(structured_summary)
    return structured_summary

# Agentic Visualizer
def visualizer_agent(query, df):
    """Agent to generate visualization suggestions and render them"""
    st.subheader("📊 Visualizer Agent")
    
    # Step 1: Ask LLM for visualization suggestions
    visualization_prompt = f"""You are a data visualization expert. For the query: "{query}", suggest the best visualizations for this dataset:
    
    Data Sample:
    {df.head().to_string()}
    
    Instructions:
    1. Suggest 2-3 visualizations that best answer the query
    2. Specify the chart type and required columns
    3. Provide a brief rationale for each visualization
    
    Required Format:
    ### Visualization 1
    - Type: [Chart Type]
    - Columns: [Required Columns]
    - Rationale: [Why this visualization is useful]
    
    ### Visualization 2
    - Type: [Chart Type]
    - Columns: [Required Columns]
    - Rationale: [Why this visualization is useful]"""
    
    with st.spinner("🧠 Generating visualization suggestions..."):
        visualization_suggestions = llm.generate([visualization_prompt]).generations[0][0].text
        st.write("### Visualization Suggestions")
        st.write(visualization_suggestions)
    
    # Step 2: Render visualizations based on suggestions
    with st.spinner("📈 Rendering visualizations..."):
        render_visualizations(query, df)

def render_visualizations(query, df):
    """Render visualizations based on query and data"""
    try:
        # Clean and prepare data
        df['Period'] = pd.to_datetime(df['Period'], format='%b %y')
        
        # Create visualization grid
        cols = st.columns(2)
        
        # Time-based analysis
        if any(kw in query.lower() for kw in ['trend', 'time', 'progress']):
            with cols[0]:
                fig, ax = plt.subplots(figsize=(10, 4))
                df.set_index('Period')['ODR'].resample('M').mean().plot(
                    kind='line', 
                    marker='o',
                    color='darkblue',
                    ax=ax
                )
                plt.fill_between(df.set_index('Period').resample('M').mean().index,
                               df.set_index('Period')['ODR'].resample('M').min(),
                               df.set_index('Period')['ODR'].resample('M').max(),
                               color='skyblue', alpha=0.3)
                plt.title("ODR Trend with Variability Band", weight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

        # Comparative analysis
        if 'region' in query.lower() and 'Region' in df.columns:
            with cols[1] if 'trend' in query.lower() else cols[0]:
                fig, ax = plt.subplots(figsize=(10, 4))
                region_data = df.groupby('Region').agg(
                    ODR=('ODR', 'mean'),
                    Bad_Count=('Bad_Count', 'sum')
                ).sort_values('ODR', ascending=False)
                
                sns.barplot(x=region_data.index, y='ODR', data=region_data, 
                           palette='viridis', ax=ax)
                plt.title("ODR by Region", weight='bold')
                plt.xticks(rotation=45)
                plt.ylabel("Average ODR")
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)

        # Default view
        if not any(kw in query.lower() for kw in ['trend', 'region']):
            with cols[0]:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(x=df['ODR'], palette='Set2', ax=ax)
                plt.title("ODR Distribution", weight='bold')
                plt.xlabel("ODR Value")
                plt.grid(True, axis='x', linestyle='--', alpha=0.7)
                st.pyplot(fig)

    except Exception as e:
        st.warning(f"Simplified view: {str(e)}")
        st.write("Raw analysis data:")
        st.dataframe(df[['Period', 'ODR', 'Bad_Count']].head())

# Unified Maker Agent with Checker and Visualizer
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
            LLM_response = agent.run(query)  # Fallback to PandasAI

        # Step 2: Checker Agent
        checker_response = checker_agent(query, LLM_response, df)
        
        # Step 3: Visualizer Agent
        visualizer_agent(query, df)
        
        return LLM_response
        
    except Exception as e:
        st.error(f"Critical LLM Error: {str(e)}")
        st.write("🚨 Falling back to PandasAI Analysis...")
        return agent.run(query)  # Fallback to PandasAI