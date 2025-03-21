from langchain.agents import initialize_agent, Tool
from models.hybrid_llm import HybridLLM
from pandasai import SmartDataframe
import streamlit as st
import json
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
from pandasai import SmartDataframe
from pandasai.llm.google_gemini import GoogleGemini
import google.generativeai as genai
import plotly.express as px
from pandas.api.types import is_datetime64_any_dtype



#GOOGLE_AI_STUDIO_API_KEY =  "AIzaSyAtLHyxyJY9L1sdwkYIIVihfg70j7OlmFM" #cogitet
GOOGLE_AI_STUDIO_API_KEY = "AIzaSyAe8rheF4wv2ZHJB2YboUhyyVlM2y0vmlk"

#gemin_llm = GoogleGemini(api_key="GOOGLE_AI_STUDIO_API_KEY", model="gemini-2.0-flash")
gemin_llm = GoogleGemini(api_key=GOOGLE_AI_STUDIO_API_KEY, model="gemini-2.0-flash")
genai.configure(api_key=GOOGLE_AI_STUDIO_API_KEY)



# Initialize HybridLLM
llm = HybridLLM()

# Visualization Templates
VISUALIZATION_TEMPLATES = {
    'time_series': {
        'columns': ['Period', 'ODR'],
        'function': lambda df: df.groupby('Period')['ODR'].mean().plot(kind='line', title='ODR Trend Over Time')
    },
    'distribution': {
        'columns': ['ODR'],
        'function': lambda df: df['ODR'].plot(kind='hist', bins=20, title='ODR Distribution')
    }
}

# Feedback Logging (Added)
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

def checker_agent(query,analysis_text, df):
    """Generates executive summary with enhanced visualizations"""
    st.subheader("Final Analysis Report")
    
    # 1. Create summary structure with data grounding
    summary_prompt = f"""You are a quality assurance analyst. Review this analysis. Ignore the {query}.
    
    Maker's Analysis:
    {analysis_text}
    
    Available Data Metrics:
    - Time Range: {df['Period'].min()} to {df['Period'].max()}
    - Total Records: {len(df):,}
    - Average ODR: {df['ODR'].mean():.4f}
    - Total Bad Count: {df['Bad_Count'].sum():,}
    
    Your Task:
    Do not start "Okay, here's my review of the provided analysis:...." It is a report, maintain the report format. Start the report with the key findings and recommendations.
    The data has ODR (Overall Defaulter Rate), Bad_Count (BC), Total_Count (TC) and time periods (Sep 18, Dec 18, Mar 19, Jun19,Sep 19), Regions like A,AC,unknown,z etc and Pools like P50, P26 etc. Use the same nomencalture in your report.
    1. Summarise the insights from the analysis_text with 3-5 bullet points it must be the primary answer for  the query. It MUST have only FINDINGS of the query.
    2. Highlight 3 key numerical insights
    3. Provide actionable recommendations based on the analysis in 2-3 points.
    4. The report should not be in first person.
    """
    
    # 2. Generate structured summary
    with st.spinner("🔍 Quality Checking Analysis..."):
        structured_summary = llm.generate([summary_prompt]).generations[0][0].text
    
    st.markdown(structured_summary)
    
    # 3. Generate intelligent visualizations
    with st.expander(" Advanced Visual Analysis"):
        generate_contextual_visualizations(query, df)
    
    return structured_summary

def generate_contextual_visualizations(query, df):
    """Agent to generate visualization suggestions and render them"""
    st.subheader("Visualizer Agent")
    
    # Initialize SmartDataframe with Gemini
    smart_df = SmartDataframe(df, config={"llm": gemin_llm})
    
    # Improved visualization prompt with structured output enforcement
    visualization_prompt = f"""As a data visualization expert analyzing this query: "{query}", create 1 or 2 visualization suggestions using this data sample:
    {df.head().to_string()}
    
    Required format (JSON only):
    {{
        "visualizations": [
            {{
                "type": "chart_type",
                "x": "column_name",
                "y": "column_name",
                "rationale": "analysis_reason",
                "priority": 1-3
            }}
        ]
    }}
    
    Rules:
    1. First visualization must show time trends if 'Period' exists
    2. Include ODR distribution analysis
    3. 1 or 2 visualization is enough.
    4. Use only these chart types: line, bar, scatter, histogram, box
    }}"""
    
    with st.spinner("🧠 Generating smart visualization suggestions..."):
        try:
            # Get and parse Gemini response
            response = smart_df.chat(visualization_prompt)
            
            # Improved parsing with fallbacks
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                suggestions = json.loads(json_str)['visualizations']
            else:
                raise ValueError("Invalid JSON format")
            
            st.success("AI-generated visualization plan:")
            st.json(suggestions)
            
        except Exception as e:
            st.warning(f"Enhanced fallback activated{str(e)}")
            suggestions = dynamic_fallback(query, df)
            #st.write("Adaptive visualization suggestions:")
            #st.json(suggestions)
    
    render_visualizations(suggestions, df, query)

def render_visualizations(suggestions, df, query):  # Added query parameter
    """Render visualizations using Plotly with enhanced logic"""
    cols = st.columns(2)
    
    for idx, viz in enumerate(sorted(suggestions, key=lambda x: x['priority'])):
        with cols[idx % 2]:
            try:
                fig = create_adaptive_chart(viz, df, query)  # Pass query
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to render {viz['type']}: {str(e)}")
                st.code(f"Failed visualization: {viz}")

def create_adaptive_chart(viz, df, query):
    """Query-aware chart builder"""
    # Handle date conversion first
    if viz['x'] == 'Period' and df[viz['x']].dtype == 'object':
        df = df.copy()
        df['Period'] = pd.to_datetime(df['Period'], format='%b %y', errors='coerce')
    
    # Base chart configuration
    fig = px.scatter(title=f"Analysis: {query[:45]}...")
    color_palette = px.colors.qualitative.Plotly
    
    try:
        if viz['type'] == 'line':
            fig = px.line(df, x=viz['x'], y=viz['y'], 
                         title=f"Trend: {viz['y']} Over Time",
                         markers=True,
                         color_discrete_sequence=[color_palette[0]])
            
        elif viz['type'] == 'bar':
            fig = px.bar(df, x=viz['x'], y=viz['y'],
                        title=f"Comparison: {viz['y']} by {viz['x']}",
                        color=viz['x'],
                        color_discrete_sequence=color_palette)
            
        elif viz['type'] == 'histogram':
            fig = px.histogram(df, x=viz['x'], 
                             title=f"Distribution of {viz['x']}",
                             nbins=20,
                             color_discrete_sequence=[color_palette[2]])
            
        # Add regional formatting
        if 'region' in query.lower() and viz['x'] == 'Region':
            fig.update_xaxes(categoryorder='total descending')
            
    except Exception as e:
        st.error(f"Chart error: {str(e)}")
        
    return fig

def dynamic_fallback(query, df):
    """Enhanced query-aware fallback"""
    query = query.lower()
    suggestions = []
    
    # 1. Time analysis detection
    time_kws = ['trend', 'time', 'month', 'quarter', 'year', 'progress']
    if any(kw in query for kw in time_kws) and 'Period' in df.columns:
        suggestions.append({
            "type": "line",
            "x": "Period",
            "y": "ODR",
            "rationale": "Time series analysis",
            "priority": 1
        })
    
    # 2. Regional analysis 
    region_kws = ['region', 'zone', 'area', 'location']
    if any(kw in query for kw in region_kws) and 'Region' in df.columns:
        suggestions.append({
            "type": "bar",
            "x": "Region",
            "y": "ODR",
            "rationale": "Regional comparison",
            "priority": 1
        })
    
    # 3. Pool analysis
    pool_kws = ['pool', 'group', 'cluster']
    if any(kw in query for kw in pool_kws) and 'Pool' in df.columns:
        suggestions.append({
            "type": "box",
            "x": "Pool",
            "y": "ODR",
            "rationale": "Pool performance",
            "priority": 2
        })
    
    # 4. Default safety net
    if not suggestions:
        suggestions.extend([
            {"type": "line", "x": "Period", "y": "ODR", "priority": 1},
            {"type": "histogram", "x": "ODR", "priority": 2}
        ])
    
    return sorted(suggestions, key=lambda x: x['priority'])[:2]



