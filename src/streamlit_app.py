import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.Graph.workflow import build_graph
from langchain_core.messages import HumanMessage

# Page config
st.set_page_config(
    page_title="Multi-Agent Data Science Workflow",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = build_graph()

if 'file_path' not in st.session_state:
    st.session_state.file_path = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title("🤖 Multi-Agent Data Science Workflow")
st.markdown("Upload your CSV and let AI agents analyze your data!")

# Sidebar - File Upload
with st.sidebar:
    st.header("📁 Data Upload")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.file_path = str(file_path)
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Show data preview
        with st.expander("📊 Data Preview"):
            df = pd.read_csv(file_path, nrows=5)
            st.dataframe(df)
            st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns (showing first 5 rows)")
    
    st.divider()
    
    # Quick Actions
    st.header("⚡ Quick Actions")
    
    if st.button("📈 EDA Summary"):
        if st.session_state.file_path:
            st.session_state.quick_query = "Give me a summary of the columns and check for missing values."
        else:
            st.warning("Please upload a file first!")
    
    if st.button("🧹 Data Quality Check"):
        if st.session_state.file_path:
            st.session_state.quick_query = "Check data quality: missing values, duplicates, and outliers."
        else:
            st.warning("Please upload a file first!")
    
    if st.button("🔍 Column Analysis"):
        if st.session_state.file_path:
            st.session_state.quick_query = "Analyze all columns: data types, unique values, and distributions."
        else:
            st.warning("Please upload a file first!")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main area - Chat Interface
st.header("💬 Chat with Your Data")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle quick query button clicks
if 'quick_query' in st.session_state and st.session_state.quick_query:
    query = st.session_state.quick_query
    st.session_state.quick_query = None  # Clear it
    
    # Add to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Process query
    if st.session_state.file_path:
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                inputs = {
                    "messages": [HumanMessage(content=query)],
                    "file_path": st.session_state.file_path,
                    "llm_calls": 0
                }
                
                result = st.session_state.agent.invoke(inputs)
                response = result['messages'][-1].content
                
                st.markdown(response)
                st.caption(f"🔄 LLM Calls: {result.get('llm_calls', 0)}")
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

# Chat input
if prompt := st.chat_input("Ask me anything about your data..."):
    if not st.session_state.file_path:
        st.warning("⚠️ Please upload a CSV file first!")
    else:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                inputs = {
                    "messages": [HumanMessage(content=prompt)],
                    "file_path": st.session_state.file_path,
                    "llm_calls": 0
                }
                
                result = st.session_state.agent.invoke(inputs)
                response = result['messages'][-1].content
                
                st.markdown(response)
                st.caption(f"🔄 LLM Calls: {result.get('llm_calls', 0)}")
        
        # Add assistant response to chat
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

# Footer
st.divider()
st.caption("🚀 Powered by LangGraph & LangChain")