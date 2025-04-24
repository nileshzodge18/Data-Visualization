import streamlit as st
import gc
from utils.global_config_setup import glob_vars
from streamlit.delta_generator import DeltaGenerator
from matplotlib.figure import Figure
from typing import Any
from langchain_core.messages import AIMessage


def reset_chat():
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.response_output = ""
    st.session_state.response_obj = ""   
    st.session_state.simple_chat_history = []   
    st.session_state.user_query = ""
    st.session_state.agent_response = ""  
    st.session_state.charts = []

    glob_vars.llm = None
    glob_vars.final_prompt_message = ""
    glob_vars.df = None
    glob_vars.prevResp = ""
    glob_vars.response_output = ""
    glob_vars.comma_seperated = True
    glob_vars.response_df = None
    glob_vars.excel_context = ""
    glob_vars.response_format_instruction = ""

    gc.collect()

widget_id = (id for id in range(1, 100000))

def initialialization() -> None:
    """
    Initializes various global variables and session state variables for the AI Chatbot application.

    This function sets up the initial state for the chatbot, including UI elements, messages, 
    and other necessary variables required for the chatbot to function properly.

    Returns:
        None
    """

    col1, col2 = st.columns([6,1])

    with col1:
        st.header("AI Chatbot for Generating and Visualizing Graphs For Excel Sheet 📈 ")

    with col2:
        st.button("Clear ↺",key=next(widget_id), on_click=reset_chat)

    st.chat_message("assistant").write("Hello! I am an AI Chatbot specialized in generating and visualizing graphs. How can I assist you today?")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        dict = {"role": "", "content": Any}
        st.session_state.chat_history = []

    if "response_output" not in st.session_state:
        st.session_state.response_output = ""

    if "response_obj" not in st.session_state:
        st.session_state.response_obj = ""

    if "simple_chat_history" not in st.session_state:
        st.session_state.simple_chat_history = []

    if "user_query" not in st.session_state:
        st.session_state.user_query = "Hi"
    
    if "agent_response" not in st.session_state:
       st.session_state.agent_response = "Hello"

    if "charts" not in st.session_state:
        st.session_state.charts = []  # This will hold a list of matplotlib Figure objects

    if "llm" not in glob_vars:
        glob_vars.llm = None

    if "final_prompt_message" not in glob_vars:
        glob_vars.final_prompt_message = ""

    if "df" not in glob_vars:
        glob_vars.df = None

    if "prevResp" not in glob_vars:
        glob_vars.prevResp = ""

    if "response_output" not in glob_vars:
        glob_vars.response_output = ""

    if "comma_seperated" not in glob_vars:
        glob_vars.comma_seperated = True

    if "response_df" not in glob_vars:
        glob_vars.response_df = None

    if "excel_context" not in glob_vars:
        glob_vars.excel_context = ""

    if "response_format_instruction" not in glob_vars:
        glob_vars.response_format_instruction = ""



    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            

    


    
