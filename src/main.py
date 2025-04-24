import streamlit as st
from components.user_input_handler import handle_user_input
from utils.initialize_variables import initialialization, reset_chat
from components.dataframe_agent import handle_excel_upload_and_chat
from utils.global_config_setup import glob_vars, MAX_RETRY_LIMIT
from openai._exceptions import RateLimitError
import os
import pandas as pd

def main() -> None:
    """
    Main function to handle the initialization and processing of user input and Excel file upload.
    This function performs the following steps:
    1. Calls the initialization function.
    2. Handles the Excel file upload and chat interaction.
    3. Processes the result of the upload and chat interaction.
    4. Displays appropriate messages and handles user input based on the result.
    The function handles different scenarios:
    - If there is a valid JSON response and prompt, it processes the user input.
    - If there is an exception, it displays an error message.
    - If there is only a JSON response, it writes the response as a chat message.
    - If the retry count reaches the maximum limit, it displays a message indicating the inability to generate a graph and shows the dataframe table.
    Note: This function uses Streamlit for displaying messages and handling user interactions.
    """

    initialialization()
    try:
        result = handle_excel_upload_and_chat()
    except RateLimitError as e:
        st.error(f"Token Limit exceeded the permitted limit. Please repharse the prompt and make provide specific details.")
        return
    if result:
        prompt, response_str,retry_count,exception,tool_name = result
        st.session_state.simple_chat_history.append({"role": "assistant", "content": response_str})

    else:
        retry_count = 0
        prompt = None
        response_str = None
        exception = None  
        tool_name = None

    if response_str and prompt and tool_name == "fetch_count_changes_or_plot":
        with st.spinner("Processing your request. Please wait..."):
            handle_user_input(prompt, response_str)   
        reset_button_key = "Reset Chat Button"
        st.button("Reset Chat", key=reset_button_key, help="Click to reset chat history", type="primary",on_click=reset_chat)

    elif exception:
        st.error(exception)
    elif response_str:
        if tool_name == "fetch_employees_or_names":
            import io
            response_str = response_str.replace("```csv\n", "").replace("\n```", "")

            try:
                df = pd.read_csv(io.StringIO(response_str))
            except:
                # df = response_str
                st.write("Dataframe Table")
            st.write(df)
            st.session_state.messages.append({"role": "assistant", "content": df})
        else:
            st.chat_message("assistant").write(response_str)
            st.session_state.messages.append({"role": "assistant", "content": response_str})


    elif retry_count == MAX_RETRY_LIMIT:
        st.markdown("<div style='border:2px solid black; padding: 10px; text-align: center;'><strong>Dataframe Table</strong></div>", unsafe_allow_html=True)
        st.write("\n\n")
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.write(glob_vars.response_df)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 18px; text-align: center; color: red;'>Unable to generate a graph for the provided data.</div>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()