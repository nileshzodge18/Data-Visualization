import io
import json
import os
import random
import time

import pandas as pd
import streamlit as st


from utils.global_config_setup import AppConfig,AppConfigKeys, glob_vars, PromptContextInformation, AzureLLM,MAX_RETRY_LIMIT
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor,AgentType
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from components.create_dataframe_agent import create_dataframe_agent
from langchain_core.messages import SystemMessage,HumanMessage

messages = []

FETCH_COUNT_CHANGES_OR_PLOT = "fetch_count_changes_or_plot"
FETCH_EMPLOYEES_OR_NAMES = "fetch_employees_or_names"

@tool
def fetch_count_changes_or_plot() -> None:
    """
    This function will be used for fetching the count or number of employees 
    or for plotting charts or graphs based on the provided prompt.
    Args:
        None
    Returns:
        None: The function does not return any value.
    """
    pass

@tool
def fetch_employees_or_names() -> None:
    """
    This tool is used for fetching employee details or names based on the provided prompt.
    The prompt should specify whether to retrieve employee names, roles, or other related details.
    Args:
        None
    Returns:
        None: The function does not return any value.
    """
    pass

@tool
def greet_the_human_user() -> None:
    """
    Returns a greeting message.
    Returns:
        None: The function does not return any value.
    """
    pass

@tool
def describe_application_capabilities_and_functions() -> None:
    """
    Provides a description of the application's functionality and capabilities.
    Returns:
        None: The function does not return any value.
    """
    pass

def determining_tool_name(llm,prompt) -> str:
    """
    This function demonstrates tool calling using the Langchain library.
    It initializes a list of tools, sets up a prompt template, and invokes the tools with a sample input.
    The function also handles the output and displays it in the Streamlit app.
    Args:
        llm: The language model to be used for tool calling.
    Returns:
        None
    """
    tools = [fetch_count_changes_or_plot, fetch_employees_or_names]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                f"You are a helpful assistant that choses the tool based on the user query and recent chat history. "
                f"The available tools are {tools}. "
            )
        ),
        SystemMessage(
            content=f"The chat history is given below: (Recent messages are at the end)\n\n"
        ),
        MessagesPlaceholder(
            variable_name="simple_chat_history"
        ),
        HumanMessage(
            content="{input}"
        )
    ])

    llm_with_tools = llm.bind_tools(tools)
    chain = prompt | llm_with_tools

    result = chain.invoke({
        "input": prompt,
        "simple_chat_history": st.session_state.simple_chat_history
    }
    )
    tool_name = ""

    for tool_call in result.tool_calls:
        tool_name = tool_call["name"]
        return tool_name

def lower_case_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all column names of the DataFrame to lowercase.
    Args:
        df (pd.DataFrame): The DataFrame whose column names need to be converted to lowercase.
    Returns:
        None: The function modifies the DataFrame in place.
    """
    df.columns = [col.lower() for col in df.columns]
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower()
    
    return df

def generate_token_responses(chain,user_query: str,conversation_summary_memory: ConversationSummaryMemory):
    for partial_resp in chain.stream({
        "input": user_query,
        "chat_history": conversation_summary_memory.buffer,
        "simple_chat_history": st.session_state.simple_chat_history,
    }):
        token = partial_resp.content
        yield token

def validate_response(response: dict, response_format: str) -> bool:
    """
    Validates the response from the agent based on the specified response format.
    For CSV format, it checks if the response output is not empty, contains at least two columns,
    and that all columns except the first one contain strictly numerical values.
    For JSON format, it checks if the response output is a valid JSON.
    Args:
        response (dict): The response from the agent.
        response_format (str): The expected format of the response ('CSV' or 'JSON').
    Returns:
        bool: True if the response is valid, False otherwise.
    """

    if response_format == 'CSV':
        try:
            if not response['output']:
                st.error("The response output is empty.")
                return False
                
            response['output'] = response['output'].replace("```csv\n", "").replace("\n```", "")
            tempdf = pd.read_csv(io.StringIO(response['output']))
            glob_vars.response_df = tempdf
            st.write(tempdf)
            keys = list(tempdf.keys())
            if len(keys) < 2:
                st.error("The CSV must contain at least two columns.")
                return False
            
            if pd.api.types.is_numeric_dtype(tempdf[keys[0]]):
                st.error("The first column must not contain strictly numerical values.")
                return False
            
            for key in keys[1:]:
                if not pd.api.types.is_numeric_dtype(tempdf[key]):
                    st.error(f"Column '{key}' does not contain strictly numerical values.")
                    return False
                
            intermediate_steps = response['intermediate_steps']

            if intermediate_steps is None:

                query_string = intermediate_steps[0][0]
                query_string = str(query_string)
                extracted_query = query_string.split("{'query':")[1].split("}")[0]
                if "\ndata =" in extracted_query or "\ndata=" in extracted_query or ".head(" in extracted_query or ".tail(" in extracted_query:
                    st.error("Dataframe should not be created again. Use the existing dataframe")
                    st.error(f"Query : {extracted_query}")
                    st.error(f"Can't be plotted into Graph/Chart")
                    return False

        except Exception as e:
            st.error(f"Invalid CSV format. \nException : {e}")
            try:
                st.error(f"Response : {response['output']}")
            except Exception as e:
                st.error(f"Execption Caught : {e}")
            return False
        
    else:
        try:
            json.loads(response['output'])
        except Exception as e:
            st.error(f"Invalid JSON format. \nEcxeption : {e}")
        return False
    return True

def check_agent_variables(agent: AgentExecutor) -> None:

    st.write(f"Agent Name : {agent.name}")
    st.write("\n\n\n")

    st.write(f"Agent Type : {agent.agent}")
    st.write("\n\n\n")
    
    st.write(f"Agent Input Keys : {agent.input_keys}")
    st.write("\n\n\n")

    st.write(f"Agent Output Keys : {agent.output_keys}")
    st.write("\n\n\n")

    st.write(f"Agent Verbosity : {agent.verbose}")
    st.write("\n\n\n")

    st.write(f"Agent Intermediate Steps : {agent.return_intermediate_steps}")
    st.write("\n\n\n")

    st.write(f"Agent Memory: {agent.memory}")
    st.write("\n\n\n")

    st.write(f"Agent Parsing Errors : {agent.handle_parsing_errors}")
    st.write("\n\n\n")


def get_response_from_agent(prompt_context: PromptContextInformation,user_query: str,formatted_chat_history: str, tool_name: str) -> dict:
    """
    Creates a pandas dataframe agent and invokes it with the given input.
    This function initializes a pandas dataframe agent with specific parameters,
    including the language model (llm), dataframe (df), agent type, verbosity,
    intermediate steps return, and allowance for dangerous code execution.
    It then constructs an input dictionary containing the prompt and chat history,
    and invokes the agent with this input.
    Returns:
        dict: The response from the agent after processing the input.
    """
    history = ChatMessageHistory()
    history.add_messages(st.session_state.user_query)
    history.add_messages(st.session_state.agent_response)

    conversation_summary_memory = ConversationSummaryMemory(llm=glob_vars.llm, chat_memory=history,return_messages=True,input_key="input",output_key="output")
    agent_executor_kwargs = {
        "memory": conversation_summary_memory,
        "handle_parsing_errors": True
    }



    # agent_type = "tool-calling"
    agent_type = "openai-tools"
    # agent = create_pandas_dataframe_agent(
    #     llm=glob_vars.llm,
    #     df=glob_vars.df,
    #     agent_type=agent_type,
    #     verbose=True,
    #     return_intermediate_steps=True,
    #     allow_dangerous_code=True,
    #     include_df_in_prompt=False,
    #     prefix=prompt_context.get_prefix_str(formatted_chat_history),
    #     suffix=prompt_context.get_suffix_str(),
    #     agent_executor_kwargs = agent_executor_kwargs

    # )

    agent = create_dataframe_agent(
        llm=glob_vars.llm,
        df=glob_vars.df,
        agent_type=agent_type,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_code=True,
        include_df_in_prompt=False,
        prefix=prompt_context.get_prefix_str(formatted_chat_history),
        suffix=prompt_context.get_suffix_str(),
        tool_name=tool_name,
        agent_executor_kwargs = agent_executor_kwargs

    )

    memory = ChatMessageHistory(session_id="test-session")
    

    use_chat_history_with_Runnable = False


    if use_chat_history_with_Runnable == True:
        agent_with_chat_history = RunnableWithMessageHistory(
            agent,
            # This is needed because in most real world scenarios, a session id is needed
            # It isn't really used here because we are using a simple in memory ChatMessageHistory
            lambda session_id: memory,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        agent_with_chat_history.invoke(
            {"input": glob_vars.final_prompt_message},
             config={"configurable": {"session_id": "<foo>"}},
        )
    else:

        input = {
            "input": user_query,
            "chat_history": st.session_state.simple_chat_history
        }
        
    
        response = agent.invoke(input = input)


    st.session_state.chat_history.append({"Please find conversation summary here": conversation_summary_memory.buffer})
    


    return response


# def get_response_from_agent2(prompt_context,user_query) -> dict:

#     agent_type = "openai-tools"

#     st.write(f"Type of chat_history : {type(glob_vars.chat_history)}")
    
#     history = ChatMessageHistory()
#     history.add_messages(glob_vars.user_query)
#     history.add_messages(glob_vars.agent_response)

#     conversation_summary_memory = ConversationSummaryMemory(llm=glob_vars.llm, chat_memory=history)
#     agent_executor_kwargs = {
#         "memory": conversation_summary_memory,
#         "handle_parsing_errors": True
#     }




#     st.write(type(agent_executor_kwargs))

#     st.write(f"Template Message Pre : {prompt_context.template_msg_pre}")
#     st.write(f"Template Message Post : {prompt_context.template_msg_post}")

#     agent = create_pandas_dataframe_agent(
#         llm=glob_vars.llm,
#         df=glob_vars.df,
#         agent_type=agent_type,
#         verbose=True,
#         return_intermediate_steps=True,
#         allow_dangerous_code=True,
#         include_df_in_prompt=False,
#         agent_executor_kwargs=agent_executor_kwargs,
#         prefix = prompt_context.template_msg_pre,
#         suffix= prompt_context.template_msg_post
#     )

#     input = {
#         "input": user_query,
#     }

#     # input = {
#     #     "context": SystemMessage(content = prompt_context.context_template),
#     #     "input": SystemMessage(content  = glob_vars.user_query),
#     #     "chat_history": SystemMessage(content = glob_vars.chat_history),
#     #     "template_msg_pre": SystemMessage(content=prompt_context.template_msg_pre),
#     #     "input": SystemMessage(content=glob_vars.user_query),
#     #     "template_msg_post": SystemMessage(content=prompt_context.template_msg_post)
#     # }

#     check_agent_variables(agent)
#     response = agent.invoke(input=input)




#     glob_vars.user_query = user_query
#     glob_vars.agent_response = response['output']

#     return response


def handle_excel_upload_and_chat() -> tuple:
    """
    Handles the upload of an Excel file, processes it, and facilitates a chat interaction for data visualization.
    This function allows users to either upload a new Excel file or select an existing one from a specified directory.
    It reads the file into a pandas DataFrame, displays the first few rows, and provides information about the DataFrame.
    Additionally, it enables a chat interface where users can input queries related to the data visualization.
    Returns:
        tuple: A tuple containing the user query (str), the response from the agent (str), the retry count (int), 
               and an exception message (str) if any error occurs.
    """

    app_config = AppConfig()
    prompt_context = PromptContextInformation()
    df = None
    excel_level = ""
    context_input = None
    response_format = "CSV"
    with st.sidebar:
        st.header(f"Add your excel documents!")

        st.subheader("Context for the Document")

        APP_PATH = app_config.get_value(AppConfigKeys.APP_SETTINGS, AppConfigKeys.AppSettings.APP_PATH)
        files_path = os.path.join(APP_PATH, "files")

        
        context_input = st.text_area(
            "Please provide any additional context or information for the document below:",
            value=prompt_context.context_template,
            height=150,
            placeholder="Enter context here..."
        )


        if context_input == "" or context_input is not None:
            context_input = "Please find for the dataframe below Context below:\n\n" + context_input
            prompt_context.context_template = context_input

        upload_option = st.radio(
            "Choose an option:",
            ("Upload a new file", "Select from existing files")
        )   

        excel_file = None

        excel_level = st.selectbox(
            "Does the Excel file contain single-level columns or double-level columns?",
            ("Single-level columns", "Double-level columns")
        )
        header = 0
        if excel_level == "Double-level columns":
            header = [0, 1]
        else:
            header = 0

        multilevel_excel_columns = True if excel_level == "Double-level columns" else False

        prompt_context.set_multi_level_excel_columns(multilevel_excel_columns)


        if upload_option == "Upload a new file":
            excel_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls", "xlsm", "xlsb", "odf", "ods", "odt"])
            if excel_file is not None:
                save_path = os.path.join(files_path, excel_file.name)
                with open(save_path, "wb") as f:
                    f.write(excel_file.getbuffer())
        else:
            existing_files = [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))]
            selected_file = st.selectbox("Select a file", ["Select an existing file"] + existing_files)
            if selected_file and selected_file != "Select an existing file":
                excel_file_path = os.path.join(files_path, selected_file)
                with open(excel_file_path, "rb") as f:
                    excel_file = f.read()

        if excel_file is not None:
            if isinstance(excel_file, bytes):
                if excel_file_path.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(excel_file), sep=',',header=header)
                elif excel_file_path.endswith('.xls'):
                    df = pd.read_excel(io.BytesIO(excel_file), engine='xlrd',header=header)
                elif excel_file_path.endswith(('.xlsx', '.xlsm', '.xlsb')):
                    df = pd.read_excel(io.BytesIO(excel_file), engine='openpyxl',header=header)
                elif excel_file_path.endswith(('.odf', '.ods', '.odt')):
                    df = pd.read_excel(io.BytesIO(excel_file), engine='odf',header=header)
                else:
                    st.error("Unsupported file type")
                    return
            else:
                if excel_file.name.endswith('.csv'):
                    df = pd.read_csv(excel_file,header=header)
                elif excel_file.name.endswith('.xls'):
                    df = pd.read_excel(excel_file, engine='xlrd',header=header)
                elif excel_file.name.endswith(('.xlsx', '.xlsm', '.xlsb')):
                    df = pd.read_excel(excel_file, engine='openpyxl',header=header)
                elif excel_file.name.endswith(('.odf', '.ods', '.odt')):
                    df = pd.read_excel(excel_file, engine='odf',header=header)
                else:
                    st.error("Unsupported file type")
                    return
                
            st.write(df.head())
            st.info(f"This excel sheet contain {df.shape[0]} rows and {df.shape[1]} columns",icon=":material/info:")

    disabled = False
    if df is None:
        disabled=True
        st.info("Chatbox is disabled because the Excel sheet is either not uploaded or not loaded correctly.", icon=":material/info:")
    user_query = st.chat_input(disabled = disabled)
    if user_query:
        st.chat_message("user").write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.simple_chat_history.append({"role": "user", "content": user_query})
        time.sleep(0.25)
        response_valid = False
        retry_count = 0

        azure_llm = AzureLLM()
        llm = azure_llm.get_AzureChatOpenAI_llm()

        formatted_chat_history = "\n\nBelow is the chat history (most recent messages are at the end):\n\n" + str(st.session_state.chat_history)

        prompt_context.set_column_names(df)
        # st.write(prompt_context.context_template)
        final_prompt_message = prompt_context.generate_detailed_prompt(formatted_chat_history,user_query,openai_message=False)

        glob_vars.response_format_instruction = prompt_context.template_msg_post
        glob_vars.excel_context = prompt_context.context_template

        glob_vars.final_prompt_message = final_prompt_message




        try:
            glob_vars.df = df
        except UnboundLocalError as e:
            exception_message = f"Please ensure that the excel sheet is uploaded correctly."
            return None, None,0,exception_message, "",""
        except Exception as e:
            exception_message = f"Error loading the dataframe. Exception: {e}. Please ensure that the excel sheet is uploaded."
            return None, None,0,exception_message, "",""
        glob_vars.llm = llm


        tool_name = determining_tool_name(llm,user_query)
        if(tool_name == "fetch_count_changes_or_plot"):
            with st.status("Fetching data for Visualization..."):

                st.write(f"Tool Name : {tool_name}")

                while(response_valid == False and retry_count < MAX_RETRY_LIMIT):
                    response = get_response_from_agent(prompt_context,user_query,formatted_chat_history,tool_name) ## This is the original code
                    if "graph requested" in response['output']:
                        st.write(f"""The response from the agent:\n{st.session_state.response_output}""")
                        response = st.session_state.response_obj

                    response_valid = validate_response(response, response_format)
                    retry_count += 1

                if not response_valid and retry_count == MAX_RETRY_LIMIT:
                    st.error(f"Maximum number of attempts {MAX_RETRY_LIMIT} reached.")
                    return None, None, retry_count, None,""
                
                if response_valid == True:
                    # chat_history_user = {"role": "user", "content": user_query}
                    # st.session_state.chat_history.append(chat_history_user)

                    # chat_history_response = {"role": "assistant", "content": response['output']}

                    # st.session_state.chat_history.append(chat_history_response)
                    # st.chat_message("assistant").write(f"The response from the agent:\n")
                    st.write(f"```csv\n{response['output']}\n```")

                    st.session_state.response_obj = response
                    st.session_state.user_query = user_query
                    st.session_state.agent_response = response['output']
                    
                if response_valid:
                    st.success("Response is valid and parsable")

                return user_query, response['output'], retry_count, None, tool_name
        
        elif tool_name == "fetch_employees_or_names":
            with st.status("Fetching data for Visualization..."):

                st.write(f"Tool Name : {tool_name}")

                response = get_response_from_agent(prompt_context,user_query,formatted_chat_history,tool_name) ## This is the original code
                st.session_state.response_obj = response
                # response_valid = validate_response(response, response_format)
                

                # if response_valid == True:
                #     tool_name = "fetch_count_changes_or_plot"

                return None, response['output'], 0, None, tool_name
        
        # elif tool_name == "greet_the_human_user":
        #     history = ChatMessageHistory()
        #     history.add_messages(st.session_state.user_query)
        #     history.add_messages(st.session_state.agent_response)

        #     conversation_summary_memory = ConversationSummaryMemory(llm=glob_vars.llm, chat_memory=history,return_messages=True,input_key="input",output_key="output")

        #     prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             f"You are a helpful assistant that greets the human user. Today's date is {time.strftime('%Y-%m-%d')}",
        #         ),                
        #         (
        #             "placeholder", "{simple_chat_history}"
        #         ),
        #         ("human", "{input}"),
        #     ]
        #     )

        #     chain = prompt | llm
        #     response = chain.invoke(
        #         {
        #             "input": user_query,
        #             "simple_chat_history": st.session_state.simple_chat_history,
        #         }
        #     )
        #     st.session_state.user_query = user_query
        #     st.session_state.agent_response = response.content

        #     st.session_state.chat_history.append({"Please find conversation summary here": conversation_summary_memory.buffer})
        #     return None, response.content, 0, None, tool_name

        # elif tool_name == "describe_application_capabilities_and_functions":
        #     history = ChatMessageHistory()
        #     history.add_messages(st.session_state.user_query)
        #     history.add_messages(st.session_state.agent_response)
            
        #     conversation_summary_memory = ConversationSummaryMemory(llm=glob_vars.llm, chat_memory=history,return_messages=True,input_key="input",output_key="output")
        #     conversation_summary_memory.save_context({"input": st.session_state.user_query}, {"output": st.session_state.agent_response})

        #     prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             f"You are a helpful assistant that helps with fetching the data from excel sheet and visualizing them into graphs or charts with matplotlib. Availble charts/graphs are Bar,Scatter,Line and Pie chart. You can do nothing else apart from what is mentioned here. Excel sheet is already provided. Today's date is {time.strftime('%Y-%m-%d')}",
        #         ),
        #         (
        #             "system",
        #             "Please find the conversation summary below:\n\n{chat_history}"
        #         ),                (
        #             "placeholder", "{simple_chat_history}"
        #         ),
        #         ("human", "{input}"),
        #     ]
        #     )

        #     chain = prompt | llm
        #     response = chain.invoke(
        #         {
        #             "input": user_query,
        #             "chat_history": conversation_summary_memory.buffer,
        #             "simple_chat_history": st.session_state.simple_chat_history,

        #         }
        #     )
        #     st.session_state.user_query = user_query
        #     st.session_state.agent_response = response.content

        #     st.session_state.chat_history.append({"Please find conversation summary here": conversation_summary_memory.buffer})
        #     return None, response.content, 0, None, tool_name
        
        else:
            history = ChatMessageHistory()
            history.add_messages(st.session_state.user_query)
            history.add_messages(st.session_state.agent_response)
            
            conversation_summary_memory = ConversationSummaryMemory(llm=glob_vars.llm)
            conversation_summary_memory.save_context({"input": st.session_state.user_query}, {"output": st.session_state.agent_response})
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(
                    content=(
                        f"You are a helpful assistant that works with a resource report of employees containing records "
                        f"of their managers, ELTs, roles, assignments, etc., and plots various graphs/charts (Bar, Scatter, "
                        f"Line, and Pie Chart). If the user asks queries unrelated to the dataframe data or requests something "
                        f"not present in the chat history, respond appropriately. Data is already loaded so don't ask user to "
                        f"do it again. Today's date is {time.strftime('%Y-%m-%d')}."
                    )
                ),
                SystemMessage(
                    content="Please find the conversation summary below:\n\n{chat_history}"
                ),
                MessagesPlaceholder(
                    variable_name="simple_chat_history"
                ),
                HumanMessage(
                    content="{input}"
                )
            ])

            chain = prompt | llm

            msg_placeholder = st.empty()
            response = ""
            for token in generate_token_responses(chain,user_query,conversation_summary_memory):
                response += token
                msg_placeholder.markdown(response + "▌")

            msg_placeholder.empty()

            st.session_state.user_query = user_query
            st.session_state.agent_response = response

            st.session_state.chat_history.append({"Please find conversation summary here": conversation_summary_memory.buffer})
            return None, response, 0, None, tool_name            

