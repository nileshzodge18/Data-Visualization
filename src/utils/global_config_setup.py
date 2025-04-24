from typing import MutableMapping, Any, Union, TypeAlias, Iterator
import tomllib
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_openai_messages
import pandas as pd


Key: TypeAlias = Union[str, int]

MAX_RETRY_LIMIT = 3
NEXT_LINE = "\n"
APP_CONFIG_PATH = "../.config/app_config.toml"



class AzureLLM():
    """
    A class to configure and interact with Azure's Language Model (LLM) services.
    Attributes:
        api_version (str): The API version for Azure LLM.
        azure_endpoint (str): The endpoint URL for Azure LLM.
        api_key (str): The API key for accessing Azure LLM.
        model (str): The model identifier for Azure LLM.
        temperature (float): The temperature setting for the model's response generation.
        streaming (bool): A flag indicating whether streaming is enabled.
    Methods:
        get_AzureChatOpenAI_llm():
            Returns an instance of AzureChatOpenAI configured with the provided settings.
    """

    def __init__(self) -> None:
        """
        Initializes the configuration settings for the application.

        Attributes:
            api_version (str): The API version used for Azure configuration.
            azure_endpoint (str): The Azure endpoint URL.
            api_key (str): The API key for accessing Azure services.
            model (str): The model configuration for Azure services.
            temperature (float): The temperature setting for the model.
            streaming (bool): Indicates if streaming is enabled.

        Returns:
            None
        """
        self.api_version = AppConfig().get_value(AppConfigKeys.AZURE_CONFIG, AppConfigKeys.AzureConfig.API_VERSION)
        self.azure_endpoint = AppConfig().get_value(AppConfigKeys.AZURE_CONFIG, AppConfigKeys.AzureConfig.AZURE_ENDPOINT)
        self.api_key = AppConfig().get_value(AppConfigKeys.AZURE_CONFIG, AppConfigKeys.AzureConfig.API_KEY)
        self.model = AppConfig().get_value(AppConfigKeys.AZURE_CONFIG, AppConfigKeys.AzureConfig.MODEL)
        self.temperature = AppConfig().get_value(AppConfigKeys.AZURE_CONFIG, AppConfigKeys.AzureConfig.TEMPERATURE)
        self.streaming = AppConfig().get_value(AppConfigKeys.AZURE_CONFIG, AppConfigKeys.AzureConfig.STREAMING)

    def get_AzureChatOpenAI_llm(self) -> AzureChatOpenAI:
        """
        Creates and returns an instance of AzureChatOpenAI with the configured parameters.

        Returns:
            AzureChatOpenAI: An instance of the AzureChatOpenAI class initialized with the 
                     specified API version, endpoint, API key, model, temperature, 
                     and streaming settings.
        """
        return AzureChatOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            streaming=self.streaming
        )



class GlobalVariables(MutableMapping[Key, Any]):
    """
    A class to represent a global variable storage that behaves like a dictionary.
    This class implements the MutableMapping interface, allowing it to be used
    as a dictionary for storing key-value pairs. It provides methods to get, set,
    delete, and iterate over items, as well as to get the length of the stored items.
    Methods
    -------
    __init__():
        Initializes the global variable storage.
    __getitem__(key: Any) -> Any:
        Retrieves the value associated with the given key.
    __setitem__(key: Any, value: Any) -> None:
        Sets the value for the given key.
    __delitem__(key: Any) -> None:
        Deletes the item associated with the given key.
    __iter__() -> Iterator[Any]:
        Returns an iterator over the stored items.
    __len__() -> int:
        Returns the number of items stored.
    __setattr__(name: str, value: Any) -> None:
        Sets an attribute on the instance.
    """
    def __init__(self) -> None:
        """
        Initializes a new instance of the class.

        This constructor sets up an empty dictionary to store configuration data.

        Returns:
            None
        """
        self._store = {}

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve an item from the internal store using the given key.

        Args:
            key (Any): The key to look up in the internal store.

        Returns:
            Any: The value associated with the given key in the internal store.
        """
        return self._store[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Set the value for a given key in the store.

        Args:
            key (Any): The key for which the value needs to be set.
            value (Any): The value to be set for the given key.

        Returns:
            None
        """
        self._store[key] = value

    def __delitem__(self, key: Any) -> None:
        """
        Remove the item with the specified key from the store.

        Parameters:
        key (Any): The key of the item to be removed.

        Returns:
        None
        """
        del self._store[key]

    def __iter__(self) -> Iterator[Any]:
        """
        Returns an iterator over the items in the store.

        Returns:
            Iterator[Any]: An iterator over the items in the store.
        """
        return iter(self._store)

    def __len__(self) -> int:
        """
        Return the number of items in the store.

        Returns:
            int: The number of items in the store.
        """
        return len(self._store)

    def __setattr__(self, name, value):
        """
        Override the default behavior of setting an attribute.

        Args:
            name (str): The name of the attribute.
            value (Any): The value to set for the attribute.

        Returns:
            None
        """
        super().__setattr__(name, value)


glob_vars = GlobalVariables()


class AppConfig():
    """
    AppConfig is a class responsible for loading and managing application configuration.
    Attributes:
        config (dict): A dictionary containing the configuration data.
    Methods:
        __init__(): Initializes the AppConfig instance and loads the configuration.
        _load_config(config_path: str) -> dict: Loads the configuration from a TOML file.
        get_value(table_name: str, key_name: str) -> Any: Retrieves a value from the configuration.
    """

    config: dict

    def __init__(self) -> None:
            """
            Initializes the configuration setup by loading the configuration file.

            This method sets the `config` attribute by loading the configuration
            from a TOML file located at "../.config/app_config.toml".

            Returns:
                None
            """

            self.config = self._load_config(
                os.path.abspath(APP_CONFIG_PATH)
            )

    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from a TOML file.

        Args:
            config_path (str): The path to the TOML configuration file.

        Returns:
            dict: The configuration data loaded from the file.
        """
        with open(config_path, "rb") as f:
            return tomllib.load(f)
        
    def get_value(self,table_name,key_name) -> Any:
        """
        Retrieve a value from the configuration.

        Args:
            table_name (str): The name of the table in the configuration.
            key_name (str): The key within the table to retrieve the value for.

        Returns:
            Any: The value associated with the specified table and key in the configuration.
        """
        return self.config[table_name][key_name]
    
class AppConfigKeys:
    """
    A class used to represent the configuration keys for the application.
    Attributes
    ----------
    APP_SETTINGS : str
        Key for application settings.
    AZURE_CONFIG : str
        Key for Azure configuration.
    Nested Classes
    --------------
    AppSettings:
        A class used to represent the keys for application settings.
        Attributes
        ----------
        APP_NAME : str
            Key for the application name.
        LOG_PATH : str
            Key for the log file path.
        LOG_LEVEL : str
            Key for the log level.
        APP_PATH : str
            Key for the application path.
        FILES_PATH : str
            Key for the files path.
    AzureConfig:
        A class used to represent the keys for Azure configuration.
        Attributes
        ----------
        AZURE_DEPLOYMENT : str
            Key for the Azure deployment.
        API_VERSION : str
            Key for the API version.
        AZURE_ENDPOINT : str
            Key for the Azure endpoint.
        API_KEY : str
            Key for the API key.
        MODEL : str
            Key for the model.
        TEMPERATURE : str
            Key for the temperature setting.
        STREAMING : str
            Key for the streaming setting.
    """
    
    APP_SETTINGS = "APP_SETTINGS"
    AZURE_CONFIG = "AZURE_CONFIG"
    class AppSettings:
        APP_NAME = "APP_NAME"
        LOG_PATH = "LOG_PATH"
        LOG_LEVEL = "LOG_LEVEL"
        APP_PATH = "APP_PATH"
        FILES_PATH = "FILES_PATH"
        EXCEL_CONTEXT = "EXCEL_CONTEXT"
        SAMPLE_PROMPT_WITH_INSTRUCTIONS = "SAMPLE_PROMPT_WITH_INSTRUCTIONS"

    class AzureConfig:
        AZURE_DEPLOYMENT = "AZURE_DEPLOYMENT"
        API_VERSION = "API_VERSION"
        AZURE_ENDPOINT = "AZURE_ENDPOINT"
        API_KEY = "API_KEY"
        MODEL = "MODEL"
        TEMPERATURE = "TEMPERATURE"
        STREAMING = "STREAMING"


class PromptContextInformation:
    """
    A class to encapsulate the context information and prompts for generating detailed responses.
    Attributes:
    -----------
    context_template : str
        Template describing the context of the data and its structure.
    context_prompt : str
        Prompt for the context.
    template_msg_pre : str
        Pre-message template for the bot's response.
    actual_template_format : str
        Actual template format for the response.
    template_msg_post : str
        Post-message template for the bot's response.
    graph_prompt : str
        Prompt for graph-related queries.
    greetings_prompt : str
        Prompt for greeting-related queries.
    random_prompt : str
        Placeholder for random prompts.
    csv_seperator : str
        Placeholder for CSV separator.
    thinking_instr : str
        Instructions for processing user queries.
    Methods:
    --------
    __init__():
        Initializes the PromptContextInformation with default values.
    generate_detailed_prompt(formatted_chat_history: str, user_query: str) -> str:
        Generates a detailed prompt based on the formatted chat history and user query.
    _generate_context(formatted_chat_history: str, user_query: str) -> str:
        Generates the context for the prompt based on the formatted chat history and user query.
    """


    context_template: str = ""
    context_prompt: str = ""
    template_msg_pre: str
    actual_template_format: str = ""
    template_msg_post: str = ""
    graph_prompt: str = ""
    greetings_prompt: str = ""
    random_prompt: str = ""
    csv_seperator: str = ""
    thinking_instr: str = ""
    greetings_responses: list = ""
    clarification_responses: list = ""
    excel_column_level: str = ""
    df_column_names: str = ""
    sample_prompt_with_instructions: str = ""

    def __init__(self) -> None:
        self.context_template = AppConfig().get_value(AppConfigKeys.APP_SETTINGS, AppConfigKeys.AppSettings.EXCEL_CONTEXT)

        self.template_msg_pre = (
            "You are working with a pandas dataframe in Python. Pandas syntax must be valid and compilable before calling tool. The name of the dataframe is `df`. "
            "You should use 'python_repl_ast' tool to solve the user query. "
            "Please ensure that the code is free from all compilation and syntax errors. "
            "The output should be in CSV format. "
            "Don't add any extra information. The response should only contain the data in CSV format which should be parsable and should not contain any additional data. "
        )
        self.template_msg_post = (
            "Remember to follow below instructions and adhere to the instructions when generating the response:\n "
            "First column of the CSV response should contain aphabetical values and all other columns starting from the 2nd column should strictly contain numerical values only. "
            "Ensure that the CSV response strictly contains at least 2 columns and 2 rows of data. "
            "If the 2nd column of the CSV response contains aphabetical values, unstack the 2nd column and return the data in proper CSV format. "
            "Include all rows and columns. Make sure nothing is omitted or missed. "
            "Don't use any dummy data in CSV response. "
            "Keep in mind that dataframe is already loaded in the system, so no need to create or load the dataframe again. "
            "Make sure to display result using '**to_csv(index=False)**' method so that it can parsed correctly and get check all the available data. "
            "Just return data that is asked by the user only, no need to add or include all the other columns that serves no purpose in the csv response"
            )
        
        self.graph_prompt = (
            "If the user asks to visualize or plot data without requesting specific data, the output should be \"graph requested\". "
            "This is because the user only wants a graph, not the data itself. "
            "Example for above scenario is If the user prompt is : 'Plot above data in Pie chart' then return the output as \"graph requested\" as we already have requested csv response and we will use same to plot the data. "
            "If the user asks to plot data and specifies what data they want, retrieve the data using 'python_repl_ast' tool from the dataframe and NEVER respond with \"graph requested\". "
            "Example for above scenario is If the user prompt is : 'Draw or plot or show a line chart for the count of employees' then use 'python_repl_ast' tool to get the count of employees from dataframe and return the result in CSV format. "
        )
        self.thinking_instr = (
            "Please find instructions for calculating and solving the user prompt below:  \n" 
            "Firstly, think about the user prompt carefully and solve it step by step by dividing the prompt into multiple parts. "
            "When invoking python_repl_ast tool, firstly please import the necessary libraries and modules required for the code. "
            "Make sure to import the pandas and datetime modules/library even if it is not needed in the code. "
            "Think about the user prompt carefully and solve it step by step by dividing the prompt into multiple parts. "
            "Please divide each complex instruction in multiple small parts and solve it step by step one after another. "
            "Invoking python_repl_ast multiple times is highly encouraged which will help you to get the correct and more accurate answer. "
            "First column of the CSV response should contain alphanumerical values and all other columns starting from the 2nd column should contain strictly numerical values only. "
            "Ensure that the CSV response strictly contains at least 2 columns and 2 rows of data. "
            "If the 2nd column of the CSV response contains alphanumerical values, unstack the 2nd column and return the data in proper CSV format. "
        )
        
    

        self.greetings_responses = [
            "Hello! How can I assist you with your data visualization needs today?",
            "Hi there! Ready to visualize some data?",
            "Good day! How may I help you with your data visualization?",
            "Greetings! How can I assist you in visualizing your data today?",
            "Hello! What data visualization can I help you with today?",
            "Hi! How may I assist you in creating visualizations?",
            "Hello! How can I support your data visualization needs?",
            "Hi! Ready to create some data visualizations?",
            "Good to see you! How may I assist you with your data?",
            "Hello! How can I help you visualize your data today?",
            "Hey! What data visualization can I do for you today?",
            "Hi! What do you need help visualizing?",
            "Hello! How can I aid you in visualizing your data?",
            "Greetings! What data visualization can I assist you with?",
            "Hi there! What data do you need assistance visualizing?",
            "Hello! How can I support your data visualization today?",
            "Hi! How can I help you create visualizations today?",
            "Hello! What can I do to assist you with your data?",
            "Hey! How can I be of service for your data visualization?",
            "Hi! What can I help you visualize today?"
        ]

        self.clarification_responses = [
            "I'm sorry, I didn't understand your query. Could you please clarify?",
            "Could you please provide more details or clarify your request?",
            "I'm not sure I understand. Can you elaborate on your query?",
            "Can you please clarify what you mean?",
            "I didn't quite get that. Could you explain further?",
            "Could you provide more information or clarify your question?",
            "I'm having trouble understanding your request. Can you clarify?",
            "Can you please provide more details or rephrase your query?",
            "I'm not sure what you're asking. Could you clarify?",
            "Could you please explain your query in more detail?",
            "I didn't understand your request. Can you provide more details?",
            "Can you clarify your question or provide more context?",
            "I'm not sure I follow. Could you explain your query further?",
            "Could you please provide more information or clarify your request?",
            "I didn't quite understand that. Can you elaborate?",
            "Can you please clarify what you're asking?",
            "I'm having trouble understanding your query. Can you explain further?",
            "Could you provide more details or rephrase your question?",
            "I'm not sure I understand. Can you clarify your request?",
            "Can you please provide more context or details about your query?"
        ]  

        self.excel_column_level = ""

        self.sample_prompt_with_instructions = AppConfig().get_value(AppConfigKeys.APP_SETTINGS, AppConfigKeys.AppSettings.SAMPLE_PROMPT_WITH_INSTRUCTIONS)

    def set_multi_level_excel_columns(self,multilevel_excel_columns: str) -> None:
        if multilevel_excel_columns == True:
            self.excel_column_level = (
                "This dataframe contains Multi Index column so accordingly first fetch the "
                "Multi Index columns by executng 'df.columns' before running any other code and then fetch the data from the dataframe. In this case "
                "when returning the response in csv format, give meaningful name to the columns and rows. "
                "Please use double quotes to enclose the columns if it contains comma in the column name. "
            )
        else:
            self.excel_column_level = ""

    def generate_detailed_prompt(self,formatted_chat_history,user_query,openai_message: bool = True) -> str:

        if openai_message:
            return self._generate_context_openai_message(formatted_chat_history,user_query)
        else:
            return self._generate_context(formatted_chat_history,user_query)

    def _generate_context_openai_message(self,formatted_chat_history,user_query):
        final_prompt_message = [
            SystemMessage(content=f"Please find context for the dataframe below:"  + NEXT_LINE + self.context_template),
            # SystemMessage(content=formatted_chat_history),
            SystemMessage(content=self.template_msg_pre),
            HumanMessage(content=user_query),
            SystemMessage(content=self.actual_template_format),
            SystemMessage(content=self.template_msg_post),
            SystemMessage(content=self.graph_prompt),
            SystemMessage(content=self.greetings_prompt),
            SystemMessage(content=self.random_prompt),
            SystemMessage(content=self.csv_seperator),
            SystemMessage(content=self.thinking_instr),
            SystemMessage(content=self.excel_column_level)
        ]

        return convert_to_openai_messages(final_prompt_message)

    def _generate_context(self,formatted_chat_history,user_query):
            return (
                self.template_msg_pre + NEXT_LINE +
                "Please find context for the dataframe below:" + NEXT_LINE + 
                self.context_template + NEXT_LINE +  
                formatted_chat_history + NEXT_LINE +
                "Please find the user query below:" + NEXT_LINE +
                user_query + NEXT_LINE +
                self.actual_template_format + NEXT_LINE + 
                self.template_msg_post + NEXT_LINE +
                self.graph_prompt + NEXT_LINE + 
                self.thinking_instr + NEXT_LINE +
                self.excel_column_level
            )
            # return (
            #     self.thinking_instr + NEXT_LINE +
            #     self.template_msg_pre + NEXT_LINE + 
            #     self.actual_template_format + NEXT_LINE + 
            #     self.template_msg_post + NEXT_LINE +
            #     self.excel_column_level +
            #     "Please find the user query below:" + NEXT_LINE +
            #     user_query + NEXT_LINE + 
            #     formatted_chat_history + NEXT_LINE +
            #     "Please find context for the dataframe below:" + NEXT_LINE + 
            #     self.context_template + NEXT_LINE + 
            #     self.graph_prompt + NEXT_LINE + 
            #     self.greetings_prompt + NEXT_LINE + 
            #     self.random_prompt + NEXT_LINE + 
            #     self.csv_seperator + NEXT_LINE 

            # )
    def get_prefix_str(self,formatted_chat_history):
        return (
            "Please find context for the dataframe below:" + NEXT_LINE +
            self.context_template + NEXT_LINE +
            formatted_chat_history + 
            "Kind in mind to refer the chat history whenever neccessary when responding to the user query. "

        )

    
    def get_suffix_str(self):
            return (
                self.actual_template_format + NEXT_LINE + 
                self.template_msg_post + NEXT_LINE +
                self.graph_prompt + NEXT_LINE + 
                self.greetings_prompt + NEXT_LINE + 
                self.random_prompt + NEXT_LINE + 
                self.csv_seperator + NEXT_LINE + 
                self.thinking_instr + NEXT_LINE +
                self.excel_column_level
            )   

    def set_column_names(self,df: pd.DataFrame) -> None:
        self.df_column_names =  df.columns
        self.context_template += f"\nThe columns present in the dataframe are :\n {list(self.df_column_names)}"
        self.context_template += f"Date in the response should be in MM-YYYY format. For example, 01-2025, and should not be sorted in ascending or descending order.\n"