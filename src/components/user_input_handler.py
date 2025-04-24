import streamlit as st
from .llm_utils import fetch_llm_response, visualize_response
import re
from utils.global_config_setup import glob_vars

def handle_user_input(prompt,json_response) -> None:
    """
    Handles user input and generates a title and graph type for data visualization.
    This function takes a user prompt and a JSON response, appends the prompt to a global message list,
    generates a title for the data, suggests a type of graph for plotting the data, and then visualizes
    the response using the generated title and graph type.
    Args:
        prompt (str): The user input prompt.
        json_response (str): The JSON response from the chatbot.
    Returns:
        None
    """

    response = json_response
    
    title_response = ""
    input_data = "User : " + prompt + "\n" + "Chatbot : " + response
    title_prompt = " Please provide the title for the data in less than 10 words. Format of title should be <Title for the data>. Don't add insights or any additonal information. Just return title of the data."
    title_system_content = "You are a helpful assistant that return a brief title or heading for {input_data} with less than 10 words in {output} format. Only respond in {output} format and nothing elseat and nothing else."
    title_response = fetch_llm_response(title_prompt, title_prompt,title_system_content,input_data)
   
    title_response = title_response.split('\n')[0]
    title_match = re.search(r'<(.*?)>', title_response)
    if title_match:
        title = title_match.group(1)
        title_response = title

    graph_type_response = ""
    prompt = prompt.lower()

    title_system_content = "You are a helpful assistant that suggest a type of graph for plotting {input_data}. Choose only from among the provided options : Bar Graph, Line Graph, Scatter Plot & Pie Chart. Only respond in {output} format and nothing else."
    graph_type_prompt = " What should be best graph or plotting to represent above Data? No preamble. Don't add additional info. Just give the type of graph. Choose only from among the provided options : Bar Graph, Line Graph,Scatter Plot & Pie Chart"
    graph_type_response = fetch_llm_response(graph_type_prompt, graph_type_prompt,title_system_content,input_data)


    graph_type_response = graph_type_response.lower()

    if any(graph_type in prompt for graph_type in ["bar", "line", "scatter", "pie"]):
        graph_type_response = prompt
    elif not any(graph_type in graph_type_response for graph_type in ["bar", "line", "scatter", "pie"]):
        graph_type_response = prompt
        if not any(graph_type in prompt for graph_type in ["bar", "line", "scatter", "pie"]):
            graph_type_response = "bar graph"

    if prompt and response:
        visualize_response(prompt, response,title_response,graph_type_response,input_data)
    return