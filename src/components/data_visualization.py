import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.global_config_setup import glob_vars
from matplotlib.figure import Figure
from io import BytesIO
from PIL import Image
from langchain_core.messages import AIMessage
import base64


def plot_graph(df, keys, title_response, multi_bar, graph_type_response) -> None:
    """
    Plots a graph based on the specified graph type.

    Parameters:
    df (pandas.DataFrame): The data frame containing the data to be plotted.
    keys (list): The list of keys/columns to be used for plotting.
    title_response (str): The title of the graph.
    multi_bar (bool): A flag indicating whether to plot multiple bars in a bar chart.
    graph_type_response (str): The type of graph to plot. Valid options are "bar", "line", "scatter", and "pie".

    Returns:
    None
    """
    graph_plotted = False

    if "bar" in graph_type_response:
        try:
            plot_bar_chart(df, keys, title_response, multi_bar)
        except:
            st.write("Error in generating Bar Chart. Please check the data.")
            return
            
        graph_plotted = True

    if "line" in graph_type_response:
        try:
            plot_line_chart(df, keys, title_response)
        except:
            st.write("Error in generating Line Chart. Please check the data.")
            return
        
        graph_plotted = True

    if "scatter" in graph_type_response:
        try:
            plot_scatter_chart(df, keys, title_response)
        except:
            st.write("Error in generating Scatter Chart. Please check the data.")
            return
        
        graph_plotted = True


    if "pie" in graph_type_response:
        try:
            plot_pie_chart(df, keys, title_response)
        except:
            st.write("Error in generating Pie Chart. Please check the data.")
            return
        
        graph_plotted = True

    if not graph_plotted:
        st.write("Invalid Graph Type. Please provide valid Graph Type.")



def plot_bar_chart(df, keys, title,multi_bar) -> None:
    """
    Plots a bar chart using the given DataFrame and keys.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to plot.
    keys (list): A list of keys/columns to plot from the DataFrame.
    title (str): The title of the bar chart.
    multi_bar (bool): A flag indicating whether to plot multiple bars.

    Returns:
    None
    """
    implement_bar_chart(df, keys, title, multi_bar)
    df = transpose_dataframe(df)
    keys = list(df.keys())
    multi_bar = False
    if len(keys) > 2:
        multi_bar = True
    # st.write(df)

    # implement_bar_chart(df, keys, title, multi_bar)

def plot_line_chart(df, keys, title) -> None:
    """
    Plots a line chart using the provided DataFrame and keys.

    This function first implements a line chart with the given DataFrame and keys.
    It then transposes the DataFrame, updates the keys, and writes the transposed
    DataFrame to the streamlit interface. Finally, it implements another line chart
    with the transposed DataFrame and updated keys.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be plotted.
        keys (list): A list of keys/columns to be used for plotting the line chart.
        title (str): The title of the line chart.

    Returns:
        None
    """
    implement_line_chart(df, keys, title)
    df = transpose_dataframe(df)
    keys = list(df.keys())
    # st.write(df)
    # implement_line_chart(df, keys, title)

def plot_scatter_chart(df, keys, title) -> None: 
    """
    Plots a scatter chart using the provided DataFrame and keys.

    This function first implements a scatter chart with the given DataFrame and keys.
    Then, it transposes the DataFrame, updates the keys, and writes the transposed DataFrame.
    Finally, it implements another scatter chart with the transposed DataFrame and updated keys.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be plotted.
        keys (list): A list of keys/columns to be used for plotting.
        title (str): The title of the scatter chart.

    Returns:
        None
    """
    implement_scatter_chart(df, keys, title)
    df = transpose_dataframe(df)
    keys = list(df.keys())
    # st.write(df)
    # implement_scatter_chart(df, keys, title)

def plot_pie_chart(df, keys, title) -> None:
    """
    Generates and displays a pie chart based on the provided DataFrame and keys.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    keys (list): A list of column names to be used for the pie chart.
    title (str): The title of the pie chart.
    Returns:
    None
    """
    st.markdown(f"<h3 style='text-align: center;'>Generating Pie Chart</h3>", unsafe_allow_html=True)
    num_categories = len(df[keys[0]])
    font_size = min(8, 10 + (20 - num_categories) * 0.5)  # Adjust font size based on number of categories
    if font_size < 5:
        font_size = 6

    plt.figure(figsize=(10,10))
    implement_pie_chart(df, keys, title,font_size)
    orig_keys = keys


    if len(keys) < 3:
        return


    df = transpose_dataframe(df)

    num_categories = len(df[keys[0]])
    font_size = min(8, 10 + (20 - num_categories) * 0.5)  # Adjust font size based on number of categories
    
    if font_size < 5:
        font_size = 6

    column_name_content = "You are a helpful assistant that checks the {input_data} and suggests a column name for the data. Column name should be 1-3 words long. Don't bold the value or italize the value. Only return the column name and nothing else. Please refer to this question data {question_data} for column name suggestion."
    input_data = df.iloc[:, 0].tolist()

    from .llm_utils import fetch_llm_response

    column_name_response = fetch_llm_response("", st.session_state.user_query,column_name_content,input_data)


    df.columns = [column_name_response] + list(df.columns[1:])
    keys = list(df.keys())


    st.markdown("<h3 style='text-align: center;'>Generating a Pie Chart for Transposed Dataframe </h3>", unsafe_allow_html=True)


    if len(orig_keys) > 3:
        implement_pie_chart(df, keys, title, font_size)






def implement_bar_chart(df, keys, title, multi_bar) -> None:
    """
    Generates and displays a bar chart or multi-bar chart using the provided DataFrame.
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be visualized.
    keys (list): A list of column names to be used for the x-axis and y-axis values. 
                 The first element is used for the x-axis, and the remaining elements are used for the y-axis.
    title (str): The title of the chart.
    multi_bar (bool): If True, generates a multi-bar chart; otherwise, generates a single bar chart.
    Returns:
    None
    """
    differences = 0
    max_value = 0
    min_value = np.iinfo(np.int32).max
    for col in df.columns[1:]:
        max_value = max(max_value,df[col].max())
        min_value = min(min_value,df[col].min())
        differences = max_value - min_value

    if multi_bar == True:
        st.markdown("<h3 style='text-align: center;'>Generating a Multi-Bar Chart</h3>", unsafe_allow_html=True)

        num_points = len(df[keys[0]])
        figsize = (max(10, num_points * 0.6), max(10, num_points * 0.6)*(9/16))  # Increased pixel dimensions
        fig, ax = plt.subplots(figsize=figsize)

        bar_width = 0.20  # Adjusted bar width for better spacing
        group_gap = 0.5  # Gap between groups
        indices = np.arange(len(df[keys[0]])) * (len(keys) - 1) * bar_width + np.arange(len(df[keys[0]])) * group_gap  # Adjusted indices for better spacing

        # Generate unique colors for each bar group using the tab20 colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(keys) - 1))

        for i, (key, color) in enumerate(zip(keys[1:], colors)):
            bars = ax.bar(indices + i * bar_width, df[key], bar_width, label=key, color=color)
            for bar, value in zip(bars, df[key]):
                if value.is_integer():  # Check if the float value is an integer
                    ax.bar_label(bars, labels=[f'{int(value)}' for value in df[key]], fontsize=6)  # Display as integer
                else:
                    ax.bar_label(bars, labels=[f'{value:.2f}' for value in df[key]], fontsize=5)  # Display as float with 2 decimals

        ax.set_xticks(indices + bar_width * (len(keys) - 2) / 2)
        ax.set_xticklabels(df[keys[0]])

        legend_fontsize = max(5, min(10, 100 // len(df.columns)))
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=legend_fontsize, bbox_transform=ax.transAxes)
        ax.set(title=title)
        ax.set_xlabel(keys[0].capitalize())
        ax.set_ylim(bottom=-1)  # Start y-axis from -1

        # Adjust the bottom of the bars to start from -10
        for i, key in enumerate(keys[1:]):
            for j in range(len(df[key])):
                plt.gca().patches[i * len(df[key]) + j].set_y(-10)
                plt.gca().patches[i * len(df[key]) + j].set_height(df[key][j] + 10)

        fig.autofmt_xdate()
        st.pyplot(fig)
        # st.session_state.messages.append({"role": "assistant", "content": fig})
        # image_url = fig_to_base64(fig)
        # ai_msg = AIMessage(content = "Placeholder For Image")
        # ai_msg.additional_kwargs["image_url"] = image_url
        # st.session_state.messages.append(ai_msg)
        # st.session_state.charts.append(fig)
        plt.clf()
        
    else:
        st.markdown("<h3 style='text-align: center;'>Generating a Bar Chart</h3>", unsafe_allow_html=True)

        num_points = len(df[keys[0]])
        figsize = (max(8, num_points * 0.5), max(8, num_points * 0.5)*(9/16))
        fig, ax = plt.subplots(figsize=figsize)
        barContainer = ax.bar(df[keys[0]], df[keys[1]])
        ax.set(ylabel=keys[1].capitalize(), title=title)
        ax.set_xlabel(keys[0].capitalize())
        
        if pd.api.types.is_integer_dtype(df[keys[1]]):
            ax.bar_label(barContainer, fmt='%d')
        else:
            ax.bar_label(barContainer, fmt='%.2f')
        
        fig.autofmt_xdate()

        st.pyplot(fig)
        plt.clf()

        st.session_state.messages.append({"role": "assistant", "content": fig})




def implement_line_chart(df, keys, title) -> None:
    """
    Generates a line chart using the provided DataFrame and keys.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    keys (list of str): A list of column names to be used for the x-axis and y-axis values.
                        The first element is used for the x-axis, and the rest are used for the y-axis.
    title (str): The title of the chart.
    Returns:
    None
    """
   
    st.markdown("<h3 style='text-align: center;'>Generating a Line Chart</h3>", unsafe_allow_html=True)
    num_points = len(df[keys[0]])
    figsize = (max(8, num_points * 0.5), max(8, num_points * 0.5)*(9/16))
    width,height = figsize[0],figsize[1]
    fig, ax = plt.subplots(figsize=figsize)
    # for colNum in range(1, len(df.columns)):
    #     ax.plot(df[keys[0]], df[keys[colNum]], label=keys[colNum])
        
    # ax.set_xlabel(keys[0].capitalize())
    # ax.set_title(title)
    # ax.legend()
    # fig.autofmt_xdate()
    
    # st.pyplot(fig)
    # plt.clf()
    # Calculate the difference between the highest and lowest numbers from 2nd column onwards
    differences = 0
    max_value = 0
    min_value = np.iinfo(np.int32).max
    for col in df.columns[1:]:
        max_value = max(max_value,df[col].max())
        min_value = min(min_value,df[col].min())
        differences = max_value - min_value


    for colNum in range(1, len(df.columns)):
        ax.plot(df[keys[0]], df[keys[colNum]], label=keys[colNum], marker='o',markersize=3)
        for x, y in zip(df[keys[0]], df[keys[colNum]]):
            if isinstance(y, int) or (isinstance(y, float) and y.is_integer()):
                ax.text(x, y, f'{int(y)}', fontsize=6, ha='center', va='bottom', position=(x, y + differences * 0.01))
            else:
                ax.text(x, y, f'{y:.2f}', fontsize=6, ha='center', va='bottom', position=(x, y + differences * 0.01))

    ax.set_xlabel(keys[0].capitalize())
    ax.set_title(title)

    legend_fontsize = max(5, min(10, 100 // len(df.columns)))
    # Generate unique colors for each line
    colors = plt.cm.tab20(np.linspace(0, 1, len(df.columns) - 1))
    for line, color in zip(ax.get_lines(), colors):
        line.set_color(color)

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=legend_fontsize, bbox_transform=ax.transAxes)
    # ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1))

    fig.autofmt_xdate()
    
    st.pyplot(fig)
    plt.clf()

    




def implement_scatter_chart(df, keys, title) -> None:
    """
    Generates and displays a scatter chart using the provided DataFrame and keys.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    keys (list of str): A list of column names to be used for the x and y axes.
                        The first element is used for the x-axis, and the rest are used for the y-axes.
    title (str): The title of the scatter chart.
    Returns:
    None
    """
    
    st.markdown("<h3 style='text-align: center;'>Generating a Scatter Chart</h3>", unsafe_allow_html=True)
    num_points = len(df[keys[0]])
    figsize = (max(8, num_points * 0.5), max(8, num_points * 0.5)*(9/16))

    differences = 0
    max_value = 0
    min_value = np.iinfo(np.int32).max
    for col in df.columns[1:]:
        max_value = max(max_value,df[col].max())
        min_value = min(min_value,df[col].min())
        differences = max_value - min_value

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, len(df.columns) - 1))  # Generate unique colors for each scatter plot
    for colNum, color in zip(range(1, len(df.columns)), colors):
        ax.scatter(df[keys[0]], df[keys[colNum]], label=keys[colNum], color=color)
        for x, y in zip(df[keys[0]], df[keys[colNum]]):
            if isinstance(y, int) or (isinstance(y, float) and y.is_integer()):
                ax.text(x, y, f'{int(y)}', fontsize=6, ha='center', va='bottom', position=(x, y + differences * 0.01))
            else:
                ax.text(x, y, f'{y:.2f}', fontsize=6, ha='center', va='bottom', position=(x, y + differences * 0.01))
        
    ax.set_xlabel(keys[0].capitalize())
    ax.set_title(title)
    legend_fontsize = max(5, min(10, 100 // len(df.columns)))
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=legend_fontsize , bbox_transform=ax.transAxes)
    fig.autofmt_xdate()
    
    st.pyplot(fig)
    plt.clf()




def implement_pie_chart(df, keys, title,font_size) -> None:
    """
    Generates and displays pie charts for each column in the DataFrame, excluding the first column.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to visualize.
    keys (list): A list of column names where the first element is the category column and the rest are value columns.
    title (str): The title for the pie charts.
    font_size (int): The font size for the labels and percentages in the pie chart.
    Returns:
    None
    """

    
    for colNum in range(1, len(df.columns)):        
        # Filter out categories with value 0
        filtered_df = df[df[keys[colNum]] != 0]
        filtered_categories = list(filtered_df[keys[0]])
        filtered_values = list(filtered_df[keys[colNum]])
        num_filtered_categories = len(filtered_categories)

        if all(value == 0 for value in df[keys[colNum]]):
            st.markdown(f"<h3 style='text-align: center; font-size: 16px; color: red;'><em><u>Pie Chart for \"{keys[colNum]}\" has been omitted due to the absence of non-zero values</u></em></h3>", unsafe_allow_html=True)
            continue
        
        st.markdown(f"<h3 style='text-align: center; font-size: 18px; border: 1px solid darkblue; padding: 5px;'><em>Generating Pie Chart for \"{keys[colNum]}\"</em></h3>", unsafe_allow_html=True)


        colors = plt.cm.tab20(np.linspace(0, 1, num_filtered_categories))  # Generate unique colors using viridis colormap
        wedges, texts, autotexts = plt.pie(
            filtered_values, 
            labels=filtered_categories, 
            labeldistance=1.1, 
            explode=[0.05]*num_filtered_categories,
            autopct=lambda p: f'{p:.1f}%\n({p*sum(filtered_values)/100:.0f})', 
            startangle=140,
            colors=colors  # Apply the unique colors
        )

        for text in texts + autotexts:
            text.set_fontsize(font_size)

        plt.legend(wedges, filtered_categories, title=keys[0], loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.axis('equal')
        st.pyplot(plt)
        plt.clf()

    
def transpose_dataframe(df) -> pd.DataFrame:
    """
    Transposes the given DataFrame such that rows become columns and columns become rows.
    
    The first column of the original DataFrame is set as the index before transposing.
    After transposing, the index is reset and the columns are renamed to match the original DataFrame's first column.

    Args:
        df (pandas.DataFrame): The DataFrame to be transposed.

    Returns:
        pandas.DataFrame: The transposed DataFrame with the first column as the new header.
    """

    transposed_df = df.set_index(df.columns[0]).transpose().reset_index()
    transposed_df.columns = [df.columns[0]] + list(df[df.columns[0]])
    return transposed_df



def fig_to_base64(fig: Figure) -> str:
    """
    Convert a Figure object to a base64-encoded image, and return
    the resulting encoded image to be used in place of a URL.
    """

    with BytesIO() as buffer:
        fig.savefig(buffer, format="JPEG")
        buffer.seek(0)
        image = Image.open(buffer)

        return image_to_base64(image)
    

def image_to_base64(image: Image) -> str:
    """
    Convert an image object from PIL to a base64-encoded image,
    and return the resulting encoded image as a string to be used
    in place of a URL.
    """

    # Convert the image to RGB mode if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save the image to a BytesIO object
    buffered_image = BytesIO()
    image.save(buffered_image, format="JPEG")

    # Convert BytesIO to bytes and encode to base64
    img_str = base64.b64encode(buffered_image.getvalue())

    # Convert bytes to string
    base64_image = img_str.decode("utf-8")

    return f"data:image/jpeg;base64,{base64_image}"