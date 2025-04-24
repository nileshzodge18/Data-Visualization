# DataVisualization Application

## Overview
The **DataVisualization** application is a powerful tool designed to create interactive and visually appealing data visualizations. It leverages the capabilities of Python and Streamlit to provide an intuitive interface for exploring and analyzing data.

## Features
- Interactive data visualization.
- Support for multiple chart types (e.g., bar charts, line charts, scatter plots).
- Easy-to-use interface for uploading and processing datasets.
- Real-time updates and interactivity.

## File Structure
The project is organized as follows:

```
DataVisualization/
├── .config/                  # Configuration files for the application
│   ├── app_config.toml       # Application configuration settings
├── files/                    # Directory for storing additional files
│   ├── Employee_Resource_Sheet.xlsx  # Example resource file
├── src/                      # Source code for the application
│   ├── main.py               # Entry point of the application
│   ├── components/           # Modular components for the application
│   │   ├── __init__.py       # Initialization for components module
│   │   ├── create_dataframe_agent.py  # Dataframe agent creation logic
│   │   ├── data_visualization.py      # Visualization generation logic
│   │   ├── llm_utils.py      # Utilities for language model integration
│   │   ├── user_input_handler.py  # User input handling logic
│   ├── utils/                # Utility functions and helpers
│   │   ├── __init__.py       # Initialization for utils module
│   │   ├── global_config_setup.py  # Global configuration setup
│   │   ├── initialize_variables.py  # Variable initialization logic
├── README.md                 # Project documentation
├── requirement.txt           # Python dependencies
```

## Getting Started

### Prerequisites
To run the **DataVisualization** application, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)

### Installation
1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   cd DataVisualization
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirement.txt
   ```
   ```

### Running the Application
The **DataVisualization** application is built using Streamlit. To start the application, follow these steps:

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. Run the application using Streamlit:
   ```bash
   streamlit run main.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

### Usage
1. Upload your dataset using the file upload interface.
2. Select the type of visualization you want to create (e.g., bar chart, line chart, scatter plot).
3. Customize the visualization parameters as needed.
4. Interact with the visualization in real-time.

### Example Dataset
An example dataset (`Employee_Resource_Sheet.xlsx`) is provided in the `files/` directory. You can use this dataset to test the application's features.

### Configuration
The application uses a configuration file (`app_config.toml`) located in the `.config/` directory. You can modify this file to customize application settings.

### File Structure Details
- **`.config/`**: Contains configuration files for the application.
- **`files/`**: Stores additional files, including example datasets.
- **`src/`**: Contains the source code for the application.
  - **`main.py`**: The entry point of the application.
  - **`components/`**: Modular components for handling specific functionalities like data visualization and user input.
  - **`utils/`**: Utility functions and helpers for tasks like configuration setup and variable initialization.

