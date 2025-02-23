import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import json

# Title of the app
st.title("Data Visualization Dashboard with PandasAI")

# Sidebar for file upload
st.sidebar.header("Upload Your Dataset")
file_type = st.sidebar.selectbox("Select file type", ["CSV", "Excel", "JSON"])
uploaded_file = st.sidebar.file_uploader(f"Upload your {file_type} file", type=[file_type.lower()])

# Initialize variables
df = None
sdf = None

# Load data based on file type
if uploaded_file is not None:
    try:
        if file_type == "CSV":
            df = pd.read_csv(uploaded_file)
        elif file_type == "Excel":
            df = pd.read_excel(uploaded_file)
        elif file_type == "JSON":
            df = pd.read_json(uploaded_file)
        
        st.sidebar.success(f"{file_type} file uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Display dataset insights and preview if data is loaded
if df is not None:
    st.sidebar.subheader("Dataset Insights")
    st.sidebar.write("**Columns and Data Types:**")
    st.sidebar.write(df.dtypes)
    
    st.sidebar.subheader("Data Preview")
    st.sidebar.write(df.head())  # Show first 5 rows as a preview

    # Load API key and initialize PandasAI
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        st.error("API_KEY not found in .env file. Please set it up.")
    else:
        llm = OpenAI(api_token=api_key)
        sdf = SmartDataframe(df, config={"llm": llm, "save_charts": False, "chart_type": "plotly"})

# Create image container ONCE outside all logic blocks
image_container = st.empty()

# User query input
user_query = st.text_input("Enter your query (e.g., 'Plot a histogram of column X')")

if user_query and sdf is not None:
    with st.spinner('Generating response...'):
        response = sdf.chat(user_query)
        
        if isinstance(response, plt.Figure):
            # Update the container with the new figure
            with image_container:
                st.pyplot(response)
            plt.close(response)
        elif isinstance(response, str):
            if "saved" in response.lower():
                file_path = response.split("Response:")[-1].strip() if "Response:" in response else "temp_chart.png"
                if file_path and file_path.endswith(('.png', '.jpg', '.jpeg')):
                    # Update container with new image
                    image_container.image(file_path)
            else:
                # Update container with the temp chart
                image_container.image("exports/charts/temp_chart.png", caption="Generated Chart")
        elif isinstance(response, pd.DataFrame):
            st.write("DataFrame Response:", response)
        else:
            st.write("Unexpected response type. Response:", response)
elif user_query and sdf is None:
    st.warning("Please upload a dataset before querying.")

# Option to show raw data
if df is not None and st.checkbox('Show raw data'):
    st.subheader("Raw Data")
    st.write(df)
elif st.checkbox('Show raw data') and df is None:
    st.warning("No data available to display. Please upload a file.")
