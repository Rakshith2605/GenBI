import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import json

st.set_page_config(layout="wide")

# Sidebar for file upload
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=['csv', 'xlsx', 'json'],
        help="Upload CSV, Excel or JSON files"
    )

    if uploaded_file is not None:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1]
        
        try:
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
                
            st.success(f"Successfully loaded {uploaded_file.name}")
            
            # Dataset Insights
            st.header("Dataset Insights")
            st.write("**Basic Information:**")
            st.write(f"- Rows: {df.shape[0]}")
            st.write(f"- Columns: {df.shape[1]}")
            
            # Display column info
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()
    else:
        st.warning("Please upload a data file")
        st.stop()

# Main content
st.title("Data Visualization Dashboard with PandasAI")

# Create tabs for different views
tab1, tab2 = st.tabs(["Data Preview", "Analysis"])

with tab1:
    st.header("Data Preview")
    st.dataframe(df.head(10))
    
    # Display basic statistics
    if st.checkbox('Show Statistical Summary'):
        st.write("**Statistical Summary:**")
        st.write(df.describe())

with tab2:
    # Create image container ONCE outside all logic blocks
    image_container = st.empty()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("API_KEY")

    # Initialize LLM and SmartDataframe
    llm = OpenAI(api_token=api_key)
    sdf = SmartDataframe(df, config={"llm": llm, "save_charts": False, "chart_type": "plotly"})

    # User query input
    user_query = st.text_input("Enter your query (e.g., 'Plot a histogram of column X')")

    if user_query:
        with st.spinner('Generating response...'):
            response = sdf.chat(user_query)
            
            if isinstance(response, plt.Figure):
                with image_container:
                    st.pyplot(response)
                plt.close(response)
            elif isinstance(response, str):
                if "saved" in response.lower():
                    file_path = response.split("Response:")[-1].strip() if "Response:" in response else "temp_chart.png"
                    if file_path and file_path.endswith(('.png', '.jpg', '.jpeg')):
                        image_container.image(file_path)
                else:
                    image_container.image("exports/charts/temp_chart.png")
                    os.remove("exports/charts/temp_chart.png")
            elif isinstance(response, pd.DataFrame):
                st.write("DataFrame Response:", response)
            else:
                st.write("Unexpected response type. Response:", response)
