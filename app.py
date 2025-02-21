# app.py
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI  # Or another LLM
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
st.title("Data Visualization Dashboard with PandasAI")

# Load data
df = pd.read_csv("Data/data.csv")



load_dotenv()  # Load environment variables
api_key = os.getenv("API_KEY")


# Initialize LLM and SmartDataframe
llm = OpenAI(api_token=api_key)  # Replace with your actual API key
sdf = SmartDataframe(df, config={"llm": llm, "save_charts": False, "chart_type": "plotly"})
# User query input
user_query = st.text_input("Enter your query (e.g., 'Plot a histogram of column X')")

if user_query:
    with st.spinner('Generating response...'):
        
        response = sdf.chat(user_query)
        
        # Handle different types of responses from PandasAI
        if isinstance(response, plt.Figure):
            # If response is a Matplotlib figure
            st.pyplot(response)
            plt.close(response)  # Clean up to prevent memory leaks
        elif isinstance(response, str):  # If response is text (e.g., description or error)
            if "saved" in response.lower():  # If it mentions a saved file
                # Extract the file path or assume a default location
                file_path = response.split("Response:")[-1].strip() if "Response:" in response else "temp_chart.png"
                if file_path and file_path.endswith(('.png', '.jpg', '.jpeg')):
                    st.image(file_path)  # Display the saved image
                else:
                    st.write("Response 1 :", response)
            else:
                st.image("exports/charts/temp_chart.png", caption="Your caption here")

                #st.write("Response 2 :", response)
        elif isinstance(response, pd.DataFrame):  # If response is a DataFrame
            st.write("DataFrame Response:", response)
        else:
            st.write("Unexpected response type. Response:", response)

if st.checkbox('Show raw data'):
    st.write(df)