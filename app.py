import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

st.title("Data Visualization Dashboard with PandasAI")

# Create image container ONCE outside all logic blocks
image_container = st.empty()

# Load data
df = pd.read_csv("Data/data.csv")
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
                image_container.image("exports/charts/temp_chart.png", caption="Your caption here")
        elif isinstance(response, pd.DataFrame):
            st.write("DataFrame Response:", response)
        else:
            st.write("Unexpected response type. Response:", response)

if st.checkbox('Show raw data'):
    st.write(df)
