import streamlit as st
import pandas as pd
import numpy as np
from defaults import get_teams
import os
import datetime

def open_experiment():
    if "experiment.csv" not in os.listdir():
        # Create a new DataFrame if the file does not exist
        experiment = pd.DataFrame(columns=["id", "name", "date", "time", "location", "team", "import file", "result"])
        # Save the empty DataFrame to a CSV file
        experiment.to_csv("experiment.csv", index=False)
    else:
        # Load the existing DataFrame from the CSV file
        experiment = pd.read_csv("experiment.csv")
        # Convert the date and time columns to datetime objects
        experiment["date"] = pd.to_datetime(experiment["date"], errors='coerce').dt.date
        experiment["time"] = pd.to_datetime(experiment["time"], errors='coerce').dt.time
        # Convert the result column to numeric, forcing errors to NaN
        experiment["id"] = pd.to_numeric(experiment["id"], errors='coerce')
        #location is a string, so no conversion is needed
        experiment["team"] = experiment["team"].astype(str)
        experiment["import file"] = experiment["import file"].astype(str)
        experiment["result"] = pd.to_numeric(experiment["result"], errors='coerce')
        # Fill NaN values with empty strings for string columns
        experiment["name"] = experiment["name"].fillna("")
        experiment["location"] = experiment["location"].fillna("")
        experiment["team"] = experiment["team"].fillna("")
        experiment["import file"] = experiment["import file"].fillna("")

        
        # Check if the DataFrame is empty and create a new one if it is
        if experiment.empty:
            experiment = pd.DataFrame(columns=["id", "name", "date", "time", "location", "team", "import file", "result"])
            # Save the empty DataFrame to a CSV file
            experiment.to_csv("experiment.csv", index=False)    

    return experiment

def save_experiment(experiment):
    # Save the experiment data
    #st.write (experiment)
    experiment.to_csv("experiment.csv", index=False)


def main():
    experiment = open_experiment()
    experiment = st.data_editor(
        experiment,
        column_config={
            "id": st.column_config.NumberColumn ("ID"),
            "name": st.column_config.TextColumn("Név"),
            "date": st.column_config.DateColumn("Dátum"),
            "time": st.column_config.TimeColumn("Idő"),
            "location": st.column_config.TextColumn("Helyszín"),
            "team": st.column_config.SelectboxColumn ("Csapat", options=get_teams()),
            "import file": st.column_config.LinkColumn("Import fájl"),
            "result": st.column_config.NumberColumn("Eredmény"),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",

    )
    # Save the experiment data when the user clicks the button
    if st.button("Mentés"):
        save_experiment(experiment)
        st.success("A kísérlet adatai sikeresen mentve!")

if __name__ == "__main__":
    import os
    os.environ["NUMBA_THREADING_LAYER"] = "tbb"
    
    st.set_page_config(page_title="Térképolvasás kísérlet 2024-2025", page_icon=":guardsman:", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f0f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )   
    main()
