import streamlit as st
import pandas as pd 
import numpy as np
import os
import gzip
import shutil
import yaml
import yaml_to_json as y2j
import video_updater as vu
from defaults import get_teams, get_gaze_calibrations 


def import_usr_yml (file_path):
    files = os.listdir(file_path)
    #check if the files are in the directory
    if len(files) == 0:
          st.warning("No files found in the directory.")
          return None
    #check if the files are in the directory
    # check directories in the path
    directories = [d for d in files if os.path.isdir(os.path.join(file_path, d))]
    files = [f for f in files if os.path.isfile(os.path.join(file_path, f))]
    for file in files:
        if file.endswith(".gz") :
            #unzip the file
            with gzip.open(os.path.join(file_path, file), 'rb') as f_in:
                with open(os.path.join(file_path, file[:-3]), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
           
def import_project (file_path):
    files = os.listdir(file_path)
    return_data = {}
    #check if the files are in the directory
    if len(files) == 0:
          st.warning("No files found in the directory.")
          return None
    for file in files:
        if file.endswith(".prj") :
            if "BROKEN" not in file:
                data = y2j.convert_yaml_to_json(os.path.join (file_path,file))
               
                return_data[file] = data
                
    return return_data

def open_user_data (file_path):
    return_data = pd.DataFrame()
    #check the files and directoryes in the path
    files = os.listdir(file_path)
    # only gz files
    gzfiles = [f for f in files if f.endswith(".gz")]

    #check if the files are in the directory
    if len(gzfiles) == 0:
          st.warning("No files found in the directory.")
          return None
    #uncompress gz files
    for file in gzfiles:
        #st.write ("Uncompressing file: ", file)
        with gzip.open(os.path.join(file_path, file), 'rb', ) as f_in:
            content = f_in.read()
            #convert the content to string
            content = content.decode("utf-8")

            #st.text (content)
            data = y2j.custom_parser(content)
            #st.write (data.keys())  
            data_df = pd.DataFrame(data["Data"]).T
            data_df["file"] = file
            # user = firs numeric chars form the file name
            data_df["USER"] = data_df["file"].str.extract(r'(\d+)')[0] 
            data_df["MediaId"] =data_df["USER"].astype(int)

        # coincatenate the dataframes
        if return_data.empty:
            return_data = data_df
        else:
            #st.write ("Concatenating dataframes")
            #st.write (data_df.head())
            #st.write (return_data.head())
            return_data = pd.concat ([return_data, data_df], ignore_index=True)    
          
    return_data = return_data.reset_index(drop=True)
    return return_data


def import_measrements(file_path):
    """
    Import measurements from directory.


    """
    st.write ("Importing measurements from directory: ", file_path)
    #check the files and directoryes in the path
    files = os.listdir(file_path)
    #check if the files are in the directory
    if len(files) == 0:
        st.warning("No files found in the directory.")
        return None
    #check if the files are in the directory
    # check directories in the path
    directories = [d for d in files if os.path.isdir(os.path.join(file_path, d))]
    for dir in directories: 
        if dir == "result":
            pass
        elif dir == "src":

            pass
        elif dir == "user": 
            # itt vannak a videok, és a  user data filek. Ezt is átalakítjuk táblázattá.
            user_dir = os.path.join(file_path, dir) 
            user_data = open_user_data(user_dir)
            st.write ("User data file: ", user_dir)
            pass  

    return user_data
def AOI_pre_process( AOI_data):
    """
    Preprocess AOI data.
    """
    ret_data = pd.DataFrame()
    IDs= []
    names= []
    MediaIDs = []
    DataIDs = []
    starts = []
    types = []
    draw_types = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    #st.write ( AOI_data)
    
    for key, value in AOI_data.items():
        #st.write (key, value)
        for rect_key2, rect_value2 in value.items():
            if rect_key2 == "AOIRect":
                i=0
                #st.write ("AOIRect: ", key, value)
                for  rect_value in rect_value2:
                    i+=1
                    
                    IDs.append(str(value["Id"])+"_"+str(i))
                    names.append(value["Name"])
                    MediaIDs.append(value["MediaID"])
                    DataIDs.append(value["DataID"])
                    starts.append(rect_value["Start"] )
                    types.append(rect_value["Type"])
                    draw_types.append(value["DrawType"])
                    
                    x1.append(rect_value["Rect"][0])
                    y1.append(rect_value["Rect"][1])
                    x2.append(rect_value["Rect"][2] + rect_value["Rect"][0]) 
                    y2.append(rect_value["Rect"][3] + rect_value["Rect"][1])

    #st.write ( MediaIDs)    
    ret_data["ID"] = IDs
    ret_data["Name"] = names
    ret_data["MediaId"] = MediaIDs
    ret_data["DataId"] = DataIDs
    ret_data["Start"] = starts
    ret_data["Type"] = types
    ret_data["X1"] = x1
    ret_data["Y1"] = y1
    ret_data["X2"] = x2
    ret_data["Y2"] = y2
    ret_data["DrawType"] = draw_types
    ret_data["End"] = None
    
    
    # ends a ret_data ból minden, ami type=2 
    ends= ret_data[ret_data["Type"] == 2].copy()
    ret_data = ret_data[ret_data["Type"] != 2]
    
    for index, row in ret_data.iterrows():
        if row["Type"] == 1:

            end = ends[ends["Name"] == row["Name"]]
            if not end.empty:
                ret_data.at[index, "End"] = end.iloc[0]["Start"]
                #st.write ("End: ", end.iloc[0]["Start"])
            else:
                ret_data.at[index, "End"] = None
                #st.write ("End: None")                     

    ret_data = ret_data[["ID", "Name","Start", "End", "MediaId", "DataId", "Type", "X1", "Y1", "X2", "Y2", "DrawType"]]
    return ret_data

def calculate_eye_dynamics (Measurements):

    time_diff = Measurements['TIME'].diff().to_numpy()
    Measurements['eye_speed'] = ((Measurements['FPOGX'].diff().to_numpy()**2 + Measurements['FPOGY'].diff().to_numpy()**2)**0.5) * 1000.0 / time_diff
    
    
    return Measurements



def combine_AOI_task (AOI_data, Measurements):
    """
    SZTE specifikus cucc HAVAI és AUSZTRALIAI adatokhoz.
    Combine AOI data with measurements data.
    """
    Measurements["TASK"] = np.nan
    for index, row in AOI_data.iterrows():
        if row["Name"].startswith("H") or row["Name"].startswith("A"):
              Measurements["TASK"].loc[(Measurements["TIME"] >= row["Start"]) & (Measurements["TIME"] < row["End"]) & (Measurements["MediaId"] == row["DataId"])  ] = row["Name"]
        else:
            Measurements["AOI_"+str(row["DataId"])+"_"+row["Name"]] = 0
            Measurements["AOI_"+str(row["DataId"])+"_"+row["Name"]].loc[(Measurements["TIME"] >= row["Start"]) & (Measurements["TIME"] < row["End"]) & (Measurements["MediaId"] == row["DataId"])  ] = 1 
            Measurements["AOI_VISITED_"+str(row["DataId"])+"_"+row["Name"]] = 0
            Measurements["AOI_VISITED_"+str(row["DataId"])+"_"+row["Name"]].loc[
                (Measurements["AOI_"+str(row["DataId"])+"_"+row["Name"]] == 1) & 
                #(Measurements["MediaId"] == row["DataId"]) &
                (Measurements["BPOGX"] >= +row["X1"]) &
                (Measurements["BPOGX"] < +row["X2"]) &
                (Measurements["BPOGY"] >= +row["Y1"]) &
                (Measurements["BPOGY"] < +row["Y2"]) ] = 1
    # spaces is a filtered table where the maesurement["KB"] is "SPACE"
    mediaids = Measurements["MediaId"].unique()
    Measurements["ScreenId"] = np.nan
    for mediaid in mediaids:
        # & (Measurements["KBS"]==2)
        spaces = Measurements[["TIME", "KBS"]][(Measurements["KB"] == "SPACE") & (Measurements["MediaId"] == mediaid)  ].copy()
        if len (spaces[spaces["KBS"] == 2]) == len (spaces[spaces["KBS"] == 1]): 
            i = 0
            for index, space in spaces.iterrows():
                if space["KBS"] == 1:
                    Measurements["ScreenId"].loc[(Measurements["TIME"] >= space["TIME"]) & (Measurements["MediaId"] == mediaid) ] = np.nan
                elif space["KBS"] == 2:
                    i+=1
                    Measurements["ScreenId"].loc[(Measurements["TIME"] >= space["TIME"]) & (Measurements["MediaId"] == mediaid) ] = i
        
    #st.write (Measurements)
    
    return Measurements

def main():
    st.title("Gaze Importer")
    st.write("Import gaze measurements from a directory.")
    col1, col2, col3, col4 = st.columns(4)  
    files = os.listdir("import")
    teams = [d for d in files if os.path.isdir(os.path.join("import", d))]
    with col1:
        selected_team= st.sidebar.selectbox("Select team", teams)
        if selected_team:
            samples =  os.listdir(os.path.join("import", selected_team))
            samples.sort()
            # remove files in list, which is starting with .
            samples = [s for s in samples if not s.startswith('.')]
            with col2:
                selected_sample = st.sidebar.selectbox("Select sample", samples)
            file_path = os.path.join("import", selected_team, selected_sample)
        
        

        if st.sidebar.button("import Project"):
            aoi = pd.DataFrame()
            project_datas = import_project(file_path)
            for key, value in project_datas.items():
                for k, v in value.items():
                    if k == "Media":
                        pass
                        #st.write ("Media", key, v)
                    elif k == "UserData":
                        pass
                        #st.write ("User Data",key, v)
                    elif k == "AOI":
                        st.write ("AOI", key, v)
                    else:
                        pass
                        #st.write (key, v)   
    with col2:
        if st.sidebar.button("Import Measurements Data"):        
            Measurements = import_measrements(file_path)
    Annotated_video = st.checkbox("Annotate video files")

    if st.sidebar.button("Create Combined dateset"):
            do_import_process (file_path, selected_team, Annotated_video)
    if st.sidebar.button("Create Combined dateset ALL MEASUREMENTS"):
            for selected_sample in samples:
                #selected_sample = st.sidebar.selectbox("Select sample", samples)
                file_path = os.path.join("import", selected_team, selected_sample)
                do_import_process (file_path, selected_team, Annotated_video)

            """  project_datas = import_project(file_path)
            Measurements = import_measrements(file_path)
            #st.write (Measurements.head())
            for key, value in project_datas.items():
                for k, v in value.items():
                    if k == "AOI":
                        # megjelölük az AOI-kat az idősorban
                        #st.write ("AOI", key, v)
                        AOI_data = AOI_pre_process(v)
            Measurements = combine_AOI_task(AOI_data, Measurements)
            Measurements = calculate_eye_dynamics(Measurements)
            #st.write (Measurements.head())
            # save Measurements to csv file
            Measurements.to_csv(os.path.join(file_path, "result", "Measurements.csv"), index=False)
            st.write ("Measurements saved to: ", os.path.join(file_path, "result", "Measurements.csv"))
            # save Measurements to parkquet file
            Measurements.to_parquet(os.path.join(file_path, "result", "Measurements.parquet"), index=False)
            st.write ("Measurements saved to: ", os.path.join(file_path, "result", "Measurements.parquet"))

            # save AOI_data to csv file
            AOI_data.to_csv(os.path.join(file_path, "result", "AOI_data.csv"), index=False)
            st.write ("AOI_data saved to: ", os.path.join(file_path, "result", "AOI_data.csv"))
            # save AOI_data to parquet file
            AOI_data.to_parquet(os.path.join(file_path, "result", "AOI_data.parquet"), index=False)
            st.write ("AOI_data saved to: ", os.path.join(file_path, "result", "AOI_data.parquet"))
            st.write ("AOI_data: ", AOI_data)
            st.write ("Measurements: ", Measurements)
            
            
            if Annotated_video :
                # Annotate video files
                video_path = os.path.join(file_path, "user")
                st.write (video_path)

                for file in os.listdir(video_path):
                    if file.endswith("scrn.avi"):
                        video_file = os.path.join(video_path, file)
                        video_number = int (file.split("-")[0])
                        #st.write ("Video file: ", video_file, video_number)
                        AOI_df = AOI_data[AOI_data["DataId"] == video_number]
                        Measurements_df = Measurements[Measurements["MediaId"] == video_number]
                        #st.write ("AOI_df: ", AOI_df.head())
                        xcal, ycal = get_gaze_calibrations(selected_team)
                        vu.annotate_video_with_boxes(video_file, AOI_df, os.path.join(file_path, "result", file), Measurements_df, xcal=xcal, ycal=ycal)
            """
             

def do_import_process (file_path, selected_team, Annotated_video ):
            project_datas = import_project(file_path)
            Measurements = import_measrements(file_path)
            #st.write (Measurements.head())
            for key, value in project_datas.items():
                for k, v in value.items():
                    if k == "AOI":
                        # megjelölük az AOI-kat az idősorban
                        #st.write ("AOI", key, v)
                        AOI_data = AOI_pre_process(v)
            Measurements = combine_AOI_task(AOI_data, Measurements)
            Measurements = calculate_eye_dynamics(Measurements)
            #st.write (Measurements.head())
            # save Measurements to csv file
            Measurements.to_csv(os.path.join(file_path, "result", "Measurements.csv"), index=False)
            st.write ("Measurements saved to: ", os.path.join(file_path, "result", "Measurements.csv"))
            # save Measurements to parkquet file
            Measurements.to_parquet(os.path.join(file_path, "result", "Measurements.parquet"), index=False)
            st.write ("Measurements saved to: ", os.path.join(file_path, "result", "Measurements.parquet"))

            # save AOI_data to csv file
            AOI_data.to_csv(os.path.join(file_path, "result", "AOI_data.csv"), index=False)
            st.write ("AOI_data saved to: ", os.path.join(file_path, "result", "AOI_data.csv"))
            # save AOI_data to parquet file
            AOI_data.to_parquet(os.path.join(file_path, "result", "AOI_data.parquet"), index=False)
            st.write ("AOI_data saved to: ", os.path.join(file_path, "result", "AOI_data.parquet"))
            st.write ("AOI_data: ", AOI_data)
            st.write ("Measurements: ", Measurements)
            
            
            if Annotated_video :
                # Annotate video files
                video_path = os.path.join(file_path, "user")
                st.write (video_path)

                for file in os.listdir(video_path):
                    if file.endswith("scrn.avi"):
                        st.write ("Annotating video file: ", file)
                        video_file = os.path.join(video_path, file)
                        video_number = int (file.split("-")[0])
                        #st.write ("Video file: ", video_file, video_number)
                        AOI_df = AOI_data[AOI_data["DataId"] == video_number]
                        Measurements_df = Measurements[Measurements["MediaId"] == video_number]
                        #st.write ("AOI_df: ", AOI_df.head())
                        xcal, ycal = get_gaze_calibrations(selected_team)
                        vu.annotate_video_with_boxes(video_file, AOI_df, os.path.join(file_path, "result", file), Measurements_df, xcal=xcal, ycal=ycal)
                        st.write ("Annotated video file saved: ", os.path.join(file_path, "result", file))
             



if __name__ == "__main__":
    main()