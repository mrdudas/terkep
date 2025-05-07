import json
import re
import streamlit as st

counter = 0

def replacer(match):
    global counter
    replacement = f'\n"id_{counter}":'
    counter += 1
    return replacement

def fix_braces_and_commas(text):
    #remove all \n and \r characters
   
    # Add quotes around keys
    text = re.sub(r'(?<=[{,\s])(\w+):', r'"\1":', text)
    # Fix arrays (remove quotes inside arrays)
    text = re.sub(r'"\[', '[', text)
    text = re.sub(r'\]"', ']', text)
    # Replace single quotes with double quotes (if any)
    text = text.replace("'", '"')
    # Remove trailing commas before closing brackets
    text = re.sub(r',\s*([\]}])', r'\1', text)
    
    

    return text


def custom_parser(yaml_str):
    counter = 0
    

    yaml_str = re.sub(r'\[([^\]]*)\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', yaml_str)
    yaml_str = re.sub(r'\{([^\}]*)\}', lambda m: '{' + ' '.join(m.group(1).split()) + '}', yaml_str)

    yaml_str = re.sub(r'\n\s*---\s*', '', yaml_str) 
    yaml_str = re.sub(r'\n\s*-\s*', replacer, yaml_str)
    yaml_str = re.sub(r'}{', '}, {', yaml_str)
    # replace {Id:0, with ID0:{
   # yaml_str = re.sub(r'\{Id:(\d+),', r'\nId_\1:{Id:\1,', yaml_str)
    yaml_str = re.sub(r'},\n( +){', '}, {', yaml_str)

    yaml_str = yaml_str.split('\n', 1)[1]


    # sor elejéntől kettőspontig "" kulcsok

    yaml_str = re.sub(r'(\w+):', r'"\1":', yaml_str)
    
    # karakter values to ""
    yaml_str = re.sub(r'(\w+):\s*([a-zA-Z0-9_]+)', r'\1: "\2"', yaml_str)
    yaml_str = re.sub(r'(?<!")\b([A-Za-z_][A-Za-z0-9_ ]*)\b(?!")', r'"\1"', yaml_str)
    yaml_str = re.sub(r'([}\]0-9."]) *\n *(")', r'\1,\n\2', yaml_str)  
    #yaml_str = re.sub(r'"(Media|UserData|AOI|Web|DataRecordsReset|EventRecords|MobileRecords|Data)":', r'"\1":{', yaml_str)

    yaml_str = re.sub(r'"(Media|UserData|AOI|Web|DataRecordsReset|EventRecords|MobileRecords|Data)":', r'"\1":{', yaml_str)
    yaml_str = re.sub(r'"(UserData|AOI|Web|DataRecordsReset|EventRecords|MobileRecords|Data)":{', r'},\n"\1":{', yaml_str)
    #"DataRecords":{26357}
    yaml_str = re.sub(r'\n"DataRecords": (\d+),\n},', r'\n"DataRecords":\1,', yaml_str)
    #"MobileRecords":{
   

    yaml_str = re.sub(r'\n"MobileRecords":{\n', r'\n"MobileRecords":{},\n', yaml_str)

    

    #yaml_str = re.sub(r'},\n("UserData"):', r'}},\n\1:', yaml_str)
    #yaml_str = re.sub(r'},\n("AOI"):', r'}},\n\1:', yaml_str)
   
    yaml_str = re.sub(r'(\d)\.(\]|,|\})', r'\1.0\2', yaml_str)
   
   # Cseréljük az ilyen "HH":"MM" típusú dolgokat normálisan HH:MM formára
    yaml_str = re.sub(r'"(\d{2})":"(\d{2})"', r'\1:\2', yaml_str)
    yaml_str = "{\n"+yaml_str+"}}"
    
    yaml_str = re.sub(r'},}', r'}}', yaml_str)
    yaml_str = re.sub(r'},\n}', r'}\n}', yaml_str)
    yaml_str = re.sub(r'\[\],', r'', yaml_str)
    yaml_str = re.sub(r'\[\]', r'', yaml_str)
    
    
    

    # "Media": to "Media": {
    #st.text (yaml_str)


     
    
    # save to file 
    with open("yaml_str.txt", "w", encoding="utf-8") as f:
        f.write(yaml_str)

 
    
    
    data = json.loads(yaml_str)
    # 5. Tiszta JSON objektummá konvertálás (ellenőrzés)
    try:
        #st.write ("yaml_str")
        data = json.loads(yaml_str)
        #st.write (data.keys())
       
    except json.JSONDecodeError as e:
        # REPLACE LAST }} WITH }
        yaml_str = re.sub(r'}\s*}\s*$', '}', yaml_str) 
        try:
            data = json.loads(yaml_str)
        except json.JSONDecodeError as e:

           
            return None
    


    
    return data
    

def convert_yaml_to_json(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    parsed_data = custom_parser(content)
   
    return parsed_data

   

