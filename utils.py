import base64
from io import BytesIO
import streamlit as st
import pandas as pd

import configparser
from psm.email_utility import * 

def find_scaling_factor(min_demand):
    sol = 1
    if min_demand < 1:
        float_demand = '{:f}'.format(min_demand)
        x = str(float_demand).split(".")[1]
        found = False
        place = 0
        while found is False:
            if x[place] == "0":
                place += 1
            else:
                place += 1
                found = True
        sol = int("1" + (place + 1) *"0")
    return sol

def return_key(my_dict, value):
    return list(my_dict.keys())[list(my_dict.values()).index(value)]
              

def icon(title, icon_name):
    st.markdown('<div><h2 style="display: inline">{}</h2>&nbsp;&nbsp;<h1 style="display: inline" class="material-icons">{}</h1></div>'.format(title, icon_name), unsafe_allow_html=True)

@st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
def get_email_agent():
    read_config = configparser.ConfigParser()
    read_config.read("credentials.ini")
    sender_pass = read_config.get("SENDER CREDENTIALS", "sender_pass")
    sender_acct = read_config.get("SENDER CREDENTIALS", "sender_acct")
    agent = EmailAgent(sender_acct, sender_pass)

    return agent

""" Takes in dictionary of dataframes and names"""
def to_excel(**dfs):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    for sheet_name, df in dfs.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link_xlsx(name, filename, **dfs):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(**dfs)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.xlsx">Download {name} file</a>' # decode b'abc' => abc

# Pass in df and get downloadable link
def get_table_download_link_csv(df, text):
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="solution.csv" target="_blank">{text}</a>'
    return href

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate a link allowing the data in a given panda dataframe to be downloaded.
    
    Args:
        bin_file (str): a string denoting a binary file 
        file_label (str, optional): a string denoting a file label (default is "file")        
        
    Returns:
        href (str): a link to download a file
    """    

    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href