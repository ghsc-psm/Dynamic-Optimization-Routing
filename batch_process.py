
import streamlit as st
import pandas as pd
import configparser
import numpy as np
import SessionState
import random
import string
import configparser
import folium
from streamlit_folium import folium_static
from map_utils import colors, draw_route_in_map


from DRO import  DROSpecDesc
from scenario import save_scenario
from app_upload import locate_warehouse
from os import listdir
from os.path import isfile, join
from app_optimize import run_optimization
from app_upload import app_upload_batch
from app_login import check_credentials



country_config = configparser.ConfigParser()
country_config.read("./country_cfg.toml")
country_name  = "COUNTRY" if country_config.get("country", "country_name").title() =="" else country_config.get("country", "country_name").title()
data_path = f'./data/{country_name}/'
data_path_batch = f'./data/{country_name}/batch/'

def batch_main():
    """Allow user to navigate to each page of the application for batch processing"""
    
    session_state = SessionState.get(ref_file = "", 
                                     warehouse = "",
                                     user_file = "", 
                                     scenario_file = "", 
                                     scenario_data = None,
                                     scenario_ver_no = 0,
                                     username = "",
                                     user_type = "",
                                     country = country_name,
                                     calc_master_file = "",
                                     calc_order_file = "",
                                     calc_solution = [],
                                     calc_buffer = 0,
                                     data_file='',
                                     calc_type_sel = [], 
                                     dro_order_details_file = "", 
                                     dro_facility_mapping_file = "",
                                     dro_order_evaluation_template_file = "", 
                                     dro_scenario_key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)), 
                                     dro_order_file ="", 
                                     ver = "v1.4.7.8 (2022/01/26)", 
                                     opt_file_dict = {},
                                     file_confirmation = False,
                                     opt_parameters = None
                                     )


    tool_title = "Batch Processing Distribution Routing Tool"
     
    st.sidebar.title(tool_title)
    
    if session_state.username != "": 
        logout = st.sidebar.button("Logout")
        if logout:
            session_state.ref_file = ""
            session_state.warehouse = ""
            session_state.user_file = ""
            session_state.scenario_file = ""
            session_state.scenario_data = None
            session_state.scenario_ver_no = 0
            session_state.username = ""
            session_state.user_type = ""
            session_state.country = ""
            session_state.calc_order_file = ""
            session_state.calc_solution = []
            session_state.calc_buffer = 0
            session_state.data_file=''
            session_state.calc_type_sel = []
            session_state.dro_order_file =""
            st.experimental_rerun()
          
    if session_state.username == "":
        check_credentials(session_state)
    else:
        st.sidebar.markdown(f'### Hello, {"Admin" if session_state.user_type == "admin" else session_state.username}')
        app_options = ["Batch Dispatch Optimization"]
        app_mode = st.sidebar.selectbox("", app_options)
 
        if app_mode == "Batch Dispatch Optimization":
            batch_file_upload(session_state)
            if session_state.file_confirmation:
                batch_file_solve(session_state)
            else:
                st.error("Please Upload and Confirm Files")
    return

def batch_file_solve(session_state):
    """Runs multiple optimizations for all specified files
    Args:
        session_state (object): a new SessionState object
    """
    st.title('Batch Processing Optimization')

    st.markdown('### The following files will be processed:')
    
    for drt in session_state.opt_file_dict.keys():
        st.markdown(drt)
        for dro in session_state.opt_file_dict[drt]:
            st.markdown(f"- {dro.name}")

    country_config = configparser.ConfigParser()
    country_config.read("./country_cfg.toml")
      

    st.markdown("### Optimization Constraints")
    col_left, col_tl, col_par, col_right = st.columns([0.25, 1, 1, 0.25])

    opt_parameters = {"Include Return Leg Cost":True, 
                    "Enforce Volume Capacity":True,
                    "Enforce Weight Capacity":True, 
                    "Enforce Distance Limit":False,
                    "Enforce Transit Time Limit":True,
                    'Enforce Delivery Time Limit':True,
                    'Allow Missed Deliveries':True, 
                    "Optimization Runtime Limit":60, 
                    "Adjust Transit Time by Speed":False}

    par_list = ["Optimization Runtime Limit", "Include Return Leg Cost", 
        "Enforce Volume Capacity", "Enforce Weight Capacity", 
        "Enforce Distance Limit","Enforce Transit Time Limit",
        'Enforce Delivery Time Limit','Allow Missed Deliveries', 
        "Adjust Transit Time by Speed"]

    for par in par_list:
        if country_config.get("dro_specs", par).title()=="True":
            
            default_par_val =  opt_parameters[par]
            if par == "Optimization Runtime Limit": 
                opt_parameters[par] = int(col_tl.text_input(par, value=default_par_val, help=DROSpecDesc[par]))
            else:    
                opt_parameters[par] = bool(col_tl.checkbox(par, value=default_par_val, help=DROSpecDesc[par], key = par))
    st.markdown('*Note: Parameters set in the DRT File will overwrite parameters specified here.')
    use_predefined_routes = False
    if country_config.get("baseline", "predefined_routes").title() == "True":
        use_predefined_routes = bool(col_par.checkbox("Use Predefined Routes", value=False, 
                            help="Perform Route Optimization Using Route Numbers in Scenario File", key = "Use Predefined Routes"))
    
    
    
    session_state.opt_parameters = opt_parameters
    failed = []
    if st.button("Run Batch Optimization"): 
        
        for drt in session_state.opt_file_dict.keys():
            ref_file = drt
            session_state.ref_file = join(data_path, ref_file)
            session_state.warehouse = locate_warehouse(session_state.ref_file)
            if session_state.warehouse == "": 
                st.error("The reference file has yet to be mapped to a dispatch origin. Please maintain REF_WAREHOUSE_MAPPING in code")
                st.stop()
            for dro in session_state.opt_file_dict[drt]:
                # Handle the upload component here:
                status = st.empty()
                status.markdown(f"Processing {drt} - {dro.name}")
                uploaded_file = dro
                session_state.scenario_file = f"{data_path_batch}/{drt[:-5]} - {uploaded_file.name[:-5]} Solved.xlsx"
                successful_upload = app_upload_batch(session_state, uploaded_file)
                successful_solve = False
                opt_parameters = session_state.opt_parameters
                scenario = session_state.scenario_data
                for par in par_list:
                    opt_parameters[par] = opt_parameters[par] if par not in scenario["Parameters"] else scenario["Parameters"][par]
                session_state.opt_parameters = opt_parameters
                if successful_upload:
                    st.markdown("### Successful Upload")
                    st.markdown("### Running Optimization")
                    successful_solve = run_optimization(session_state, session_state.opt_parameters, use_predefined_routes)
                if successful_solve:
                    save_scenario(f"{data_path_batch}/{drt[:-5]} - {uploaded_file.name[:-5]} Solved.xlsx",session_state.scenario_data)
                    scenario = session_state.scenario_data
                    m = folium.Map(location = [scenario['Facility_DF'].iloc[0].latitude, scenario['Facility_DF'].iloc[0].longitude], zoom_start = 10)
                    folium.Marker([scenario['Facility_DF'].iloc[0].latitude, scenario['Facility_DF'].iloc[0].longitude], tooltip = scenario['Facility_DF'].iloc[0].facility).add_to(m)
                    for _, route in scenario['SolSummary_DF'].iterrows(): 
                        route_no = int(route['route'].split(' ')[-1])
                        m = draw_route_in_map(m, scenario['SolDetail_DF'][scenario['SolDetail_DF']["route"] == route["route"]], colors(route_no-1), "dash", True)
                    folium_static(m, width = 800)
                    #Need to add the download option here!
                    m.save(f"{data_path_batch}/{drt[:-5]} - {uploaded_file.name[:-5]}_Routes.html")
                    st.markdown(f"**Map saved as an HTML: {scenario['Scenario']}_Routes.html**")
                else:
                    failed.append(f'{drt[:-5]} - {uploaded_file.name[:-5]}')
        st.success("All Optimizations Complete.")
        if len(failed) > 0:
            st.markdown("### Optimization Failed for:")
            for f in failed:
                st.markdown(f)

    return

# goal: return list of all files that are to be optimized and the reference file
def batch_file_upload(session_state):
    """Handles the file upload process into a dictionary
    Args:
        session_state (object): a new SessionState object
    """

    st.title("Upload File 📤")

    # get all files in data directory and process to include only correct format
    only_files = [f for f in listdir(f'./data/{country_name}') if isfile(join(f'./data/{country_name}', f))
                    and (len(f.split(' ')) >= 2) and f.split(' ')[1] == 'DRT']

    country_files = [f for f in only_files if f.split(' ')[0].title() == session_state.country.title()]
    if len(country_files) != 0:
        ref_index = only_files.index(country_files[0])
    else: 
        only_files = [""] + only_files
        ref_index = 0
    st.markdown('### Reference File for Batch Processing')
    ref_file = st.selectbox(
        label='Select reference file',
        options=only_files,
        index=ref_index,
        format_func=(lambda x : x.replace('.xlsx', ''))
    )

    if ref_file == "": 
        st.stop()

    st.markdown('### Upload Respective Order Evaluation Template')
    uploaded_files = st.file_uploader("", type=["xlsx", "xlsm"], key="OptFile",accept_multiple_files=True)
    

    if st.button("Confirm DRT and Order Evaluation Template File Selections"):
        session_state.opt_file_dict.update({ref_file:uploaded_files})
    
    
    st.markdown("***")

    st.markdown("### Selected DRT and Order Evaluation Template Files for optimization")

    for drt in session_state.opt_file_dict.keys():
        st.markdown(drt)
        for dro in session_state.opt_file_dict[drt]:
            st.markdown(f"- {dro.name}")

    if st.button("Clear All Optimizations Previously Selected"):
        session_state.opt_file_dict = {}
    
    st.markdown("***")

    st.markdown("## Finalize File Selection")
    if st.button("Confirm Files for Optimization"): 
        st.success("Files to be Processed Confirmed!")
        session_state.file_confirmation = True
        
    return 


if __name__ == "__main__":
    batch_main()

    