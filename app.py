import streamlit as st
from PIL import Image
from pathlib import Path
st.set_page_config(layout="wide", initial_sidebar_state='auto')
import SessionState
import traceback
import pandas as pd
from datetime import datetime
import random
import string
import configparser
import psm.db
from app_upload import app_upload
from app_refine import app_refine
from app_review import app_review
from app_optimize import app_optimize
from app_login import check_credentials
from app_admin import app_update_reference_data
from app_drorders import app_drorders
from app_release_notes import app_release_notes

from scenario import create_download_dict, save_scenario
from utils import get_table_download_link_xlsx, get_email_agent, get_binary_file_downloader_html

from psm.email_utility import * 

filestore_path = os.path.join(os.getcwd(), 'filestore')
# Check if the filestore folder exists
if not os.path.exists(filestore_path):
    # Create the filestore folder if it does not exist
    os.makedirs(filestore_path)

def app_main():
    """Allow user to navigate to each page of the application."""

    country_config = configparser.ConfigParser()
    country_config.read("./country_cfg.toml")
    country_name  = "COUNTRY" if country_config.get("country", "country_name").title() =="" else country_config.get("country", "country_name").title()

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
                                     )


    
    banner_path = './images/banner.jpg'
    tool_title = "Dynamic Routing Tool"
    if session_state.country != "":
        if country_config.get("country", "tool_title").title() != "":
            tool_title = country_config.get("country", "tool_title").title()
        else:
            tool_title = f'{session_state.country.capitalize()} Routing Tool'
        try: 
            banner_path = [str(f) for f in Path("images").rglob(f'{session_state.country}*')][0]
        except:
            pass
               
    
    st.sidebar.title(tool_title)
    st.sidebar.markdown(f"###### {session_state.ver}")
    
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
    st.image(Image.open(banner_path), use_column_width=True)
    st.markdown('<style>' + open('icons.css').read() + '</style>', unsafe_allow_html=True)
    
    warning_scenario_message = "Please initialize scenario in **Upload data**"
    warning_solve_message = "Please solve scenario to review results in **Solve scenario** "
    
    # facility group problems 

    # add an index into the distance and time (for both reference and when reading and writing data)
    # will have to change the way I'm subsetting data and stuff
    
    if session_state.username == "":
        psm.db.encrypt("Password1")
        check_credentials(session_state)
    else:
        st.sidebar.markdown(f'### Hello, {"Admin" if session_state.user_type == "admin" else session_state.username}')

        app_options = ["Order Evaluation", "Dispatch Optimization", "Release Notes"]
        show_app_options = [f for f in app_options if country_config.get('drt_functionality', f).title() == 'True']
        
        if session_state.user_type == "admin":
            show_app_options = ["Tool Administration"] + app_options
        app_mode = st.sidebar.selectbox("", show_app_options)

        title_column, filename_column, space = st.columns([3.8, 1, 0.2])
        title_column.title(app_mode)

        if app_mode == "Dispatch Optimization":
            if session_state.scenario_data: 
                routing_options = ["Upload data", "Refine data", "Solve scenario", "Review results"] 
            else:
                routing_options = ["Upload data"]
            routing_mode = st.sidebar.radio("", routing_options) 

            # Populate scenario info
            if session_state.scenario_data is not None: 
                # st.sidebar.markdown(f"### Scenario {session_state.scenario_data['Scenario']}")
                filename_column.markdown(f"### Scenario {session_state.scenario_data['Scenario']}")
                download_scenario = filename_column.button("Download scenario")
                if download_scenario:
                    filename_column.markdown(get_binary_file_downloader_html(session_state.scenario_file, f"Scenario  {session_state.scenario_data['Scenario']}"), unsafe_allow_html=True)

                    # download_df_dict = create_download_dict(session_state.scenario_data)
                    # filename = session_state.scenario_data["Scenario"]
                    # filename_column.markdown(get_table_download_link_xlsx("scenario", f"Scenario {filename}", **dict(download_df_dict)), unsafe_allow_html=True)

                # Due to email policy change, disable this functionality for now.
                # email_scenario = filename_column.button("Email scenario")
                # if email_scenario:
                #     agent = get_email_agent()
                #     subject = f'{session_state.country.capitalize()} Routing tool: Scenario file {session_state.scenario_data["Scenario"]}'
                #     body = "Please find the scenario file attached"
                #     save_scenario(session_state.scenario_file, session_state.scenario_data)
                #     agent.send(session_state.username, subject, body, "", msg_type="html", attach_file=session_state.scenario_file)
                #     send_date = datetime.now().strftime('%m/%d/%Y %H:%M')
                #     filename_column.markdown(f"###### Sent: {send_date}")
                    
            try:
                if routing_mode == "Upload data":
                    app_upload(session_state)
                elif routing_mode == "Refine data":
                    if session_state.scenario_data is None:
                        st.markdown(f"### {warning_scenario_message}")
                    else:
                        app_refine(session_state)
                elif routing_mode == "Solve scenario":
                    if session_state.scenario_data is None:
                        st.markdown(f"### {warning_scenario_message}")
                    else:
                        app_optimize(session_state)
                elif routing_mode == "Review results":
                    if session_state.scenario_data is None:
                        st.markdown(f"### {warning_scenario_message}")
                    elif pd.isna(session_state.scenario_data['Solved']):
                        st.markdown(f"### {warning_solve_message}")
                    else:
                        app_review(session_state)
            except Exception as e:
                st.error(f"Application Error: {e}\n{traceback.format_exc()}")
        elif app_mode == "Tool Administration":
            app_update_reference_data(session_state)
        elif app_mode == "Order Evaluation": 
            app_drorders(session_state)
        elif app_mode == "Release Notes": 
            app_release_notes(session_state)
       


if __name__ == "__main__":
    app_main()