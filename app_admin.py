"""This module contains functions used for admin tasks."""

from datetime import time
from re import split
from numpy.lib.shape_base import column_stack
import streamlit as st
import pandas as pd
from toml import load
import openpyxl as op
from openpyxl.utils.dataframe import dataframe_to_rows
from streamlit.logger import update_formatter
from utils import get_table_download_link_xlsx, get_binary_file_downloader_html
from os import listdir, replace
from os.path import isfile, join
import io
import matrix_utils as matu
import traceback

SPEED = 45

def app_update_reference_data(session_state):
    """Allow user to update reference data.
    Args:
        session_state (object): a new SessionState object 
    """      

    st.markdown("## Update Reference Data 🔑", unsafe_allow_html = True)
    st.markdown("***")
    left_padding, file_select, upload_space, right_padding = st.columns([0.5, 2, 2, 0.5])
    #left_button, download_button, middle_button, upload_button, right_button = st.columns([1.2, 1.3, .5, 1, 1])

    with upload_space:
        uploaded_file = st.file_uploader("", type="xlsx", key='AdminUpload')

    with file_select:
        # get all files in data directory and process to include only correct format
        data_path = './data/COUNTRY'
        app_list = ['DRT', 'ALL', 'DRO']
        only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))
                     and (len(f.split(' ')) >= 2) 
                     and f.split(' ')[1].replace('.xlsx','') in app_list
                     and f[-4:] == 'xlsx']

        ref_file = st.radio(
            label='Choose reference file to update:',
            options=only_files,
            format_func=(lambda x : x.replace('.xlsx', ''))
        )
        session_state.ref_file = join(data_path, ref_file)

        with upload_space:
            upload = st.button("Update reference data")
            if uploaded_file is None and upload:
                st.error("Please upload a file")
            elif uploaded_file is not None and upload:
                if 'DRT' in session_state.ref_file:
                    validate, message = validate_drt_reference_data(uploaded_file)
                elif 'Facility' in session_state.ref_file: 
                    validate, message = validate_fac_mapping_date(uploaded_file)
                elif 'Order' in session_state.ref_file: 
                    validate, message = validate_order_date(uploaded_file)
                else:
                    validate = True
                    message = 'no validation for this data type'

                if validate:
                    with open(session_state.ref_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if 'DRT' in session_state.ref_file:
                        facs = pd.read_excel(session_state.ref_file, sheet_name='Facility')
                        links = pd.read_excel(session_state.ref_file, sheet_name='Links')

                        if links.shape[0]:
                            for idx, row in links.iterrows():
                                if pd.isnull(row['Link Time']):
                                    links.loc[idx, 'Link Time'] = row['Link Distance'] / SPEED

                            distance_matrix = matu.link_list_to_mat(links, facs, False)
                            time_matrix = matu.link_list_to_mat(links, facs, True)

                            book = op.load_workbook(session_state.ref_file)
                            writer = pd.ExcelWriter(session_state.ref_file, engine='openpyxl')
                            writer.book = book
                            writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

                            distance_matrix.to_excel(writer, sheet_name='Distance', index=False)
                            time_matrix.to_excel(writer, sheet_name='Time', index=False)
                            writer.save()

                    st.success(f"{session_state.ref_file} updated successfully. \n{message}")
                else:
                    st.error(message)

        with file_select:
            download = st.button("Download reference data")

            if download:
                st.markdown(get_binary_file_downloader_html(join(data_path, ref_file), ref_file), unsafe_allow_html=True)
                # download_reference_dict = create_reference_dict(session_state)
                # filename = session_state.ref_file.split("/")[-1].split(".")[0]
                # st.markdown(get_table_download_link_xlsx("reference data", filename, **dict(download_reference_dict)), unsafe_allow_html=True)
    
    with st.expander("Filestore Data Retrieval", expanded=False):
        fs_path = './filestore/'
        only_fs_files = [f for f in listdir(fs_path) if isfile(join(fs_path, f))
                     and f[-4:] in {'xlsx', 'xlsm', '.zip'}]

        fs_file = st.radio(
            label='Choose filestore file to download:',
            options=only_fs_files
        )
        if(st.button("Download filestore file")): 
            st.markdown(get_binary_file_downloader_html(join(fs_path, fs_file), fs_file), unsafe_allow_html=True)


def validate_order_date(uploaded_file): 
    """Verify the columns in a order data file.
    Args:
        uploaded_file (file): an excel file containing order data
    Returns:
        (bool): True if uploaded_file meets criteria, False otherwise
    """     

    try: 
        order_details_df = pd.read_excel(uploaded_file) 
        assert 'Customer ID' in order_details_df.columns
        assert 'Customer Order Number' in order_details_df.columns
        assert 'Unit Weight' in order_details_df.columns
        assert 'Line Weight' in order_details_df.columns
    except AssertionError as error:
        return False, f"Validation failure, please verify. {error}"
    except Exception as exception:
        return False, f"Validation failure, please verify. {exception}" 

    return True, "Validation successful"

def validate_fac_mapping_date(uploaded_file): 
    """Verify the columns in a FAC data file.
    Args:
        uploaded_file (file): an excel file containing order data
    Returns:
        (bool): True if uploaded_file meets criteria, False otherwise
    """    

    try: 
        fac_df = pd.read_excel(uploaded_file, sheet_name='Fac Mapping')
        assert 'WMS_Fac_Code' in fac_df.columns
        assert 'Delivery_Destination_Name' in fac_df.columns
        assert 'District_Health_Office_Code' in fac_df.columns
    except AssertionError as error:
        return False, f"Validation failure, please verify. {error}"
    except Exception as exception:
        return False, f"Validation failure, please verify. {exception}" 

    return True, "Validation successful"

def validate_drt_reference_data(uploaded_file):
    book = op.load_workbook(uploaded_file)
    sheet_names = ["Facility", "Distance", "Links", "Time", "Cubage", "Fleet", "Fleet Exclusions", "Facility Groups", "Distance Adj", "Parameters"]
    try:
        # """ Assert all appropriate sheet names are in reference data """
        # for sheet_name in sheet_names:
        #     assert sheet_name in book.sheet_names, f'{sheet_name} sheet not in reference data'
        
        """ Validating facility data"""
        facility = pd.read_excel(uploaded_file, sheet_name = "Facility")
        facility_cols = ["facility", "facility_id", "type", "latitude", "longitude"] 
        for col in facility_cols:
            assert col in facility.columns, f'{col} not in Facility columns'

        assert "Central" in pd.unique(facility.type) or "ZAMMSA Hub" in pd.unique(facility.type), "Expected to have at least one facility that is either Central or ZAMMSA Hub"

        """ Validating fleet data"""
        fleet = pd.read_excel(uploaded_file, sheet_name = "Fleet")
        fleet_cols = ["warehouse", "truck_type", "max_routes", "vol_cap", "weight_cap", "dist_cap", "transit_time_cap", "delivery_time_cap", "speed", "base_cost", "liters_per_km", "fuel_price"] 
        for col in fleet_cols:
            assert col in fleet.columns, f'{col} not in Fleet columns'

        """ Validating fleet exclusions"""
        fleet_exclusions = pd.read_excel(uploaded_file, sheet_name = "Fleet Exclusions")
        fleet_exclusions_cols = ["warehouse", "truck_type", "facility_id"] 
        for col in fleet_exclusions_cols:
            assert col in fleet_exclusions.columns, f'{col} not in Fleet Exclusions columns'
        assert set(fleet_exclusions.facility_id.astype(str)) - set(facility.facility_id.astype(str)) == set(), f'At least one facility ID in Fleet Exclusions not found in Facility table'

        """ Validating facility groups """
        facility_groups = pd.read_excel(uploaded_file, sheet_name = "Facility Groups")
        facility_groups_cols = ["group_id", "facility_id"] 
        for col in facility_groups_cols:
            assert col in facility_groups.columns, f'{col} not in Facility Groups columns'
        # assert set(facility_groups.facility_id.astype(str)) - set(facility.facility_id.astype(str)) == set(), f'At least one facility ID in Facility Groups not found in Facility table'

        """ Validating and processing links """
        links = pd.read_excel(uploaded_file, sheet_name = "Links")
        links_cols = ["Origin Facility", "Origin Facility Code", "Destination Facility", "Destination Facility Code", "Link Distance", "Link Time"]
        for col in links_cols:
            assert col in links.columns, f'{col} not in Links columns'

        # ensure all facilities are present in links
        links['Origin Pair'] = list(zip(links['Origin Facility'], links['Origin Facility Code']))
        links['Destination Pair'] = list(zip(links['Destination Facility'], links['Destination Facility Code']))
        unique = list(set(links['Origin Pair'].to_list() + links['Destination Pair'].to_list()))

        if links.shape[0]:
            assert facility.shape[0] ==  len(unique), f'Facility mismatch in links'
    

        # """ Validating Distance adj """
        # distance_adj = pd.read_excel(uploaded_file, sheet_name = "Distance Adj")
        # distance_adj_cols = ["from_facility_id", "to_facility_id", "distance_adj"]
        # for col in distance_adj_cols:
        #     assert col in distance_adj.columns, f'{col} not in Distance Adj columns'
        # assert set(distance_adj.from_facility_id.astype(str)) - set(facility.facility_id.astype(str)) == set(), f'At least one facility ID in Distance Adj not found in Facility table'
        # assert set(distance_adj.to_facility_id.astype(str)) - set(facility.facility_id.astype(str)) == set(), f'At least one facility ID in Distance Adj not found in Facility table'

        # """ Make sure distance and time matrix includes every facility """
        # distance = pd.read_excel(uploaded_file, sheet_name = "Distance")
        # time = pd.read_excel(uploaded_file, sheet_name = "Time")
        
        # assert time.shape[0] == time.shape[1], f'Time matrix: {time.shape[0]} x {time.shape[1]} is not n x n matrix'
        # assert distance.shape[0] == distance.shape[1], f'Distance matrix: {distance.shape[0]} x {distance.shape[1]} is not n x n matrix'
        # assert facility.shape[0] == distance.shape[0], f'Number of facilities ({facility.shape[0]}) does not match distance matrix {distance.shape[0]} x {distance.shape[1]}'

        # """ Make sure distance and time matrix have no NAs """
        # assert distance.isna().sum().sum() == 0, f'There are {distance.isna().sum().sum()} missing distances in the distance matrix'
        # assert time.isna().sum().sum() == 0, f'There are {time.isna().sum().sum()} missing times in the time matrix'

        return True, "Validation successful"
    except AssertionError as error:
        return False, f"Validation failure, please verify. {error}"
    except Exception as exception:
        return False, f"Validation failure, please verify. {exception}" 

def create_reference_dict(session_state):
    """Create a dictionary object containing each sheet of a reference file.
    Args:
        session_state (object): a new SessionState object 
    Returns:
        download_df_dict (dict): a dictionary object containing each sheet of a reference file
    """

    download_df_dict = dict()
    if session_state.ref_file[-4:] == 'xlsx': 
        sheet_names = pd.ExcelFile(session_state.ref_file).sheet_names
        for sheet_name in sheet_names:
            download_df_dict[sheet_name] = pd.read_excel(session_state.ref_file, sheet_name = sheet_name)
    return download_df_dict


def create_matricies(links, facs):
    """Create a rectangular array used to represent distance and time relationships.
    Args:
        links (pandas.core.frame.DataFrame): a dataframe object containing link data
        
        facs (pandas.core.frame.DataFrame): a dataframe object containing FAC data
    
    Returns:
        distance_matrix (pandas.core.frame.DataFrame): a dataframe object containing a distance matrix
        
        time_matrix (pandas.core.frame.DataFrame): a dataframe object containing a time matrix
    """       

    for idx, row in links.iterrows():
            if not row['Link Time']:
                est_time = row['Link Distance'] / SPEED
                links.loc[idx, 'Link Time'] = est_time

    # create matricies
    distance_matrix = matu.link_list_to_mat(links, facs, False)
    time_matrix = matu.link_list_to_mat(links, facs, True)

    return distance_matrix, time_matrix
