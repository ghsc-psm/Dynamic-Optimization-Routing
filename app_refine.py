import pandas as pd
import streamlit as st
from datetime import datetime
from scenario import save_scenario

def app_refine(session_state):
    """Allow user to refine and save a scenario.
    Args:
        session_state (object): a new SessionState object
    """

    st.markdown("## Refine Scenario Data 🖊", unsafe_allow_html=True)
    st.markdown("***")

    """ Instantiate so it will cache"""
    order_data = session_state.scenario_data["Facility_DF"].copy()
    initial_ff_pairs = session_state.scenario_data["Facility Groups"].copy()
    session_state.ff_pairs_dict = instantiate_ff_pairs(order_data, initial_ff_pairs, session_state.scenario_file)

    with st.expander("Manage fleet data", expanded=True):
        vf_exclusions_dict = select_vehicle_fleet_exclusions(session_state)

    # pair_facilities = st.expander("Group facility deliveries", expanded=True)
    # select_ff_pairs(session_state, pair_facilities)
    
    st.markdown("***")
    save_scenario_button = st.button("Update scenario")

    if save_scenario_button:
        order_data = session_state.scenario_data["Facility_DF"].copy()
        order_data.index = order_data.index.astype(str)
        fleet_df = session_state.scenario_data["Fleet_DF"].copy()
        fleet_df_cols = fleet_df.columns
        
        """ Updating Facility groups and vehicle exclusion"""
        session_state.scenario_data["Facility Groups"] = {k: set([order_data.index[order_data.facility == fac][0] for fac in v]) for k, v in session_state.ff_pairs_dict.items()}
        session_state.scenario_data["Vehicle Exclusion"] = {k: set([order_data.index[order_data.facility == fac][0] for fac in v]) for k, v in vf_exclusions_dict["excluded_facilities"].items() if len(v) > 0}
        
        """ Updating fleet information """
        fleet_df = fleet_df.join(pd.DataFrame.from_dict(vf_exclusions_dict["max_routes"], orient="index").rename(columns = {0: "max_routes"}), lsuffix="_old").drop(["max_routes_old"], axis=1)
        fleet_df["max_routes"] = fleet_df["max_routes"].astype(int)
        fleet_df = fleet_df.join(pd.DataFrame.from_dict(vf_exclusions_dict["available"], orient="index").rename(columns = {0: "available"}), lsuffix="_old").drop(["available_old"], axis=1)
        fleet_df = fleet_df[fleet_df_cols]
        session_state.scenario_data["Fleet_DF"] = fleet_df

        session_state.scenario_data['Modified'] = pd.to_datetime(datetime.now())
        session_state.scenario_data['Modified By'] = session_state.username
        save_scenario(session_state.scenario_file, session_state.scenario_data)
        session_state.scenario_ver_no += 1

        modification_date = session_state.scenario_data["Modified"].strftime('%m/%d/%Y %H:%M')
        st.markdown(f"###### Last updated: {modification_date}")

@st.cache(allow_output_mutation=True)
def instantiate_ff_pairs(order_data, initial_ff_pairs, scenario_file):
    """Create initial facility delivery groups.
    Args:
        order_data (pandas.core.frame.DataFrame): a pandas dataframe object containing order data
        
        initial_ff_pairs (dict): a dictionary object containing excluded vehicles fleets 
        
        scenario_file (file): a file containing scenario data
    Returns:
        ff_pairs_dict (dict): a dictionary object containing initial facility delivery groups 
    """

    """ When scenario file is changed, this gets run again"""
    ff_pairs_dict = dict()
    order_data.index = order_data.index.astype(str)

    
    for k, v in initial_ff_pairs.items():
        if initial_ff_pairs:
            val = [order_data.loc[str(facility_id), "facility"] for facility_id in v if str(facility_id) in order_data.index]
            """ Length of value can be 0 where global settings are not found in the current scenario """
            if len(val) != 0: 
                ff_pairs_dict[k] = val
    
    """ Add a blank spot """
    if len(initial_ff_pairs) == 0:
        ff_pairs_dict[len(initial_ff_pairs) + 1] = []
    else:
        ff_pairs_dict[max(max(initial_ff_pairs.keys()) + 1, len(initial_ff_pairs) + 1)] = []

    return ff_pairs_dict

def select_ff_pairs(session_state, pair_facilities):
    """Allow user to select facility delivery groups.
    Args:
        session_state (object): a new SessionState object 
        
        pair_facilities (object): a new SessionState object 
 
    """    

    pair_button = pair_facilities.button("Add another group")

    order_data = session_state.scenario_data["Facility_DF"].copy()
    order_data.index = order_data.index.astype(str)
    
    for k, v in session_state.ff_pairs_dict.items():
        session_state.ff_pairs_dict[k] = pair_facilities.multiselect("Which facilities should be delivered together?", options = list(order_data.facility[order_data.type != "Warehouse"]), default = v, key = str(k))

    """ Add another facility group if button is pressed"""
    if pair_button: 
        if session_state.ff_pairs_dict:
            session_state.ff_pairs_dict[max(session_state.ff_pairs_dict.keys()) + 1] = st.multiselect("Which facilities should be delivered together?", options = list(order_data.facility[order_data.type != "Warehouse"]), key = str(max(session_state.ff_pairs_dict.keys()) + 1))
        else:
            session_state.ff_pairs_dict[1] = pair_facilities.multiselect("Which facilities should be delivered together?", options = list(order_data.facility[order_data.type != "Warehouse"]), key = str(1))
            

def select_vehicle_fleet_exclusions(session_state):
    """Allow user to exclude vehicle fleets from certain facilities.
    Args:
        session_state (object): a new SessionState object 
         
 
    Returns:
        vf_exclusions_dict (dict): a dictionary object containing excluded vehicles fleets
    """      

    truck_title, space, excluded_facilities = st.columns([2, 0.5, 10])

    truck_title.markdown("### Truck type")
    excluded_facilities.markdown("### Exclude trucks from certain facilities")
    st.markdown("***")

    vf_exclusions_dict = dict()
    order_data = session_state.scenario_data["Facility_DF"].copy()
    order_data.index = order_data.index.astype(str)
    fleet_df = session_state.scenario_data["Fleet_DF"].copy()
    excluded_facilities = session_state.scenario_data["Vehicle Exclusion"].copy()

    vf_exclusions_dict["available"] = {index: row["available"] for index, row in fleet_df.iterrows() }
    vf_exclusions_dict["max_routes"] = {index: row["max_routes"] for index, row in fleet_df.iterrows() }
    vf_exclusions_dict["excluded_facilities"] = excluded_facilities

    vehicle_fleet_columns = {index: dict() for index, row in fleet_df.iterrows() }

    """ Define columns for each row"""
    for k, v in vehicle_fleet_columns.items():
        vehicle_fleet_columns[k]["available"], space, vehicle_fleet_columns[k]["excluded_facilities"] = st.columns([2, 0.5, 10])
        """ Don't draw a line for after the last option"""
        if k != list(vehicle_fleet_columns.keys())[-1]:
            st.markdown("***")

    for k, v in vf_exclusions_dict["available"].items():
        vf_exclusions_dict["available"][k] = vehicle_fleet_columns[k]["available"].checkbox(str(k), value=v, key = f"excl_{k}")
    
    for k, v in vf_exclusions_dict["max_routes"].items():
        # vf_exclusions_dict["max_routes"][k] = vehicle_fleet_columns[k]["max_routes"].text_input("Max # of routes", value=v, max_chars=2, key = f"max_{k}")

        if k in vf_exclusions_dict["excluded_facilities"]:
            val = [order_data.loc[str(facility_id), "facility"] for facility_id in vf_exclusions_dict["excluded_facilities"][k] if str(facility_id) in order_data.index]
        else:
            val = None

        vf_exclusions_dict["excluded_facilities"][k] = vehicle_fleet_columns[k]["excluded_facilities"].multiselect("Which facilities are EXCLUDED for " + str(k) + "?", 
                                                        options = list(order_data.facility[order_data.type != "Warehouse"]), default = val, key = str(k))

    return vf_exclusions_dict


