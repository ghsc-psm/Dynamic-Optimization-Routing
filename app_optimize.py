"""This module contains functions used to optimize scenarios."""

from requests import session
import streamlit as st
import pandas as pd
import time
from math import ceil
from datetime import datetime
import configparser
import numpy as np
from copy import deepcopy
import math

from DRO import DRO, DROFacility, DROVehicle, DROSpec, DROSpecDesc, DROINF
from psm import env as psmenv
from scenario import initialize_scenario, read_scenario, save_scenario
from truck_optimize import truck_optimize

from ortools.linear_solver import pywraplp

SCALE = {'VOL':100, 
         'WGT':10, 
         'TIME':60,
         'DIST':10,
         'VALUE':10}

def get_transit(dro_data, facility_id):
    """Calculate transit details from DRO Data.
    Args:
        dro_data (pandas.core.frame.DataFrame): a pandas dataframe object containing DRO data
        
        facility_id (str): a unique identifier for a facility
    Returns:
        (float): a float value denoting transit distance
        
        (float): a float value denoting transit time
    """    

    o = dro_data['depot_no'] 
    d = dro_data['facility_no'][facility_id]
    return dro_data['distance'][o][d]/SCALE['DIST'], dro_data['time'][o][d]/SCALE['TIME']

def get_fleet(scenario_data, facility_id, transit_dist): 
    """Get fleet that meets scenario requirements.
    Args:
        scenario_data (pandas.core.frame.DataFrame): a pandas dataframe object containing scenario data
       
        facility_id (str): a unique identifier for a facility
        
        transit_dist (float): a float value denoting transit distance
    Returns:
         fleet (dict): a dictionary object containing fleet information for scenario
    """ 

    fleet = {}
    for t, t_row in scenario_data['Fleet_DF'].iterrows(): 
        if not t_row['available']: 
            continue
        if (t in scenario_data["Vehicle Exclusion"] and facility_id in scenario_data["Vehicle Exclusion"][t]): 
            continue
        if transit_dist/(1e-9 + t_row['speed']) > t_row['transit_time_cap'] or transit_dist > t_row['dist_cap']: 
            continue
        fleet[t] = {c:t_row[c] for c in scenario_data['Fleet_DF'].columns}
    return fleet

def optimize_dedicated_delivery(scenario_data, dro_data, facility_id): 
    """Perform knapsack optimization for each delivery.
    Args:
        scenario_data (pandas.core.frame.DataFrame): a pandas dataframe object containing scenario data
        
        dro_data (pandas.core.frame.DataFrame): a pandas dataframe object containing DRO data
        
        facility_id (str): a unique identifier for a facility
    
    Returns:
        (bool): True if optimization criteria is met, False otherwise
    """     

    transit_dist, transit_time = get_transit(dro_data, facility_id)
    fleet = get_fleet(scenario_data, facility_id, transit_dist)
    vol = scenario_data["Facility_DF"].loc[facility_id]['vol']
    weight = scenario_data["Facility_DF"].loc[facility_id]['weight']

    if len(fleet) == 0 or transit_dist == 99999 or pd.isna(vol) or vol==0: 
        return False
    if scenario_data["Parameters"]["Enforce Weight Capacity"] and (pd.isna(weight) or weight==0): 
        return False

    warehouse = scenario_data['Facility_DF'].loc[dro_data['depot_id'], "facility"]
    fac_name = scenario_data['Facility_DF'].loc[facility_id, "facility"]

    t_type_ordered = [t_type for (t_type, cap) in sorted([(t_type, fleet[t_type]['vol_cap']) for t_type in fleet], key=lambda t: t[1], reverse=True)]


    ## For the dedicated dispatch solver
    solver = pywraplp.Solver('Calculator', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    x = {t_type:solver.IntVar(0.0, 5+ceil(vol/(1e-9+fleet[t_type]['vol_cap'])), f"{t_type}") for t_type in fleet} 
    solver.Add(sum(fleet[t_type]['vol_cap']*x[t_type] for t_type in fleet) >= vol)
    if scenario_data["Parameters"]["Enforce Weight Capacity"]: 
        solver.Add(sum(fleet[t_type]['weight_cap']*x[t_type] for t_type in fleet) >= weight)
    per_km_cost = {t_type: fleet[t_type]['base_cost']+fleet[t_type]['liters_per_km']*fleet[t_type]['fuel_price'] for t_type in fleet}
    solver.Minimize(sum(per_km_cost[t_type]*x[t_type] for t_type in fleet))

    ##
    
    route_no = len(scenario_data["SolSummary_DF"])+1
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        route_detail = []
        route_summary = []
        remaining_vol = vol
        remaining_weight = weight
        density = weight/vol
        for t_type in t_type_ordered: 
            for _ in range(int(x[t_type].solution_value())):
                vol_cap = fleet[t_type]['vol_cap']
                weight_cap = fleet[t_type]['weight_cap']

                vol_to_load = min(remaining_vol, vol_cap)
                if vol_to_load <= 0:
                    continue

                dispatch_id = f"Dispatch {route_no:03}"
                weight_to_load = vol_to_load*density
                per_km_rate = per_km_cost[t_type]
                per_km_fuel = fleet[t_type]['liters_per_km']
                stops = [warehouse, fac_name]
                route_detail.append((dispatch_id, 0, dro_data['depot_id'], 0, 0, 0, 0))
                route_detail.append((dispatch_id, 1, facility_id, transit_dist, transit_time, per_km_fuel*transit_dist, per_km_rate*transit_dist))
                if scenario_data["Parameters"]["Include Return Leg Cost"]: 
                    route_detail.append((dispatch_id, 2, dro_data['depot_id'], transit_dist, transit_time, per_km_fuel*transit_dist, per_km_rate*transit_dist))
                    stops.append(warehouse)

                route_summary.append((dispatch_id, t_type, " -> ".join(stops), 1, vol_to_load, weight_to_load))

                if facility_id not in scenario_data["SolFacDispatches"]: 
                    scenario_data["SolFacDispatches"][facility_id] = []
                scenario_data["SolFacDispatches"][facility_id].append(dispatch_id)
                scenario_data["SolDispatches"][dispatch_id] = (t_type, vol_cap, weight_cap)

                remaining_vol -= vol_to_load
                remaining_weight -= weight_to_load                 
                route_no += 1

        if len(route_summary) > 0: 
            route_detail_df = pd.DataFrame(route_detail, columns=['route', 'stop_no', 'facility_id', 'distance', 'time', 'fuel_usage', 'cost'])
            route_summary_df = pd.DataFrame(route_summary, columns=['route', 'truck_type', 'path', 'num_stops', 'vol', 'weight'])

            route_detail_df = pd.merge(route_detail_df, scenario_data['Facility_DF'].reset_index(), how='left', on='facility_id')
            route_sum = route_detail_df.groupby("route").agg({"distance":"sum", "time":"sum", "fuel_usage":"sum", "cost":"sum"}).reset_index()
            
            route_summary_df = pd.merge(route_summary_df, route_sum, how='left', on='route')
            route_summary_df = pd.merge(route_summary_df, scenario_data['Fleet_DF'].reset_index()[["truck_type", 'vol_cap', 'weight_cap']], how='left', on='truck_type')
            route_summary_df["vol_utilization"] = route_summary_df['vol']/route_summary_df['vol_cap']
            route_summary_df["weight_utilization"] = route_summary_df['weight']/route_summary_df['weight_cap']
            route_summary_df["source"] = 'Capacity Solver'

            scenario_data["SolSummary_DF"] = pd.concat([scenario_data["SolSummary_DF"], route_summary_df], ignore_index=True)
            scenario_data["SolDetail_DF"] = pd.concat([scenario_data["SolDetail_DF"], route_detail_df], ignore_index=True)
            return True
    else: 
        st.error(f"Knapsack optimization failed for {facility_id}: {status}")

    return False


def create_pick_waves(scenario_data, dro_data): 
    """Create batches of shipment lines that are pick released together based on scenario criteria.
    Args:
        scenario_data (pandas.core.frame.DataFrame): a pandas dataframe object containing scenario data
       
        dro_data (pandas.core.frame.DataFrame): a pandas dataframe object containing DRO data
    """    

    order_df = pd.merge(scenario_data['Order Info'][['Route', 'Order Number', 'Customer Name', 'Customer ID', 'District', 'Province', 'Loading Volume']], 
                        scenario_data['Deliveries'][['Route', 'Customer ID', 'Dispatch Destination', 'Dispatch Destination Code']], 
                        on = ['Route', 'Customer ID'])
    order_df["Program"] = order_df['Order Number'].apply(lambda v: v[:2])
    order_df = order_df.sort_values(['Route', 'District', 'Customer Name', "Program"])

    wave_orders = []
    dispatch_remaining_cap = {dispatch_id: scenario_data["SolDispatches"][dispatch_id][1] for dispatch_id in scenario_data["SolDispatches"]}
    for _, r in order_df.iterrows(): 
        assigned = False
        fac_id = r['Dispatch Destination Code']
        if fac_id in scenario_data["SolFacDispatches"]: 
            if len(scenario_data["SolFacDispatches"][fac_id]) == 1: 
                d_id = scenario_data["SolFacDispatches"][fac_id][0]
                wave_orders.append([r['Route'], r['Order Number'], r['Program'], r['Customer Name'], r['Customer ID'], r['District'], r['Province'], r['Loading Volume'], 
                                    d_id, scenario_data["SolDispatches"][d_id][0], r['Dispatch Destination']])
                dispatch_remaining_cap[d_id] -= r['Loading Volume']
                assigned = True
            else:
                remaining_vol = r['Loading Volume']
                for n in range(len(scenario_data["SolFacDispatches"][fac_id])):
                    d_id = scenario_data["SolFacDispatches"][fac_id][n]
                    if dispatch_remaining_cap[d_id]>0 and remaining_vol>0: 
                        load_vol = min(remaining_vol, dispatch_remaining_cap[d_id])
                        wave_orders.append([r['Route'], r['Order Number'], r['Program'], r['Customer Name'], r['Customer ID'], r['District'], r['Province'], load_vol, 
                                            d_id, scenario_data["SolDispatches"][d_id][0], r['Dispatch Destination']])
                        dispatch_remaining_cap[d_id] -= load_vol
                        remaining_vol -= load_vol
                        assigned = True
                # print(f"{fac_id} has multiple dispatches. {r['Loading Volume']} {assigned}, {remaining_vol}")
        else: 
            # print(f"Warning: {fac_id} doesn't have disaptch assigned")
            pass 

        if not assigned: 
            wave_orders.append([r['Route'], r['Order Number'], r['Program'], r['Customer Name'], r['Customer ID'], r['District'], r['Province'], r['Loading Volume'], 
                                "", "", r['Dispatch Destination']])

    if len(wave_orders) > 0: 
        wave_orders = sorted(wave_orders, key=lambda r: (r[8], r[0], r[5], r[3]))  # sort by 'Dispatch No', 'Route', 'District', 'Customer'
        wave_orders[0].append(1)
        for n in range(1, len(wave_orders)): 
            last_wn = wave_orders[n-1][-1]
            if wave_orders[n][8] != wave_orders[n-1][8]:        # different dispatch
                wave_orders[n].append(last_wn+1)
            elif wave_orders[n][5] != wave_orders[n-1][5]:       # different district
                wave_orders[n].append(last_wn+1)
            else: 
                wave_orders[n].append(last_wn)

        wave_orders_df = pd.DataFrame(wave_orders, columns=['Route', 'Order Number', 'Program', 'Customer', 'Customer ID', 'District', 'Province', 'Est. Volume (m3)', 'Dispatch No', 'Truck Type', 'Dispatch Destination', 'Pick Wave No'])
        wave_orders_df['Pick Wave'] = wave_orders_df['Pick Wave No'].apply(lambda n: f"PW{n:03}")
        scenario_data["Loading Plan"] = scenario_data["Loading Plan"].append(wave_orders_df[['Dispatch No', 'Truck Type', 'Dispatch Destination', 'Route', 'Province', 'District', 'Pick Wave', 'Customer', 'Order Number', 'Est. Volume (m3)']], ignore_index = True)
       
def to_dro_data(scenario_data): 
    """Transform scenario data to DRO Data.
    Args:
        scenario_data (dict): a dictionary object containing secenario data
    Returns:
        (dict): a dictionary object containing data needed to run DRO
    """
    """
    DRO Data contains the following data elements: 
    - spec: DROSpec object
    - facilities: list of DROFacility objects
    - facility_no: dict from facility id to facility no
    - vehicles: list of DROVehicle objects
    - distance: list of lists, n by n distance matrix
    - time: list of lists, n by n transit time matrix
    - vf_exclusions: set of (vehicle no, facility no) tuples for vehicle/facility exclusion
    - ff_pairs: set of (facility no, facility no) tuples for facility pairs on the same route
    - dist_updates: set of (from facility no, to facility no, updated distance) tuples for distance updates
    """

    dro_data = {}

    dedicated_dispatch_destinations = set(scenario_data["Deliveries"][scenario_data["Deliveries"]['Dedicated Trucks']=='Yes']['Dispatch Destination Code'].to_list())
    pars = scenario_data["Parameters"]
    dro_data['spec'] = DROSpec(pars["Include Return Leg Cost"],        # include return cost
                        pars["Enforce Volume Capacity"],               # enforce volume cap
                        pars["Enforce Weight Capacity"],               # enforce weight cap
                        pars["Enforce Distance Limit"],                # enfroce distance cap
                        pars['Enforce Transit Time Limit'],            # enfroce transit time cap
                        pars['Enforce Delivery Time Limit'],           # include delivery time cap
                        pars['Allow Missed Deliveries'],               # allow miss deliveries
                        pars["Optimization Runtime Limit"],            # time limit in seconds
                        pars["Adjust Transit Time by Speed"],          # use speed for transit time
                )
    
    try:
        if len(scenario_data["Facility_DF"]['load_mins']) > 0 :
            no_load_tab = False
        else:
            no_load_tab = True
    except:
        no_load_tab = True
    if no_load_tab:
        st.error('No column named \'load_mins\' in Facilities Tab in DRT File, defaulting to 15 minutes for vehicle load time for optimization.')
    dro_data['facilities'] = [DROFacility(r['facility_id'],                 # facility name
                                            0 if r['facility_id'] in dedicated_dispatch_destinations else int(math.ceil((0 if pd.isna(r['vol']) else r['vol'])*SCALE['VOL'])),     # Scaled volume
                                            0 if r['facility_id'] in dedicated_dispatch_destinations else int(math.ceil((0 if pd.isna(r['weight']) else r['weight'])*SCALE['WGT'])),  # Scaled weight
                                            0 if r['facility_id'] in dedicated_dispatch_destinations else DROINF,                         # Miss penalty
                                            0 if r['facility_id'] in dedicated_dispatch_destinations else int(math.ceil(.25*SCALE['TIME'] if no_load_tab else r['load_mins']))                         # Processing time
                                        ) for _, r in scenario_data["Facility_DF"].reset_index().iterrows()]

    dro_data['depot_no'] = -1
    dro_data['depot_id'] = -1
    n = 0
    for _, r in scenario_data["Facility_DF"].reset_index().iterrows(): 
        if r['type'] == 'Warehouse': 
            dro_data['depot_no'] = n
            dro_data['depot_id'] = r['facility_id']
        n += 1
    if dro_data['depot_no'] == -1: 
        st.error("Missing depot in DRO data. ")
        st.stop()

    dro_data['facility_no'] = {str(f.id):n for n, f in enumerate(dro_data['facilities'])}

    dro_data['vehicles'] = []
    message_displayed = False
    for _, r in scenario_data["Fleet_DF"].reset_index().iterrows(): 
        if r['available']: 
            for i in range(r['max_routes']): 
                try:
                    fixed_cost = int(r['fixed_cost']*SCALE['VALUE'])
                except:
                    fixed_cost = 0
                    if not message_displayed:
                        st.error('Please update DRT File to include new column named \"fixed_cost\" in the Fleet Tab. This new column represents the fixed cost associated with using a specific vehicle type. Fixed Cost defaulting to 0.')
                        message_displayed = True
                dro_data['vehicles'].append(
                    DROVehicle(r['truck_type'],                                 # truck type/name
                                int(r['vol_cap']*SCALE['VOL']),                 # Scaled volume capacity
                                int(r['weight_cap']*SCALE['WGT']),              # Scaled weight capacity
                                int(r['dist_cap']*SCALE['DIST']),               # Scaled distance limit
                                int(r['transit_time_cap']*SCALE['TIME']),       # Scaled transit time limit
                                int(r['delivery_time_cap']*SCALE['TIME']),      # Scaled delivery time limit
                                int(r['speed']*SCALE['DIST']/SCALE['TIME']),    # Scaled speed
                                fixed_cost,                                              # Fixed cost
                                int((r['base_cost']+r['liters_per_km']*r['fuel_price'])*SCALE['VALUE']/SCALE['DIST']), # Scaled variable cost
                                0,                                              # Per Drop cost
                            ))

    dro_data['distance'] = (scenario_data["Distance_DF"]*SCALE['DIST']).astype(int).values.tolist()
    dro_data['time'] = (scenario_data["Time_DF"]*SCALE['TIME']).astype(int).values.tolist()
    
    dro_data['vf_exclusions'] = set()
    for n, v in enumerate(dro_data['vehicles']): 
        if v.type in scenario_data['Vehicle Exclusion']:
            for fac_id in scenario_data['Vehicle Exclusion'][v.type]: 
                if str(fac_id) in dro_data['facility_no']:
                    dro_data['vf_exclusions'].add((n, dro_data['facility_no'][str(fac_id)]))
    
    dro_data['ff_pairs'] = set()
    for grp in scenario_data['Facility Groups'].values(): 
        grp = [fac_id for fac_id in grp if fac_id in dro_data["facility_no"]]
        if len(grp) >= 2: 
            fac_no_0 = dro_data['facility_no'][grp[0]]
            for i in range(1, len(grp)): 
                dro_data['ff_pairs'].add((fac_no_0, dro_data['facility_no'][grp[i]]))
    
    dro_data['dist_updates'] = {(dro_data['facility_no'][s], dro_data['facility_no'][t], min(DROINF, int(v*SCALE['DIST']))) for (s,t), v in scenario_data['Distance Adj'].items() if s in dro_data["facility_no"] and t in dro_data["facility_no"]}
    return dro_data

def app_optimize(session_state):
    """Perform optimization based on scenario criteria.
    Args:
        session_state (object): a new SessionState object 
    """    

    scenario = session_state.scenario_data
    st.markdown(f"## Optimize Scenario ({scenario['Scenario']}) ✨", unsafe_allow_html=True)
    st.markdown("***")

    opt_parameters, use_predefined_routes = set_parameters(scenario)

    st.markdown("***")
    col_left, col_button, col_status, col_right = st.columns([0.5, 0.5, 1.5, 0.5])
    if col_button.button("Run Optimization"): 
        run_optimization(session_state, opt_parameters, use_predefined_routes)

        with st.expander("Debug Data", expanded=False):
            st.text(f"Scenario Verion No: {session_state.scenario_ver_no}")
            dro_data = to_dro_data(session_state.scenario_data)
            st.text(f"{len(dro_data['facilities'])} facilities: {dro_data['facilities']}")
            st.text(f"{len(dro_data['vehicles'])} vehicles: {dro_data['vehicles']}")
            st.text(f"spec: {dro_data['spec']}")
            st.text(f"distance: {dro_data['distance']}")
            st.text(f"time: {dro_data['time']}")
            st.text(f"fac_no: {dro_data['facility_no']}")
            st.text(f"vf_exclusions: {dro_data['vf_exclusions']}")
            st.text(f"ff_pairs: {dro_data['ff_pairs']}")
            st.text(f"dist_updates: {dro_data['dist_updates']}")
            # Note: Commented out as it has thrown errors in specific scenarios
            #st.text("Original Missed Deliveries:")
            #st.dataframe(missed_deliveries_df)
            st.text("Route Details:")
            st.dataframe(session_state.scenario_data["SolDetail_DF"])
            st.text("Route Summary:")
            st.dataframe(session_state.scenario_data["SolSummary_DF"])
            st.text("Missed Deliveries:")
            st.dataframe(session_state.scenario_data["SolMiss_DF"])
            st.text(f"Dispatches: {session_state.scenario_data['SolDispatches']}")
            st.text(f"Facility Dispatches: {session_state.scenario_data['SolFacDispatches']}")
            st.dataframe(session_state.scenario_data["Loading Plan"])

def run_optimization(session_state, opt_parameters, use_predefined_routes):
    """ Run the optimization using specified optimization parameters
    Args:
        session_state (object): a new SessionState object 
        opt_parameters (dict): Dictionary of true and false values for optimization constraints
        use_predefined_routes (bool): True if route name in Order Eval should be used to separate optimizations
    Returns:
        (bool): True after optimizations complete
    """    

    col_left, col_status, col_right = st.columns([0.5, 2, 0.5])
    t_start = time.perf_counter()
    progress_bar = st.progress(0)
    status = col_status.empty()
    
    predefined_routes = session_state.scenario_data['Deliveries']['Route'].astype(str).str.upper().drop_duplicates().to_list() if use_predefined_routes else None
    num_optimizations = len(predefined_routes) if use_predefined_routes else 1
    dispatches = {}
    fac_dispatches = {}
    route_no = 0

    with st.spinner('Update scenario data ...'):
        session_state.scenario_data["Parameters"] = opt_parameters
        session_state.scenario_data["Solved"] = None
        session_state.scenario_data["SolSummary_DF"] = pd.DataFrame(columns = session_state.scenario_data["SolSummary_DF"].columns)
        session_state.scenario_data["SolDetail_DF"] = pd.DataFrame(columns = session_state.scenario_data["SolDetail_DF"].columns)
        session_state.scenario_data["SolDispatches"] = {}
        session_state.scenario_data["SolFacDispatches"] = {}
        initial_scenario = deepcopy(session_state.scenario_data)
        time.sleep(1)
        progress_bar.progress(10)

    opt_failed = False  
    for i in range(num_optimizations):
        with st.spinner('Converting data for Dynamic Routing Optimizer (DRO) ...'):
            scenario = deepcopy(initial_scenario)
            if use_predefined_routes:
                scenario = preprocessing_for_dro_data(i, scenario, predefined_routes)
            dro_data = to_dro_data(scenario)
            time.sleep(1)
        progress_bar.progress((10 + i*90/(num_optimizations) + .2*90/(num_optimizations))/100)

        with st.spinner('Initialize Dynamic Routing Optimizer (DRO) ...'):
            solver = DRO(psmenv.Env(r"DRO.ini", r"DRO.log"), dro_data['distance'], dro_data['time'], depot_no=dro_data['depot_no'], debug=False)
            if len(dro_data['dist_updates']) > 0: 
                solver.update_edge_dist(dro_data['dist_updates'])
            status.text(f"Status: {len(dro_data['facilities'])} facilities, {len(dro_data['vehicles'])} vehicles")
            time.sleep(1)
        progress_bar.progress((10 + i*90/(num_optimizations) + .5*90/(num_optimizations))/100)

        with st.spinner('Optimizing Dispatches ...'):
            # Presolve to reduce the vehicle pool, but it is yet to show effectiveness. Assess later. 
            # if dro_data['spec'].time_limit > 0:     # Conduct presolve when time_limit is set to positive
            #     solver.solve(dro_data['facilities'], dro_data['vehicles'], dro_data['vf_exclusions'], dro_data['ff_pairs'], dro_data['spec'], presolve=True)
            #     if solver.status == "ROUTING_SUCCESS":
            #         vehicles_new = [dro_data['vehicles'][v_no] for v_no in range(len(dro_data['vehicles'])) if len(solver.solution_routes[v_no])>2]
            #         if len(vehicles_new) > 0: 
            #             status.text(f"Status: Presolve objective: {solver.solution.ObjectiveValue()}, reduce to {len(vehicles_new)} vehicles")
            #         dro_data['vehicles'] = vehicles_new
            #     progress_bar.progress(60)

            solver.solve(dro_data['facilities'], dro_data['vehicles'], dro_data['vf_exclusions'], dro_data['ff_pairs'], dro_data['spec'])
            progress_bar.progress((10 + i*90/(num_optimizations) + .8*90/(num_optimizations))/100)

        if solver.status == "ROUTING_SUCCESS": 
            t_end = time.perf_counter()
            status.text(f"Status: Completed ({solver.num_facs} facilities, {solver.num_veh} vehicles) \nObjective: {solver.solution.ObjectiveValue()} ({t_end - t_start:0.4f} seconds)")
            with st.spinner('Analyze dispatch result from optimization ...'):
                route_detail_df = analyze_dispatch_results(i, solver, session_state, dro_data, initial_scenario, predefined_routes, route_no, fac_dispatches, dispatches)
                progress_bar.progress((10 + i*90/(num_optimizations) + .85*90/(num_optimizations))/100)

            with st.spinner('Process dedicate truck dispatches ...'):
                dedicate_truck_dispatches(session_state, dro_data, scenario, route_detail_df)
                progress_bar.progress((10 + i*90/(num_optimizations) + .9*90/(num_optimizations))/100)

            with st.spinner('Create pick waves ...'):
                create_pick_waves(session_state.scenario_data, dro_data)
                progress_bar.progress((10 + i*90/(num_optimizations) + .95*90/(num_optimizations))/100)

            with st.spinner('Finalize data for review ...'):
                save_scenario(session_state.scenario_file, session_state.scenario_data)

                time.sleep(1)
                progress_bar.progress((10 + i*90/(num_optimizations) + 90/(num_optimizations))/100)
                
        else: 
            st.error("Optimization Failed")
            opt_failed = True
            break
    if use_predefined_routes:
        truck_optimize(session_state, to_dro_data(session_state.scenario_data))
    if not opt_failed:
        st.success("Route optimization completed. Ready to view results.")
    return True

def set_parameters(scenario):
    """ Provides functionality to set the optimization parameters used in the optimization
    Args:
        scenario (object): a new SessionState object 
    Returns:
        opt_parameters (dict): Dictionary of true and false values for optimization constraints
        use_predefined_rotues (bool): True is route name should be used to run separate optimizations
    """ 

    country_config = configparser.ConfigParser()
    country_config.read("./country_cfg.toml") 
    col_left, col_tl, col_par, col_right = st.columns([0.5, .75, 1, 0.5])

    opt_parameters = {"Include Return Leg Cost":True, 
                    "Enforce Volume Capacity":True,
                    "Enforce Weight Capacity":True, 
                    "Enforce Distance Limit":False,
                    "Enforce Transit Time Limit":True,
                    'Enforce Delivery Time Limit':False,
                    'Allow Missed Deliveries':True, 
                    "Optimization Runtime Limit":60, 
                    "Adjust Transit Time by Speed":False}

    par_list = ["Optimization Runtime Limit", "Include Return Leg Cost", 
        "Enforce Volume Capacity", "Enforce Weight Capacity", 
        "Enforce Distance Limit","Enforce Transit Time Limit",
        'Enforce Delivery Time Limit','Allow Missed Deliveries', 
        "Adjust Transit Time by Speed"]

    # Optimization Parameter Selections for App_Optimize
    for par in par_list:
        if country_config.get("dro_specs", par).title()=="True":
            default_par_val = opt_parameters[par] if par not in scenario["Parameters"] else scenario["Parameters"][par]
            if par == "Optimization Runtime Limit": 
                opt_parameters[par] = int(col_tl.text_input(par, value=default_par_val, help=DROSpecDesc[par]))
            else:    
                opt_parameters[par] = bool(col_tl.checkbox(par, value=default_par_val, help=DROSpecDesc[par], key = par))
    
    # Checkbox for predefined routes (to be expanded for the sub problem solve)
    use_predefined_routes = False
    if country_config.get("baseline", "predefined_routes").title() == "True":
        use_predefined_routes = bool(col_par.checkbox("Use Predefined Routes", value=False, 
                            help="Perform Route Optimization Using Route Numbers in Scenario File", key = "Use Predefined Routes"))
    return opt_parameters, use_predefined_routes


def preprocessing_for_dro_data(i, scenario, predefined_routes):
    """ Preprocessing necessary if predefined routes are going to be run
    Args:
        i (int): index corresponding to a predefined route
        scenario (object): a new session state object
        predefined_routes (list): list of the applicable route names
    Returns:
        scenario (object): subset of the original scenario provided
    """    

    # Preparing the scenario based on the predefined route
    subsetted_routes = [True if str(r['orig_route']) == 'nan' or predefined_routes[i] in str(r['orig_route']).upper() else False for _, r in scenario["Facility_DF"].iterrows()]
    subsetted_indexes = [i for i in range(len(subsetted_routes)) if subsetted_routes[i]]
    scenario["Facility_DF"] = scenario["Facility_DF"][subsetted_routes] 
    scenario['Deliveries'] = scenario['Deliveries'][scenario['Deliveries']['Route'].astype(str).str.upper() == predefined_routes[i]]
    scenario['Distance_DF'].columns = range(scenario['Distance_DF'].columns.size)
    scenario['Time_DF'].columns = range(scenario['Time_DF'].columns.size)
    scenario['Distance_DF'] = scenario["Distance_DF"][subsetted_routes]
    scenario['Distance_DF'] = scenario["Distance_DF"][subsetted_indexes]
    scenario['Time_DF'] = scenario["Time_DF"][subsetted_routes]
    scenario['Time_DF'] = scenario["Time_DF"][subsetted_indexes]
    return scenario

def analyze_dispatch_results(i, solver, session_state, dro_data, initial_scenario, predefined_routes, route_no, fac_dispatches, dispatches):
    """ Aggregating solver solution into a dataframe 
    Args:
        i (int): index corresponding to a predefined route
        solver (object): Google OR Tools solver
        session_state (object): a new session state object
        dro_data (dict): dictionary containg all optimization data
        initial_scenario: a new session state object
        predefined_routes (list): list of the applicable route names or None
        route_no (int): route number corresponding to route
        fac_dispatches (dict): dictionary containing all the routes a specific facility is on
        dispatches (dict): dictionary containing truck specific information for a dispatch
    Returns:
        route_detail_df (Dataframe): Dataframe of route specific data
    """ 
    
    
    route_detail = []
    route_summary = []
    num_routes = 0
    for v_no in range(solver.num_veh): 
        if len(solver.solution_routes[v_no]) <= 2: 
            continue

        # Error Handling and Fleet Updates for Predefined Routes
        if predefined_routes != None:
            if num_routes >= 1:
                st.markdown(f'Optimization Split Predefined Route into More than 1 route: {predefined_routes[i]}')
            truck_type = dro_data['vehicles'][v_no].type
            initial_scenario['Fleet_DF'].loc[truck_type, "max_routes"] = initial_scenario['Fleet_DF'].loc[truck_type, "max_routes"] - 1
            if initial_scenario['Fleet_DF'].loc[truck_type, "max_routes"] == 0:
                st.markdown(f"There are 0 {truck_type}s remaining")
        
        # Getting the Route informaiton from Solver
        num_routes += 1
        route_no = max(route_no +1, len(session_state.scenario_data["SolSummary_DF"])+1)
        route_summary, route_detail, fac_dispatches, dispatches = get_route_details(v_no, solver, session_state, dro_data, route_no,
                                                                                    route_summary, route_detail,fac_dispatches, dispatches)

    # Generating  "SolSummary_DF" and  "SolDetail_DF" for scenario data
    route_detail_df = pd.DataFrame(route_detail, columns=['route', 'stop_no', 'facility_id', 'distance', 'time', 'fuel_usage', 'cost'])
    route_summary_df = pd.DataFrame(route_summary, columns=['route', 'truck_type', 'path', 'num_stops'])

    route_detail_df = pd.merge(route_detail_df, session_state.scenario_data['Facility_DF'].reset_index(), how='left', on='facility_id')
    route_sum = route_detail_df.groupby("route").agg({"vol":"sum", "weight":"sum", "distance":"sum", "time":"sum", "fuel_usage":"sum", "cost":"sum"}).reset_index()
    
    route_summary_df = pd.merge(route_summary_df, route_sum, how='left', on='route')
    route_summary_df = pd.merge(route_summary_df, session_state.scenario_data['Fleet_DF'].reset_index()[["truck_type", 'vol_cap', 'weight_cap']], how='left', on='truck_type')
    route_summary_df["vol_utilization"] = route_summary_df['vol']/route_summary_df['vol_cap']
    route_summary_df["weight_utilization"] = route_summary_df['weight']/route_summary_df['weight_cap']
    route_summary_df["source"] = 'VRP Solver'

    # Updating the Session State
    session_state.scenario_data["Solved"] = datetime.now()
    session_state.scenario_data["SolSummary_DF"] = session_state.scenario_data["SolSummary_DF"].append(route_summary_df, ignore_index=True)
    session_state.scenario_data["SolDetail_DF"] = session_state.scenario_data["SolDetail_DF"].append(route_detail_df, ignore_index=True)
    session_state.scenario_data["SolDispatches"] = dispatches
    session_state.scenario_data["SolFacDispatches"] = fac_dispatches
    
    # returning route_detail_df since needed for dedicate_truck_dispatches
    # Note: session_state.scenario_data['SolDetail_DF'] does not work in dedicate_truck_dispatches
    return route_detail_df

def get_route_details(v_no, solver, session_state, dro_data, route_no, route_summary, route_detail,fac_dispatches, dispatches):
    """ Generating Critical Information to be used in later calculations for the route
    Args:
        v_no (int): vehicle index for route
        solver (object): Google OR Tools solver
        session_state (object): a new session state object
        dro_data (dict): dictionary containg all optimization data
        route_no (int): index corresponding to the route
        predefined_routes (list): list of the applicable route names or None
        route_no (int): route number corresponding to route
        route_summary (list): list of strings containing all facilities on a specific route
        route_detail (list): list of tuples containing stats pertaing to each stop on a route
        fac_dispatches (dict): dictionary containing all the routes a specific facility is on
        dispatches (dict): dictionary containing truck specific information for a dispatch
    Returns:
        route_summary (list): list of strings containing all facilities on a specific route
        route_detail (list): list of tuples containing stats pertaing to each stop on a route
        fac_dispatches (dict): dictionary containing all the routes a specific facility is on
        dispatches (dict): dictionary containing truck specific information for a dispatch
    """ 
    
    
    # Generating Critical Information to be used in later calculations for the route
    dispatch_id = f"Dispatch {route_no:03}"
    route = solver.solution_routes[v_no] if dro_data['spec'].return_cost else solver.solution_routes[v_no][:-1]
    truck_type = dro_data['vehicles'][v_no].type
    truck_vol_cap = dro_data['vehicles'][v_no].vol_cap/SCALE['VOL']
    truck_weight_cap = dro_data['vehicles'][v_no].weight_cap/SCALE['WGT']
    truck_ltr_per_km = session_state.scenario_data['Fleet_DF'].loc[truck_type, "liters_per_km"]
    truck_base_cost = session_state.scenario_data['Fleet_DF'].loc[truck_type, "base_cost"]
    truck_fuel_price = session_state.scenario_data['Fleet_DF'].loc[truck_type, "fuel_price"]
    truck_fixed_cost = 0
    try:
        truck_fixed_cost = session_state.scenario_data['Fleet_DF'].loc[truck_type, "fixed_cost"]
    except:
        pass
    stops = []
    for stop_no, f_no in enumerate(route): 
        # Calculating the distance, time, fuel and cost by stop
        fac_id = dro_data["facilities"][f_no].id
        fac_name = session_state.scenario_data['Facility_DF'].loc[fac_id, "facility"]
        del_dist = 0 if stop_no == 0 else round(solver.get_vehicle_distance(route[stop_no-1], f_no, v_no)/SCALE['DIST'], 1)
        del_time = 0 if stop_no == 0 else round(solver.get_vehicle_delivery_time(route[stop_no-1], f_no, v_no)/SCALE['TIME'], 1)
        fixed_cost = truck_fixed_cost if stop_no == 0 else 0
        fuel = round(del_dist*truck_ltr_per_km, 1)
        cost = round(del_dist*(truck_base_cost+truck_ltr_per_km*truck_fuel_price)+fixed_cost, 2)
        
        route_detail.append((dispatch_id, stop_no, fac_id, del_dist, del_time, fuel, cost))
        stops.append(fac_name)

        if fac_id not in fac_dispatches: 
            fac_dispatches[fac_id] = []
        fac_dispatches[fac_id].append(dispatch_id)

    #taking route details and creating route_summary info
    nstops = len(route)- (2 if stops[0] == stops[-1] else 1)
    route_summary.append((dispatch_id, truck_type, " -> ".join(stops), nstops))
    dispatches[dispatch_id] = (truck_type, truck_vol_cap, truck_weight_cap)
    return route_summary, route_detail, fac_dispatches, dispatches

def dedicate_truck_dispatches(session_state, dro_data, scenario, route_detail_df):
    """ Pulling the missed deliveries and solves separately if possible
    Args:
        session_state (object): a new session state object
        dro_data (dict): dictionary containg all optimization data
        scenario (object): a new session state object
        route_detail_df (Dataframe): Dataframe of route specific data
    """ 
    
    missed_deliveries_df = pd.merge(scenario['Facility_DF'].reset_index(), route_detail_df[["facility_id", "route"]], how='left', on='facility_id')
    missed_deliveries_df = missed_deliveries_df[pd.isna(missed_deliveries_df['route']) & (missed_deliveries_df['type'] != 'Warehouse')].drop(columns="route")
    missed_facility_ids = []
    for facility_id in missed_deliveries_df['facility_id'].to_list(): 
        if not optimize_dedicated_delivery(session_state.scenario_data, dro_data, str(facility_id)):
            missed_facility_ids.append(facility_id)

    session_state.scenario_data["SolMiss_DF"] = missed_deliveries_df[missed_deliveries_df['facility_id'].isin(missed_facility_ids)]
    session_state.scenario_ver_no += 1


if __name__ == "__main__":

    from PIL import Image
    import SessionState
    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    st.sidebar.title("Zambia Routing Tool")

    st.image(Image.open('./images/app_banner.jpg'), use_column_width=True)
    st.markdown('<style>' + open('icons.css').read() + '</style>', unsafe_allow_html=True)
    
    session_state = SessionState.get(ref_file = "Reference Tables.xlsx", 
                                     user_file = "", 
                                     scenario_file =  r"./filestore/Scenario BUX72X_v3.xlsx", 
                                     scenario_data = None,
                                     scenario_ver_no = 0,) 
    session_state.scenario_data = read_scenario(session_state.scenario_file, session_state.scenario_ver_no) 
    app_optimize(session_state)
                