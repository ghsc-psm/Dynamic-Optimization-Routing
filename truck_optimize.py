""" Handles optimization after the initial optimization to ensure the most optimal truck is assigned."""

from requests import session
import streamlit as st
import pandas as pd
import time
from math import ceil
from datetime import datetime
import configparser
import numpy as np
from copy import deepcopy

from ortools.linear_solver import pywraplp

from DRO import DRO, DROFacility, DROVehicle, DROSpec, DROSpecDesc, DROINF
from psm import env as psmenv
from scenario import initialize_scenario, read_scenario, save_scenario
import app_optimize as ao


SCALE = {'VOL':100, 
         'WGT':10, 
         'TIME':60,
         'DIST':10,
         'VALUE':10}

import time

def route_truck_set(scenario, dro_data):
    """ Creates set of route truck tuples that are feasible based on constraints
    Args:
        scenario (object): a new SessionState object
        dro_data (dict): dictionary representation of scenario above
    Returns:
        truck_route_combos (set): set of route truck tuples
    """
    truck_route_combos = []
    count = 0
    for i, r in scenario["SolSummary_DF"].iterrows():
        for j, v in enumerate(dro_data['vehicles']):
            route_feasible = True
            volume_check = v.vol_cap/SCALE['VOL'] < r['vol'] 
            weight_check = v.weight_cap/SCALE['WGT'] < r['weight']
            dist_check = v.dist_cap/SCALE['DIST'] < r['distance']
            if volume_check or weight_check or dist_check:
                count +=1
                continue
            #st.table(scenario["SolDetail_DF"])
            route_facs = scenario["SolDetail_DF"][scenario["SolDetail_DF"]['route'] == r['route']]#.drop_duplicates()
            facs_nums = list(dro_data['facility_no'].keys())
            for (n, fac_id) in dro_data['vf_exclusions']:
                #st.markdown((route_facs['facility_id'] == facs_nums[fac_id]).any())
                if n == j and (route_facs['facility_id'] == facs_nums[fac_id]).any():
                    route_feasible = False
                    count +=1
                    break
                #st.markdown(f"{n} and  {fac_id}")
                #st.markdown(dro_data['facility_no'][list(dro_data['facility_no'].keys())[fac_id]] == fac_id)
            if route_feasible:
                truck_route_combos.append((r['route'], j))
    #st.markdown(len(truck_route_combos)) 
    return truck_route_combos
                     
            
def get_optimal_truck_pairing(scenario, dro_data, truck_route_combos):
    """ Performs assignment optimization leveraging Google OR Tools
    Args:
        scenario (object): a new SessionState object
        dro_data (dict): dictionary representation of scenario abov
        truck_route_combos (set): set of route truck tuples
    Returns: 
        solution_dict (dict): Dictionary of route information with optimal truck
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    #st.markdown(truck_route_combos)
    truck_route_combos = set(truck_route_combos)
    trucks = np.unique([truck for (_, truck) in truck_route_combos])
    routes =np.unique([route for (route, _) in truck_route_combos])
    num_trucks = len(trucks)
    num_routes = len(routes)

    # Route Truck Combinations that are allowed
    x = {}
    for combo in truck_route_combos:
        x[combo] = solver.IntVar(0, 1, '')

    # Dummy Variables for Routes if truck not selected
    y= {}
    for r in routes:
            y[r] = solver.IntVar(0, 1, '')
    
    # Each truck used 1 or 0 times
    # This should be changed to be based on volume and based on number of days allowable
    for t in trucks:
        solver.Add(solver.Sum([x[(r,t)] for r in routes if (r,t) in truck_route_combos]) <= 1)

    # Each route must be assigned to a 
    # Otherwise dummy variable is used and assigned 999999 cost
    for r in routes:
        solver.Add(solver.Sum([x[(r,t)] for t in trucks if (r,t) in truck_route_combos]) + y[r] == 1)
    
    # defing cost as done in DRO Optimization
    def get_cost(r,t):
        truck_type = dro_data['vehicles'][t].type
        del_dist = np.sum(scenario["SolDetail_DF"][scenario["SolDetail_DF"]['route'] == r].reset_index()['distance'])
        truck_ltr_per_km = scenario['Fleet_DF'].loc[truck_type, "liters_per_km"]
        truck_base_cost = scenario['Fleet_DF'].loc[truck_type, "base_cost"]
        truck_fuel_price = scenario['Fleet_DF'].loc[truck_type, "fuel_price"]
        truck_fixed_cost = 0
        try:
            truck_fixed_cost = scenario['Fleet_DF'].loc[truck_type, "fixed_cost"]
        except:
            pass

        cost = round(del_dist*(truck_base_cost+truck_ltr_per_km*truck_fuel_price)+ truck_fixed_cost, 2)
        return cost

    # Generating Cost for the Assignments to minimize
    objective_terms = []
    for r in routes:
        objective_terms.append(y[r]*999999)
        for t in trucks:
            if (r,t) in truck_route_combos:
                objective_terms.append(get_cost(r,t) * x[(r,t)])

    solver.Minimize(solver.Sum(objective_terms))

    # Solving Optimizattion
    status = solver.Solve()
    solution_dict = {}
    # Resulting Output
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        total_cost = 0
        for r in routes:
            if y[r].solution_value() > 0.5:
                #st.markdown(f"Route: {r} not assigned - Cost: 999999")
                total_cost += 999999
                continue
            for t in trucks:
                if (r,t) in truck_route_combos:
                    if x[(r,t)].solution_value() > 0.5:
                        total_cost += get_cost(r,t)
                        truck_type = dro_data['vehicles'][t].type
                        solution_dict[r] = truck_type
                        #st.markdown(f"Route: {r} using Vehicle {t} - Cost: {get_cost(r,t)}")
        return solution_dict
    else:
        print('No solution found.')
        return None

def truck_optimize(session_state, dro_data):
    """ Performs truck optimization and updates sessionstate information
    Args:
        session_state (object): a new SessionState object
        dro_data (dict): dictionary representation of scenario above
    
    """
    truck_route_combos = route_truck_set(session_state.scenario_data, dro_data)
    route_truck_dict = get_optimal_truck_pairing(session_state.scenario_data, dro_data, truck_route_combos)
    if len(session_state.scenario_data['SolSummary_DF']) > len(route_truck_dict):
        st.markdown("Truck Optimization Not Feasible.")
        return -1
    for i,r in session_state.scenario_data["SolDetail_DF"].iterrows():
        truck_type = route_truck_dict[r['route']]
        truck_ltr_per_km = session_state.scenario_data['Fleet_DF'].loc[truck_type, "liters_per_km"]
        truck_base_cost = session_state.scenario_data['Fleet_DF'].loc[truck_type, "base_cost"]
        truck_fuel_price = session_state.scenario_data['Fleet_DF'].loc[truck_type, "fuel_price"]
        del_dist  = r['distance']
        truck_fixed_cost = 0
        try:
            truck_fixed_cost = session_state.scenario_data['Fleet_DF'].loc[truck_type, "fixed_cost"]
        except:
            pass
        fixed_cost = truck_fixed_cost if r['stop_no'] == 0 else 0
        r['fuel_usage'] = round(del_dist*truck_ltr_per_km, 1)
        r['cost'] = round(del_dist*(truck_base_cost+truck_ltr_per_km*truck_fuel_price)+fixed_cost, 2)

    #st.markdown(session_state['Fleet_DF'].index)
    dispatches = {}
    for i, r in session_state.scenario_data["SolSummary_DF"].iterrows():
        r['truck_type'] = route_truck_dict[r['route']]
        r['fuel_usage'] = np.sum(session_state.scenario_data["SolDetail_DF"][session_state.scenario_data["SolDetail_DF"]['route'] == r['route']].fuel_usage)
        r['cost'] = np.sum(session_state.scenario_data["SolDetail_DF"][session_state.scenario_data["SolDetail_DF"]['route'] == r['route']].cost)
        r['vol_cap'] = session_state.scenario_data['Fleet_DF'].loc[route_truck_dict[r['route']], 'vol_cap']
        r['weight_cap'] = session_state.scenario_data['Fleet_DF'].loc[route_truck_dict[r['route']], 'weight_cap']
        r['vol_utilization'] = r['vol']/r['vol_cap']
        r['weight_utilization'] = r['weight']/r['weight_cap'] 
        dispatches[r['route']] = (r['truck_type'], r['vol_cap'], r['weight_cap'])
    session_state.scenario_data["SolDispatches"] = dispatches
    st.success("Truck Optimization Complete")

if __name__ == "__main__":

    from PIL import Image
    import SessionState

    scenario_file_path = r"filestore\Scenario 4GS7HX.xlsx"

    session_state = SessionState.get(ref_file = "Reference Tables.xlsx", 
                                     user_file = "", 
                                     scenario_file =  scenario_file_path, 
                                     scenario_data = None,
                                     scenario_ver_no = 0,) 
    st.markdown("# Running Truck Optimization")
    st.markdown(f"File Path: {scenario_file_path}")
    session_state.scenario_data = read_scenario(session_state.scenario_file, session_state.scenario_ver_no)
    truck_optimize(session_state, ao.to_dro_data(session_state.scenario_data))
    save_scenario(f"{scenario_file_path[:-5]}-Truck Optimization.xlsx",session_state.scenario_data)
    st.success("File Successfully Saved")