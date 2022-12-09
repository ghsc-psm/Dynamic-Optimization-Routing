import pandas as pd
import streamlit as st
from datetime import datetime
import random
import string

@st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
def read_scenario(scenario_file, ver=0):
    scenario_data = {}
    scenario_data['Loading Plan'] = pd.read_excel(scenario_file, sheet_name="Loading Plan")
    scenario_data['Facility_DF'] = pd.read_excel(scenario_file, sheet_name="Facilities")
    scenario_data['Facility_DF']["facility_id"] = scenario_data['Facility_DF']["facility_id"].apply(lambda c: str(c)[:8])
    facility_ids = set(scenario_data['Facility_DF']["facility_id"].to_list())
    scenario_data['Facility_DF']=scenario_data['Facility_DF'].set_index('facility_id')

    scenario_data['Fleet_DF'] = pd.read_excel(scenario_file, sheet_name="Fleet").set_index('truck_type')
    scenario_data['Distance_DF'] = pd.read_excel(scenario_file, sheet_name="Distance")
    scenario_data['Time_DF'] = pd.read_excel(scenario_file, sheet_name="Time")
    scenario_data["SolSummary_DF"] = pd.read_excel(scenario_file, sheet_name = "Solution Summary")
    scenario_data["SolDetail_DF"] = pd.read_excel(scenario_file, sheet_name = "Solution Detail")
    scenario_data["SolDetail_DF"]['facility_id'] = scenario_data["SolDetail_DF"]['facility_id'].apply(lambda c: str(c)[:8])
    scenario_data["SolMiss_DF"] = pd.read_excel(scenario_file, sheet_name = "Solution Miss")
    scenario_data["SolMiss_DF"]['facility_id'] = scenario_data["SolMiss_DF"]['facility_id'].apply(lambda c: str(c)[:8])

    scenario_data['Parameters'] = {r['parameter']:r['value'] 
                                        for _, r in pd.read_excel(scenario_file, sheet_name="Parameters").iterrows()}
    scenario_data['Distance Adj'] = {(str(r['from_facility_id']), str(r['to_facility_id'])): r['distance_adj'] 
                                        for _, r in pd.read_excel(scenario_file, sheet_name="Distance Adj").iterrows()}
    
    scenario_data['Facility Groups'] = {}
    for _, r in pd.read_excel(scenario_file, sheet_name="Facility Groups").iterrows(): 
        if str(r['facility_id']) in facility_ids: 
            if r['group_id'] not in scenario_data['Facility Groups']: 
                scenario_data['Facility Groups'][r['group_id']] = set()
            scenario_data['Facility Groups'][r['group_id']].add(str(r['facility_id']))

    scenario_data['Vehicle Exclusion'] = {}
    for _, r in pd.read_excel(scenario_file, sheet_name="Fleet Exclusions").iterrows():
        if str(r['facility_id']) in facility_ids: 
            if r['truck_type'] not in scenario_data['Vehicle Exclusion']: 
                scenario_data['Vehicle Exclusion'][r['truck_type']] = set()
            scenario_data['Vehicle Exclusion'][r['truck_type']].add(str(r['facility_id']))

    # Read in meta data
    meta = pd.read_excel(scenario_file, sheet_name = "Metadata").set_index('attribute')
    scenario_data['Created'] = pd.to_datetime(meta.loc['Date created']['value'])
    scenario_data['Modified'] = pd.to_datetime(meta.loc['Latest modification date']['value'])
    scenario_data['Solved'] = pd.to_datetime(meta.loc['Latest solve date']['value'])
    scenario_data['Created By'] = meta.loc['Created By']['value']
    scenario_data['Modified By'] = meta.loc['Modified By']['value']
    scenario_data['Scenario'] = meta.loc['Scenario']['value']
    scenario_data['Version'] = meta.loc['Version']['value']

    if "Deliveries" in pd.ExcelFile(scenario_file).sheet_names: 
        scenario_data['Deliveries'] = pd.read_excel(scenario_file, sheet_name="Deliveries")
        scenario_data['Deliveries']['Customer ID'] = scenario_data['Deliveries']['Customer ID'].apply(lambda c: str(c)[:8])
        scenario_data['Deliveries']['Hub Code'] = scenario_data['Deliveries']['Hub Code'].apply(lambda c: str(c)[:8])
        scenario_data['Deliveries']['District Health Office Code'] = scenario_data['Deliveries']['District Health Office Code'].apply(lambda c: str(c)[:8])
        scenario_data['Deliveries']['Dispatch Destination Code'] = scenario_data['Deliveries']['Dispatch Destination Code'].apply(lambda c: str(c)[:8])

        scenario_data['Order Info'] = pd.read_excel(scenario_file, sheet_name="Order Info")
        scenario_data['Order Info']['Customer ID'] = scenario_data['Order Info']['Customer ID'].apply(lambda c: str(c)[:8])

        scenario_data['Order Details'] = pd.read_excel(scenario_file, sheet_name="Order Details")
        scenario_data['Order Details']['Customer ID'] = scenario_data['Order Details']['Customer ID'].apply(lambda c: str(c)[:8])

    return scenario_data


def save_scenario(scenario_file, scenario_data):
    writer = pd.ExcelWriter(scenario_file, engine = "openpyxl")
    scenario_data["Loading Plan"].to_excel(writer, sheet_name = "Loading Plan", index=False)
    scenario_data["Facility_DF"].to_excel(writer, sheet_name = "Facilities")
    scenario_data["Distance_DF"].to_excel(writer, sheet_name = "Distance", index=False)
    scenario_data["Time_DF"].to_excel(writer, sheet_name = "Time", index=False)
    scenario_data["Fleet_DF"].to_excel(writer, sheet_name = "Fleet")

    vehicle_exclusion_DF = pd.DataFrame(columns = ['truck_type', 'facility_id'])
    if len(scenario_data["Vehicle Exclusion"]) > 0:
        vehicle_exclusion_DF = pd.DataFrame.from_dict(scenario_data["Vehicle Exclusion"], orient="index").reset_index().melt(id_vars="index", value_name="facility_id")
        vehicle_exclusion_DF = vehicle_exclusion_DF[["index", "facility_id"]].rename(columns = {"index": "truck_type"}).dropna(axis=0, how="any")    
    vehicle_exclusion_DF.to_excel(writer, sheet_name = "Fleet Exclusions", index=False)
    
    facility_groups_DF = pd.DataFrame(columns = ["group_id", "facility_id"])
    if len(scenario_data["Distance Adj"]) > 0:
        facility_groups_DF = pd.DataFrame.from_dict(scenario_data["Facility Groups"], orient="index").reset_index().melt(id_vars="index", value_name="facility_id").sort_values("index")
        facility_groups_DF = facility_groups_DF[["index", "facility_id"]].rename(columns = {"index": "group_id"}).dropna(axis=0, how="any")
    facility_groups_DF.to_excel(writer, sheet_name = "Facility Groups", index=False)

    dist_adj_DF = pd.DataFrame([], columns=["from_facility_id", "to_facility_id", "distance_adj"])
    if len(scenario_data["Distance Adj"]) > 0: 
        dist_adj_DF = pd.DataFrame.from_dict(scenario_data["Distance Adj"], orient="index").reset_index()
        dist_adj_DF[["from_facility_id", "to_facility_id"]] = pd.DataFrame(dist_adj_DF["index"].tolist(), index=dist_adj_DF.index)
        dist_adj_DF = dist_adj_DF.rename(columns = {0: "distance_adj"})    
    dist_adj_DF[["from_facility_id", "to_facility_id", "distance_adj"]].to_excel(writer, sheet_name = "Distance Adj", index=False)

    parameters_DF = pd.DataFrame.from_dict(scenario_data["Parameters"], orient="index").reset_index().rename(columns = {"index": "parameter", 0: "value"})
    parameters_DF.to_excel(writer, sheet_name = "Parameters", index=False)

    meta_attributes = ["Date created", "Latest modification date", "Latest solve date", "Created By", "Modified By", "Scenario", "Version"]
    meta_keys = ["Created", "Modified", "Solved", "Created By", "Modified By", "Scenario", "Version"]
    meta_dict = {meta_attributes[i] : scenario_data[meta_keys[i]] for i in range(0, len(meta_keys))}
    meta_DF = pd.DataFrame.from_dict(meta_dict, orient="index").reset_index().rename(columns = {"index": "attribute", 0: "value"})
    meta_DF.to_excel(writer, sheet_name = "Metadata", index=False)

    if "Deliveries" in scenario_data: 
        scenario_data["Deliveries"].to_excel(writer, sheet_name = "Deliveries", index=False)
        scenario_data["Order Info"].to_excel(writer, sheet_name = "Order Info", index=False)
        scenario_data["Order Details"].to_excel(writer, sheet_name = "Order Details", index=False)

    scenario_data["SolSummary_DF"].to_excel(writer, sheet_name = "Solution Summary", index=False)
    scenario_data["SolDetail_DF"].to_excel(writer, sheet_name = "Solution Detail", index=False)
    scenario_data["SolMiss_DF"].to_excel(writer, sheet_name = "Solution Miss", index=False)

    writer.save()
    

def create_download_dict(scenario_data):
    download_df_dict = {}
    download_df_dict["Loading Plan"] = scenario_data["Loading Plan"]
    download_df_dict["Facilities"] = scenario_data["Facility_DF"].reset_index()
    download_df_dict["Distance"] = scenario_data["Distance_DF"]
    download_df_dict["Time"] = scenario_data["Time_DF"]
    download_df_dict["Fleet"] = scenario_data["Fleet_DF"].reset_index()

    vehicle_exclusion_DF = pd.DataFrame(columns = ['truck_type', 'facility_id'])
    if len(scenario_data["Vehicle Exclusion"]) > 0:
        vehicle_exclusion_DF = pd.DataFrame.from_dict(scenario_data["Vehicle Exclusion"], orient="index").reset_index().melt(id_vars="index", value_name="facility_id")
        vehicle_exclusion_DF = vehicle_exclusion_DF[["index", "facility_id"]].rename(columns = {"index": "truck_type"}).dropna(axis=0, how="any")    
    download_df_dict["Fleet Exclusions"] = vehicle_exclusion_DF

    facility_groups_DF = pd.DataFrame(columns = ["group_id", "facility_id"])
    if len(scenario_data["Distance Adj"]) > 0:
        facility_groups_DF = pd.DataFrame.from_dict(scenario_data["Facility Groups"], orient="index").reset_index().melt(id_vars="index", value_name="facility_id").sort_values("index")
        facility_groups_DF = facility_groups_DF[["index", "facility_id"]].rename(columns = {"index": "group_id"}).dropna(axis=0, how="any")
    download_df_dict["Facility Groups"] = facility_groups_DF

    dist_adj_DF = pd.DataFrame([], columns=["from_facility_id", "to_facility_id", "distance_adj"])
    if len(scenario_data["Distance Adj"]) > 0: 
        dist_adj_DF = pd.DataFrame.from_dict(scenario_data["Distance Adj"], orient="index").reset_index()
        dist_adj_DF[["from_facility_id", "to_facility_id"]] = pd.DataFrame(dist_adj_DF["index"].tolist(), index=dist_adj_DF.index)
        dist_adj_DF = dist_adj_DF.rename(columns = {0: "distance_adj"})    
    download_df_dict["Distance Adj"] = dist_adj_DF[["from_facility_id", "to_facility_id", "distance_adj"]]

    download_df_dict["Parameters"] = pd.DataFrame.from_dict(scenario_data["Parameters"], orient="index").reset_index().rename(columns = {"index": "parameter", 0: "value"})

    meta_attributes = ["Date created", "Latest modification date", "Latest solve date", "Created By", "Modified By", "Scenario", "Version"]
    meta_keys = ["Created", "Modified", "Solved", "Created By", "Modified By", "Scenario", "Version"]
    meta_dict = {meta_attributes[i] : scenario_data[meta_keys[i]] for i in range(0, len(meta_keys))}
    meta_DF = pd.DataFrame.from_dict(meta_dict, orient="index").reset_index().rename(columns = {"index": "attribute", 0: "value"})
    download_df_dict["Metadata"] = meta_DF

    if "Deliveries" in scenario_data: 
        download_df_dict["Deliveries"] = scenario_data["Deliveries"]
        download_df_dict["Order Info"] = scenario_data["Order Info"]
        download_df_dict["Order Details"] = scenario_data["Order Details"]
    
    download_df_dict["Solution Summary"] = scenario_data["SolSummary_DF"]
    download_df_dict["Solution Detail"] = scenario_data["SolDetail_DF"]
    download_df_dict["Solution Miss"] = scenario_data["SolMiss_DF"]

    return download_df_dict


def initialize_scenario(session_state, facilities):
    # Contruct Scenario data object
    # Save scenario to file
    # return scenario data object

    # session_state.ref_file, session_state.user_file, session_state.warehouse, 

    scenario_data = {}

    """ Read in reference data """
    facility_DF = pd.read_excel(session_state.ref_file, sheet_name = "Facility")
    facility_columns = facility_DF.columns.tolist()
    facility_DF["facility_id"] = facility_DF["facility_id"].apply(lambda c: str(c)[:8])
    distance_DF = pd.read_excel(session_state.ref_file, sheet_name = "Distance")
    time_DF = pd.read_excel(session_state.ref_file, sheet_name = "Time")
    fleet_DF = pd.read_excel(session_state.ref_file, sheet_name = "Fleet")
    parameter_DF = pd.read_excel(session_state.ref_file, sheet_name = "Parameters")
    fleet_columns = fleet_DF.columns

    """ Read in raw order data and transform to usable format """

    if "Delivery" in pd.ExcelFile(session_state.user_file).sheet_names:   # Order evaluation file
        scenario_data["Deliveries"] = pd.read_excel(session_state.user_file, sheet_name='Delivery')
        scenario_data["Deliveries"] = scenario_data["Deliveries"][scenario_data["Deliveries"]['Dispatch Destination'] != "Self-collect"]
        scenario_data['Deliveries']['Customer ID'] = scenario_data['Deliveries']['Customer ID'].apply(lambda c: str(c)[:8])
        scenario_data['Deliveries']['Hub Code'] = scenario_data['Deliveries']['Hub Code'].apply(lambda c: str(c)[:8])
        scenario_data['Deliveries']['District Health Office Code'] = scenario_data['Deliveries']['District Health Office Code'].apply(lambda c: str(c)[:8])
        scenario_data['Deliveries']['Dispatch Destination Code'] = scenario_data['Deliveries']['Dispatch Destination Code'].apply(lambda c: str(c)[:8])

        scenario_data["Order Info"] = pd.read_excel(session_state.user_file, sheet_name='Order Info')
        scenario_data['Order Info']['Customer ID'] = scenario_data['Order Info']['Customer ID'].apply(lambda c: str(c)[:8])

        scenario_data["Order Details"] = pd.read_excel(session_state.user_file, sheet_name='Order Details')
        scenario_data['Order Details']['Customer ID'] = scenario_data['Order Details']['Customer ID'].apply(lambda c: str(c)[:8])

        dispatch_df = scenario_data["Deliveries"].groupby(['Dispatch Destination']).agg({'Loading Weight':'sum', 'Loading Volume':'sum'}).reset_index()
        dispatch_df.columns = ['facility', 'weight', 'vol']
        dispatch_df['orig_route'] = None
        for i, r in dispatch_df.iterrows():
            dispatch_df.at[i,'orig_route'] = scenario_data["Deliveries"][scenario_data["Deliveries"]['Dispatch Destination'] == r['facility']].Route.drop_duplicates().to_list()
        facility_DF = facility_DF.merge(dispatch_df, on='facility', how="left")
        facility_DF = facility_DF[facility_DF.facility.isin(facilities + [session_state.warehouse])]
        facility_DF.to_csv('./data/tester.csv')
    else: 
        user_file = pd.read_excel(session_state.user_file)
        user_file["Facility ID"] = user_file["Facility ID"].apply(lambda c: str(c)[:8])
        facility_DF = facility_DF.merge(user_file, right_on = "Facility ID", left_on = "facility_id", how="left").rename(columns = {"Volume (cubic meters)" : "vol"})
        facility_DF["weight"] = facility_DF["vol"]*100
        facility_DF = facility_DF[facility_DF.facility.isin(facilities + [session_state.warehouse])]

    """ Subset distance and time matrix based on order data"""
    fac_non_empty = facility_DF.index
    distance_DF = distance_DF.iloc[fac_non_empty]
    distance_DF = distance_DF[distance_DF.columns[fac_non_empty]]
    time_DF = time_DF.iloc[fac_non_empty]
    time_DF = time_DF[time_DF.columns[fac_non_empty]]
    distance_DF = distance_DF.reset_index(drop=True)
    time_DF = time_DF.reset_index(drop=True)

    facility_ids = set(facility_DF['facility_id'].to_list())
    facility_DF = facility_DF[facility_columns + ["vol", "weight", "orig_route"]]
    facility_DF = facility_DF.set_index("facility_id")
    
    fleet_DF = fleet_DF[fleet_DF.warehouse == session_state.warehouse]
    fleet_DF["available"] = True
    fleet_DF = fleet_DF[["available"] + fleet_columns[fleet_columns != "warehouse"].tolist()]
    fleet_DF = fleet_DF.set_index("truck_type")

    scenario_data["Facility_DF"] = facility_DF
    scenario_data["Distance_DF"] = distance_DF
    scenario_data["Time_DF"] = time_DF
    scenario_data["Fleet_DF"] = fleet_DF
    
    vehicle_exclusion = pd.read_excel(session_state.ref_file, sheet_name = "Fleet Exclusions")
    vehicle_exclusion = vehicle_exclusion[((vehicle_exclusion.warehouse == session_state.warehouse) | (vehicle_exclusion.warehouse.isna()))]
    scenario_data['Vehicle Exclusion'] = {}
    for _, r in vehicle_exclusion.iterrows():
        if str(r['facility_id']) in facility_ids: 
            if r['truck_type'] not in scenario_data['Vehicle Exclusion']: 
                scenario_data['Vehicle Exclusion'][r['truck_type']] = set()
            scenario_data['Vehicle Exclusion'][r['truck_type']].add(str(r['facility_id']))

    scenario_data['Facility Groups'] = {}
    for _, r in pd.read_excel(session_state.ref_file, sheet_name="Facility Groups").iterrows(): 
        if str(r['facility_id']) in facility_ids:
            if r['group_id'] not in scenario_data['Facility Groups']: 
                scenario_data['Facility Groups'][r['group_id']] = set()
            scenario_data['Facility Groups'][r['group_id']].add(str(r['facility_id']))
    
    scenario_data['Distance Adj'] = {(str(r['from_facility_id']), str(r['to_facility_id'])): r['distance_adj'] 
                                        for _, r in pd.read_excel(session_state.ref_file, sheet_name="Distance Adj").iterrows()}
    
    #parameter_DF = pd.DataFrame({"parameter": ["Include Return Leg Cost", "Enforce Weight Capacity", "Optimization Runtime Limit"], "value": [True, False, 60]})
    scenario_data['Parameters'] = {r['parameter']:r['value'] 
                                        for _, r in parameter_DF.iterrows()}

    scenario_data['Created'] = pd.to_datetime(datetime.now())
    scenario_data['Modified'] = pd.to_datetime(datetime.now())
    scenario_data['Solved'] = None
    scenario_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    scenario_data['Scenario'] = scenario_name
    scenario_data['Created By'] = session_state.username
    scenario_data['Modified By'] = session_state.username
    scenario_data['Version'] = session_state.ver
    
    scenario_data["Loading Plan"] = pd.DataFrame(columns = ['Route', 'Order Number', 'Customer', 'Customer ID', 'District', 'Weight (kg)', 'Volume (m3)', 'Dispatch No', 'Truck Type', 'Pick Wave'])
    scenario_data["SolSummary_DF"] = pd.DataFrame(columns = ['route', 'truck_type', 'path', 'num_stops', 'vol', 'weight', 'distance', 'time', 'fuel_usage', 'cost', 'vol_cap', 'weight_cap', 'vol_utilization', 'weight_utilization'])
    scenario_data["SolDetail_DF"] = pd.DataFrame(columns = ['route', 'stop_no', 'facility_id', 'distance', 'time', 'fuel_usage', 'cost', 'facility', 'type', 'latitude', 'longitude', 'vol', 'weight'])
    scenario_data["SolMiss_DF"] = pd.DataFrame(columns = ["facility_id"] + facility_DF.columns.tolist())

    scenario_filepath = f"filestore/Scenario {scenario_name}.xlsx"
    save_scenario(scenario_filepath, scenario_data)

    return scenario_data

if __name__ == "__main__":
    pass
    
