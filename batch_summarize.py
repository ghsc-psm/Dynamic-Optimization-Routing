"""Script to aggregate all Solved Scenario Excel Files into a single script"""

import pandas as pd
import numpy as np
import os
import glob

#use glob to get all the excel files in the folder
path = f'./data/Kenya/batch'
xlsx_files = glob.glob(os.path.join(path, "*.xlsx"))

summary_df = pd.DataFrame(columns=['filename', 'region', 'num_sites', 'total_vol', 'parameters [Return,Vol,Wgt,Dist,TT,DT,Missed,Limit,AdjSpeed]', 'num_dispatches', 
                    'avg_stops', 'avg_ute', 'total_dist', 'total_cost', 'runtime (mins)'])

#loop over files
for f in xlsx_files:
    print('Working on file ', f)
    xls = pd.ExcelFile(f)
    ss = pd.read_excel(xls, 'Solution Summary')
    dels = pd.read_excel(xls, 'Deliveries')
    params = pd.read_excel(xls, 'Parameters')
    meta = pd.read_excel(xls, 'Metadata')
    fac = pd.read_excel(xls, 'Facilities')
    dist_mat = pd.read_excel(xls, 'Distance')
    fleet = pd.read_excel(xls, 'Fleet')

    #capture return distances and costs
    for row in ss:
        path = ss['path'].str.replace(' -> ',', ')
        return_dist_list = []

        for item in path:
            item = list(item.split(", "))
            first_stop = item[0]
            last_stop = item[-1]
            lookup = list(fac.loc[fac['facility'] == last_stop].index.values)
            lookup_dist = dist_mat.iloc[0,lookup[0]]
            return_dist_list.append(lookup_dist)

        veh = ss['truck_type']
        return_truck_variable_cost = []
        return_truck_fixed_cost = []
        for item in veh:
            #print(veh)
            #print(list(fleet.columns))
            lookup = fleet.loc[fleet['truck_type'] == item]
            lookup_cost = lookup['cost_rate'].values[0]
            lookup_fixed_cost = lookup['fixed_cost'].values[0]
            variable_cost = lookup_cost-lookup_fixed_cost
            #print(lookup_cost)
            return_truck_variable_cost.append(variable_cost)
            return_truck_fixed_cost.append(lookup_fixed_cost)
            
    ss['return_dist'] = return_dist_list
    ss['return_veh_var_cost'] = return_truck_variable_cost
    ss['return_veh_daily_cost'] = return_truck_fixed_cost
    #This is just an esimate - would really need to subtract the time remaining on the last OB day to be precise
    ss['return_days'] = ss['return_dist']/40/10.5
    ss.return_days = ss.return_days.round()
    ss['return_leg_cost'] = (ss['return_veh_var_cost'] * ss['return_dist']) + (ss['return_veh_daily_cost'] * ss['return_days']) 
    ss['round_trip_dist'] = ss['return_dist'] + ss['distance']
    ss['round_trip_cost'] = ss['return_leg_cost'] + ss['cost']

    #ss.to_excel("./data/Kenya/batch/summary/ss.xlsx")


    #print(ss) 
         
    filename = f 
    region = fac['county'][0]
    num_sites = len(dels['Customer Name'].value_counts())
    total_vol = round(ss['vol'].sum(),2)
    parameters = list(params['value'])
    num_dispatches = len(ss['route'])
    avg_stops = round(ss['num_stops'].mean(),2)
    sum_ute = ss['vol'].sum()
    sum_cap = ss['vol_cap'].sum()
    avg_ute = round(sum_ute/sum_cap,2)
    total_dist = ss['round_trip_dist'].sum()
    total_cost = round(ss['round_trip_cost'].sum(),2)
    start_time = meta['value'][1]
    end_time = meta['value'][2]
    runtime = end_time - start_time
    runtime = round(runtime.seconds/60,2)

    new_row = []
    cols = [filename,region,num_sites, total_vol, parameters, num_dispatches, 
            avg_stops, avg_ute, total_dist, total_cost, runtime]
    for col in cols:
        new_row.append(col)
   
    summary_df = summary_df.append(pd.DataFrame([new_row],
                                    columns=['filename', 'region', 'num_sites', 'total_vol', 'parameters [Return,Vol,Wgt,Dist,TT,DT,Missed,Limit,AdjSpeed]', 'num_dispatches','avg_stops','avg_ute', 'total_dist', 'total_cost', 'runtime (mins)']),
                                    ignore_index=True)
    #print(new_row)
    
#print(summary_df)

summary_df.to_excel("./data/Kenya/batch/summary/batch_summary.xlsx")
