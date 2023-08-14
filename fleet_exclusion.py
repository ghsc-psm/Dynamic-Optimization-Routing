import pandas as pd
import numpy as np
from pandas import ExcelWriter

def generate_fleet_restrictions(fname):
    """Generates Fleet restrictions based on filled out template
    
    Args:
        fname (str): path to template   
        
    Returns:
        0 if successful - output saved to specified path name with  "- Completed.xlsx" appended
        1 otherwise
    """    

    facs = pd.read_excel(fname, sheet_name='Fac_restrict').reset_index()  
    fleet = pd.read_excel(fname, sheet_name='Fleet_specs').reset_index()  

    #Ensuring no duplicated data in either Fac or Fleet List
    if len(np.unique(fleet['truck_type'])) != len(fleet):
        print("Confirm there are no duplicated truck_types in Fleet List.")
        return 1
    if len(np.unique(facs['facility_id'])) != len(facs):
        print("Confirm there are no duplicated facility_id in Facility_List.")
        return 1    
    if len(np.unique(facs['warehouse_or_hub']))!= 1:
        print("Confirm that warehouse_or_hub name in Facility Tab is accurate. There should only be one warehouse_or_hub listed.")
        return 1
    
    # QA Checks for Template
    for i, fac in facs.iterrows():
        try:
            if fleet[fleet['truck_type'] == fac['min_veh']].reset_index().vol_cap[0] > fleet[fleet['truck_type'] == fac['max_veh']].reset_index().vol_cap[0]:
                print("Ensure that min_veh <= max_veh for every facility. Fleet Restrictions not generated.")
                print(f"Namely, check facility: {fac.facility_id}")
                return 1
        except:
            print("Confirm that all Vehicle Restrictions in Facility_List located in the Fleet_List.")
            print(f"Namely, confirm the accuracy for facility: {fac.facility_id}")
            return 1

    # Data Cleaning for Security Feild:
    facs['security'] =  facs['security'].map(lambda x: str(x).lower())
    facs['security'] = facs['security'].replace(to_replace=['yes','no'], value=[True,False])
    facs['security'] = facs['security'].replace(to_replace=["1","0"], value=[True,False])
    facs['security'] = facs['security'].replace(to_replace=['y','n'], value=[True,False])
    facs['security'] = facs['security'].replace(to_replace=['true','false'], value=[True,False])

    # Creating Dictionary for facs to signal if the fac volume greater than the max vehicle volume capacity
    flagged = {}

    # Generating Fleet Restriction
    fac_excluded_fleet = pd.DataFrame(columns=['warehouse_or_hub','facility_id','truck_type'])    
    for i, fac in facs.iterrows():
        exclude_fleet_security = []
        if fac['security']:
            exclude_fleet_security = fleet[fleet['security'] != fac['security']]['truck_type'].tolist()
            if fac['vol'] > fleet[fleet['truck_type'] == '1T Truck-1D'].reset_index().vol_cap[0]:
                flagged[fac['facility_id']] = 'Facility Volume Greater than Max Vehicle Volume Capacity.'
        
        exclude_fleet_min_vol = fleet[fleet['vol_cap'] < fleet[fleet['truck_type'] == fac['min_veh']].reset_index().vol_cap[0]]['truck_type'].tolist() #+ \
        exclude_fleet_max_vol = fleet[fleet['vol_cap'] > fleet[fleet['truck_type'] == fac['max_veh']].reset_index().vol_cap[0]]['truck_type'].tolist()
        
        if fac['vol'] > fleet[fleet['truck_type'] == fac['max_veh']].reset_index().vol_cap[0]:
            flagged[fac['facility_id']] = 'Facility Volume Greater than Max Vehicle Volume Capacity.'

        exclude_fleet = exclude_fleet_security + exclude_fleet_min_vol + exclude_fleet_max_vol 
        exclude_fleet = np.unique(np.array(exclude_fleet))
        
        fac_excluded_fleet = pd.concat([fac_excluded_fleet, pd.DataFrame({'warehouse_or_hub':[fac['warehouse_or_hub'] for i in range(len(exclude_fleet))],
                                    'truck_type':exclude_fleet,
                                    'facility_id':[fac['facility_id'] for i in range(len(exclude_fleet))],
                                    })], 
                                    axis = 0)
    
    flagged_facs = pd.DataFrame({'warehouse_or_hub':[fac['warehouse_or_hub'] for i, fac in facs.iterrows()],
                        'facility_id':[fac['facility_id'] for i, fac in facs.iterrows()],
                        'max_vehicle_vol': [fleet[fleet['truck_type'] == fac['max_veh']].reset_index().vol_cap[0] for i, fac in facs.iterrows()],
                        'vol':[fac['vol'] for i, fac in facs.iterrows()],
                        'flagged':[flagged[fac['facility_id']] if fac['facility_id'] in set(flagged.keys()) else "" for i, fac in facs.iterrows()]
                })

    writer = ExcelWriter(f'{fname[:-5]}-Completed.xlsx')
    fac_excluded_fleet.to_excel(writer,'Fleet_Restriction_Output',index=False)
    flagged_facs.to_excel(writer,'Volume_Capacity_QA',index=False)
    writer.save()   
    return 0


if __name__ == "__main__":
    fname = 'data\Kenya\Fleet_Restrict_Template_Kenya_082022.xlsx'
    success = generate_fleet_restrictions(fname)
    print(success)