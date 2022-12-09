from DRO import DRO, DROFacility, DROVehicle, DROSpec, DROINF
from psm import env as psmenv
import pandas as pd

def test_dro(env): 

    dist_matrix = [[0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662],
                    [548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210],
                    [776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754],
                    [696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358],
                    [582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244],
                    [274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708],
                    [502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480],
                    [194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856],
                    [308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514],
                    [194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468],
                    [536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354],
                    [502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844],
                    [388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730],
                    [354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536],
                    [468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194],
                    [776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798],
                    [662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0],]
    volume = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
    capacity = [5, 5, 30, 30]
    cost = [1, 1, 1, 1]
    time_matrix = dist_matrix

    facilities = [ DROFacility("ABC", 
                            volume[f],               # Volume
                            volume[f],               # Weight
                            999*volume[f],           # Penalty
                            0)                       # Processing Time
                    for f in range(len(volume))]

    vehicles = [DROVehicle("Truck", 
                        capacity[v],                # volume capacity
                        capacity[v],                # weight capacity
                        2000,                      # distance capacity
                        25,                      # transit time limit
                        2000,                      # delivery time limit
                        60,                         # speed
                        100,                        # fixed vehicle usage cost
                        cost[v],                    # variable distnce cost  
                        0,                          # per drop cost
                        )                                
                    for v in range(len(capacity))]

    spec = DROSpec(True,        # include return cost
                   True,        # enforce volume cap
                   True,        # enforce weight cap
                   False,       # enfroce distance cap
                   False,       # enfroce transit time cap
                   True,        # include delivery time cap
                   True,        # Allow miss
                   30,          # time limit in seconds
                   False,        # use speed for transit time
    )

    # set of (veh_no, fac_no) excluded from assignment
    vf_exclusions = {(3, 5), (3, 14), (2, 13), (2, 11)}
    # vf_exclusions = set()
    
    # set of (fac_no, fac_no) that should be delivered by the same vehicle
    ff_pairs = {(6, 7), (9, 11)}
    # ff_pairs = set()

    dist_updates = {(0, 5, 350), (0, 2, 10), (1, 5, 99999), (3, 15, 99999), (3, 4, 99999)}

    solver = DRO(env, dist_matrix, time_matrix, debug=False)
    solver.update_edge_dist(dist_updates)
    solver.update_edge_dist({(3, 15, 99999), (3, 4, 99999)}, veh_no=2)
    solver.solve(facilities, vehicles, vf_exclusions, ff_pairs, spec)

    print(solver.solution_routes)

def zambia_3pl(env):
    df_fac = pd.read_excel(r"./data/DRO_Ref_Zambia.xlsx", sheet_name="Facility")
    df_dist = pd.read_excel(r"./data/DRO_Ref_Zambia.xlsx", sheet_name="Distance")

    facilities = [DROFacility(row['facility_id'], 1, 1, 0, 0) for _, row in df_fac.iterrows()]
    
    facility_idmap = {row['facility_id']:row['facility'].strip() for _, row in df_fac.iterrows()}
    facility_names = [facility_idmap[f.id] for f in facilities]
    facility_ids = [f.id for f in facilities]

    dist_matrix = df_dist.values.tolist()
    assert(df_dist.columns.tolist() == facility_ids)
    print(facility_names)

    solver = DRO(env, dist_matrix, None, debug=False)
    solver._calc_incidence_matrix(threshold=0.01)
    inc_matrix = solver.inc_matrix

    dist_output = []
    for from_f_no in range(solver.num_facs): 
        for to_f_no in range(from_f_no+1, solver.num_facs): 
            from_fac = facility_names[from_f_no]
            to_fac = facility_names[to_f_no]
            distance = dist_matrix[from_f_no][to_f_no]
            inc_dist = inc_matrix[from_f_no][to_f_no]
            inc = 1 if inc_dist < DROINF else 0
            dist_output.append([from_fac, to_fac, distance, inc])

    pd.DataFrame(dist_output, columns=['From Facility', 'To Facility', 'Distance', 'Inc']).to_excel("ZM_Distance.xlsx", index=False)

if __name__ == "__main__":

    env = psmenv.Env(r"./dro.ini", r"c:/temp/dro.log")
    # test_dro(env)
    zambia_3pl(env)