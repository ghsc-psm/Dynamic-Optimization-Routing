from functools import partial
from collections import namedtuple
import random
import copy

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

DROFacility = namedtuple('Facility',['id', 'vol', 'weight', 'miss_penalty', 'proc_time'])
DROVehicle = namedtuple('Vehicle', ['type', 'vol_cap', 'weight_cap', 'dist_cap', 'transit_time_cap', 'delivery_time_cap', 'speed', 'fix_cost', 'var_cost', 'drop_cost']) 
DROSpec = namedtuple('DROSpec', ['return_cost', 'v_vol_cap', 'v_weight_cap', 'v_dist_cap', 'v_transit_time_cap', 'v_delivery_time_cap', 'allow_miss', 'time_limit', 'use_speed'])

DROSpecDesc = {"Include Return Leg Cost": "Whether to include the last leg of the route returning to the warehouse in cost calculation.",
                    "Enforce Volume Capacity": "Whether to enforce the vehicles volume capacity limit.",
                    "Enforce Weight Capacity": "Whether to enforce the vehicles weight capacity limit",
                    "Enforce Distance Limit": "Whether to enforce the overall transit distance limit on route..",
                    "Enforce Transit Time Limit": "Whether to enforce the overall transit time limit on route.",
                    "Enforce Delivery Time Limit": "Whether to enforce the overall delivery time limit on route.",
                    "Allow Missed Deliveries": "Whether to allow deliveries missed due to limitations (capacity, weight, etc.)", 
                    "Optimization Runtime Limit":"Runtime limit (in seconds) for optimization search. Longer times may further improve solution.",
                    "Adjust Transit Time by Speed": "Whether to adjust transit time based on vehicle speed",}
DROINF = 999999
class DRO:
    """Delivery Route Optimizer
    """
    
    def __init__(self, env, dist_matrix, time_matrix=None, depot_no=0, debug=False):
        """Constructor

        Args:
            env (Env): opex environment object
            dist_matrix (list of list of float/int): square distance matrix, n x n
            depot_no (int, optional): the facility number corresponding to no. Defaults to 0.
            debug (bool, optional): whether to display extra information for debugging. Defaults to False.
        """
        self.log = env.log          # log handle
        self.cfg = env.config       # config handle
        self.debug = debug          # debug flag

        self.original_dist_matrix = copy.deepcopy(dist_matrix)      # original dist matrix is not scaled
        self.dist_matrix = dist_matrix                              # (scaled) distance matrix
        self.time_matrix = time_matrix                              # (scaled) time matrix
        self.dist_matrix_veh = {}                                   # (scaled) distance matrix for vehicle, if specified
        self.time_matrix_veh = {}                                   # (scaled) distance matrix for vehicle, if specified (TODO)

        self.num_facs = len(dist_matrix)        # number of facilities
        self.depot_no = depot_no                # facility no corresponding to depot

        self.dist_factor = 1        # distance factor
        self.time_factor = 1        # time factor
        self.cost_factor = 1        # cost factor
        self.volume_factor = 1      # volume factor
        self.weight_factor = 1      # weight factor

        self.validate_matrices()
        self._calc_incidence_matrix()

    def validate_matrices(self):
        """ Validate data supplied from constructor
        """
        assert 0 <= self.depot_no < self.num_facs

        for m in range(len(self.dist_matrix)): 
            row = self.dist_matrix[m]
            assert self.num_facs == len(row)
            for n in range(self.num_facs): 
                if m == n: 
                    assert self.dist_matrix[m][n] == 0
                else:
                    assert self.dist_matrix[m][n] >=0

        if self.time_matrix is not None: 
            for m in range(len(self.time_matrix)): 
                row = self.time_matrix[m]
                assert self.num_facs == len(row)
                for n in range(self.num_facs): 
                    if m == n: 
                        assert self.time_matrix[m][n] == 0
                    else:
                        assert self.time_matrix[m][n] >= 0       

    def _calc_incidence_matrix(self, threshold=0.001): 
        """Calculate incidence matrix 
        The incidence matrix and original incidence matrix are both unscale.

        Args:
            threshold (float, optional): threshold for deviation for triangular check. Defaults to 0.01.
        """
        self.original_inc_matrix = copy.deepcopy(self.original_dist_matrix)
        for k in range(self.num_facs):
            for i in range(self.num_facs):
                for j in range(self.num_facs):
                    if self.original_dist_matrix[i][k]>0 and self.original_dist_matrix[k][j]>0 and self.original_dist_matrix[i][j]>0 and abs(self.original_dist_matrix[i][j] - self.original_dist_matrix[i][k] - self.original_dist_matrix[k][j]) <= threshold*self.original_dist_matrix[i][j]: 
                        self.original_inc_matrix[i][j] = DROINF

        self.inc_matrix = copy.deepcopy(self.original_inc_matrix)   # Original incidence matrix is not scaled
        if self.debug: 
            for i in range(self.num_facs): 
                self.log.info(f"{i}: {self.original_dist_matrix[i]} : {self.original_inc_matrix[i]}")

    def update_edge_dist(self, dist_updates, symmetric=True, veh_no=None): 
        """Update distance matrix based on updates to the edges

        Args:
            dist_updates (set of tuples): distance updates in the form of a set of (from_fac_no, to_fac_no, new_dist) 
            symmetric (bool, optional): whether the updates should be applied on both directions. Defaults to True.
            veh_no ([type], optional): if supplied, the updates are for a particular veichle. Defaults to None.
        """
        if len(dist_updates) == 0: 
            return 

        if veh_no is None:          # Only update the self.inc_matrix if it is not vehicle specific
            for (f1, f2, new_dist) in dist_updates: 
                self.inc_matrix[f1][f2] = new_dist
                if symmetric: 
                    self.inc_matrix[f2][f1] = new_dist

        dist_matrix = copy.deepcopy(self.inc_matrix)
        if veh_no is not None:      # Update the distance matrix with vehicle specific updates
            for (f1, f2, new_dist) in dist_updates: 
                dist_matrix[f1][f2] = new_dist
                if symmetric: 
                    dist_matrix[f2][f1] = new_dist

        for k in range(self.num_facs):          # All-pair shortest path algorithm
            for i in range(self.num_facs):
                for j in range(self.num_facs):
                    dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k]+dist_matrix[k][j])

        if self.dist_factor != 1:               # Scale distance matrix with distance factor. 
            for i in range(self.num_facs):      
                for j in range(self.num_facs):
                    dist_matrix[i][j] *= self.dist_factor

        if veh_no is None: 
            self.dist_matrix = dist_matrix
        else:
            self.dist_matrix_veh[veh_no] = dist_matrix

        if self.debug: 
            for i in range(self.num_facs):
                for j in range(self.num_facs):
                    if dist_matrix[i][j] != self.original_dist_matrix[i][j]: 
                        self.log.info(f"Distance from {i} to {j} is updated from {self.original_dist_matrix[i][j]} to {dist_matrix[i][j]}")
        
            for i in range(self.num_facs): 
                self.log.info(f"{i}: {dist_matrix[i]} : {self.inc_matrix[i]}")

    def get_volume(self, fac_no):
        """Get volume for a facility

        Args:
            fac_no (int): facility no
        Returns:
            float: Unscaled volume to be delivered to the facility
        """
        return self._volume_C(self.manager.NodeToIndex(fac_no))/self.volume_factor

    def _volume_C(self, fac_index):
        """Internal volume callback for a facility

        Args:
            fac_index (int64): internal facility index
        Returns:
            int: scaled volume to be delivered to the facility
        """
        return int(self.facilities[self.manager.IndexToNode(fac_index)].vol)

    def get_weight(self, fac_no):
        """Get weight for a facility

        Args:
            fac_no (int): facility no
        Returns:
            float: Unscaled weight to be delivered to the facility
        """
        return self._weight_C(self.manager.NodeToIndex(fac_no))/self.weight_factor

    def _weight_C(self, fac_index):
        """Internal weight callback for a facility

        Args:
            fac_index (int64): internal facility index
        Returns:
            int: scaled weight to be delivered to the facility
        """
        return int(self.facilities[self.manager.IndexToNode(fac_index)].weight)

    def get_vehicle_distance(self, from_no, to_no, veh_no): 
        """Get distance traveled by vehicle from one facility to another

        Args:
            from_no (int): from-facility no 
            to_no (int): to-facility no
            veh_no (int): vehicle no
        Returns:
            float: Unscaled distance traveled by vehicle from one facility to another
        """
        return self._vehicle_distance_C(self.manager.NodeToIndex(from_no), self.manager.NodeToIndex(to_no), veh_no)/self.dist_factor

    def _vehicle_distance_C(self, from_index, to_index, veh_no, adj_return_arc=True): 
        """Internal distance callback for a vehicle from one facility to another

        Args:
            from_index (int64): from-facility index
            to_index (int64): to-facility index
            veh_no (int): vehicle no
            adj_return_arc (bool): whether to adj distance for arc return to depot, defult to True
        Returns:
            int: scaled distance traveled by vehicle from one facility to another
        """
        from_no = self.manager.IndexToNode(from_index)    # routing variable Index to demands NodeIndex
        to_no = self.manager.IndexToNode(to_index)        # routing variable Index to demands NodeIndex

        if not self.spec.return_cost and to_no==self.depot_no:    # return to depot distance not included if return cost is not considered
            return 0

        # By setting the arc factor to be less than 1 help the solver to choose ending the route at the farthest facility
        arc_factor = 0.95 if adj_return_arc and to_no==self.depot_no else 1.0
        return int(arc_factor*self.dist_matrix[from_no][to_no]) if veh_no not in self.dist_matrix_veh else int(arc_factor*self.dist_matrix_veh[veh_no][from_no][to_no])

    def get_vehicle_transit_cost(self, from_no, to_no, veh_no):
        """ Get transit cost by vehicle from one facility to another

        Args:
            from_no (int): from-facility no 
            to_no (int): to-facility no
            veh_no (int): vehicle no
        Returns:
            float: Unscaled transit cost by vehicle from one facility to another
        """
        return self._vehicle_transit_cost_C(self.manager.NodeToIndex(from_no), self.manager.NodeToIndex(to_no), veh_no)/self.cost_factor

    def _vehicle_transit_cost_C(self, from_index, to_index, veh_no):
        """Internal transit cost callback for a vehicle from one facility to another

        Args:
            from_index (int64): from-facility index
            to_index (int64): to-facility index
            veh_no (int): vehicle no
        Returns:
            int: scaled transit cost by vehicle from one facility to another
        """
        return int(self.vehicles[veh_no].var_cost * self._vehicle_distance_C(from_index, to_index, veh_no))
    
    def get_vehicle_delivery_cost(self, from_no, to_no, veh_no):
        """Get delivery cost (transit cost + drop cost) for a vehicle from one facility to another

        Args:
            from_no (int): from-facility no 
            to_no (int): to-facility no
            veh_no (int): vehicle no
        Returns:
            float: unscaled delivery cost by vehicle from one facility to another
        """
        return self._vehicle_delivery_cost_C(self.manager.NodeToIndex(from_no), self.manager.NodeToIndex(to_no), veh_no)/self.cost_factor

    def _vehicle_delivery_cost_C(self, from_index, to_index, veh_no):
        """Internal delivery cost callback for a vehicle from one facility to another

        Args:
            from_index (int64): from-facility index
            to_index (int64): to-facility index
            veh_no (int): vehicle no
        Returns:
            int: scaled delivery cost by vehicle from one facility to another
        """
        to_node = self.manager.IndexToNode(to_index)    # routing variable Index to demands NodeIndex
        if to_node == self.depot_no: 
            return self._vehicle_transit_cost_C(from_index, to_index, veh_no)
        return int(self._vehicle_transit_cost_C(from_index, to_index, veh_no) + self.vehicles[veh_no].drop_cost*self.cost_factor)

    def get_vehicle_transit_time(self, from_no, to_no, veh_no):
        """Get transit time for a vehicle from one facility to another

        Args:
            from_no (int): from-facility no 
            to_no (int): to-facility no
            veh_no (int): vehicle no
        Returns:
            float: unscaled transit time by vehicle from one facility to another
        """
        return self._vehicle_transit_time_C(self.manager.NodeToIndex(from_no), self.manager.NodeToIndex(to_no), veh_no)/self.time_factor

    def _vehicle_transit_time_C(self, from_index, to_index, veh_no, adj_return_arc=True): 
        """Internal transit time callback for a vehicle from one facility to another

        Args:
            from_index (int64): from-facility index
            to_index (int64): to-facility index
            veh_no (int): vehicle no
            adj_return_arc (bool): whether to adj distance for arc return to depot, defult to True
        Returns:
            int: scaled transit time by vehicle from one facility to another
        """
        if self.spec.use_speed or self.time_matrix is None: 
            return int(self._vehicle_distance_C(from_index, to_index, veh_no)/(self.dist_factor*self.vehicles[veh_no].speed))
        else:
            from_no = self.manager.IndexToNode(from_index)    # routing variable Index to demands NodeIndex
            to_no = self.manager.IndexToNode(to_index)        # routing variable Index to demands NodeIndex

            if not self.spec.return_cost and to_no==self.depot_no:    # return to depot distance not included if return cost is not considered
                return 0
            
            # By setting the arc factor to be less than 1 help the solver to choose ending the route at the farthest facility
            arc_factor = 0.95 if adj_return_arc and to_no==self.depot_no else 1.0
            return int(arc_factor*self.time_matrix[from_no][to_no]) if veh_no not in self.time_matrix_veh else int(arc_factor*self.time_matrix_veh[veh_no][from_no][to_no])

    def get_vehicle_delivery_time(self, from_no, to_no, veh_no):
        """Get delivery time (transit + drop) for a vehicle from one facility to another

        Args:
            from_no (int): from-facility no
            to_no (int): to-facility no
            veh_no (int): vehicle no
        Returns:
            float: unscaled delivery time by vehicle from one facility to another
        """
        return self._vehicle_delivery_time_C(self.manager.NodeToIndex(from_no), self.manager.NodeToIndex(to_no), veh_no)/self.time_factor

    def _vehicle_delivery_time_C(self, from_index, to_index, veh_no): 
        """Internal delivery time callback for a vehicle from one facility to another

        Args:
            from_index (int64): from-facility index
            to_index (int64): to-facility index
            veh_no (int): vehicle no
        Returns:
            int: scaled transit time by vehicle from one facility to another
        """
        to_node = self.manager.IndexToNode(to_index)    # routing variable Index to demands NodeIndex
        return int(self._vehicle_transit_time_C(from_index, to_index, veh_no) + self.facilities[to_node].proc_time*self.time_factor)

    def validate_data(self): 
        """Data Validation
        """
        assert(len(self.facilities) == self.num_facs)
        for fac in self.facilities:
            assert fac.vol >= 0
            assert fac.weight >= 0
            assert fac.miss_penalty >= 0
            assert fac.proc_time >= 0
        for veh in self.vehicles:
            assert veh.vol_cap >= 0
            assert veh.weight_cap >= 0
            assert veh.dist_cap >= 0
            assert veh.transit_time_cap >= 0
            assert veh.delivery_time_cap >= 0
            assert veh.speed > 0
            assert veh.fix_cost >= 0
            assert veh.var_cost >= 0
        for (veh, fac) in self.vf_exclusions: 
            assert fac != self.depot_no
            assert 0 <= fac < self.num_facs
            assert 0 <= veh < self.num_veh

        self.log.info("Successful data validation")

    def solve(self, facilities, vehicles, vf_exclusions, ff_pairs, spec, presolve=False):
        """Vehicle routing solve

        Args:
            facilities (list of Facility): list of facilities with delivery information
            vehicles (list of Vehicle): list of vehicles with vehicle information
            vf_exclusions (list of tuples): list of (vehicle no, facility no) tuples indicating exclusion of assignment
            ff_pairs (list of tuples): list of (facility no, facility no) tuples indicating assigned to the same vehicle
            spec (Spec): solver control specification
        """
        self.facilities = facilities
        self.vehicles = vehicles
        self.vf_exclusions = vf_exclusions
        self.ff_pairs = ff_pairs
        self.spec = spec
        self.num_veh = len(self.vehicles)

        self.log.info(self.spec)
        self.log.info(f"Depot No. :{self.depot_no}")
        self.log.info(f"# of Facilities: {len(self.facilities)}")
        self.log.info(f"# of vehicles  : {len(self.vehicles)}")
        self.log.info(f"# of Exclusions: {len(self.vf_exclusions)} ({self.vf_exclusions})")
        self.log.info(f"# of Fac Pairs : {len(self.ff_pairs)} ({self.ff_pairs})")
        
        self.process_data()

        # Check for additional non-accessible facility for trucks
        dist_exclusions = set()
        for f in range(self.num_facs): 
            if self.dist_matrix[self.depot_no][f] >= DROINF: 
                for v in range(self.num_veh): 
                    dist_exclusions.add((v, f))
        for v in self.dist_matrix_veh: 
            for f in range(self.num_facs): 
                if self.dist_matrix_veh[v][self.depot_no][f] >= DROINF: 
                    dist_exclusions.add((v, f))
        if len(dist_exclusions): 
            self.log.info(f"Dist Exclusions: {len(dist_exclusions)} ({dist_exclusions})")
            self.vf_exclusions = self.vf_exclusions.union(dist_exclusions)
            self.log.info(f"# of Exclusions: {len(self.vf_exclusions)}")

        # Create index manager and routing model
        self.manager = pywrapcp.RoutingIndexManager(self.num_facs, self.num_veh, self.depot_no)
        self.routing = pywrapcp.RoutingModel(self.manager)

        # Register call backs
        self.volume_CI = self.routing.RegisterUnaryTransitCallback(self._volume_C)
        self.weight_CI = self.routing.RegisterUnaryTransitCallback(self._weight_C)

        self.distance_CIL = [self.routing.RegisterTransitCallback(partial(self._vehicle_distance_C, veh_no=n)) for n in range(self.num_veh)]
        self.transit_time_CIL = [self.routing.RegisterTransitCallback(partial(self._vehicle_transit_time_C, veh_no=n)) for n in range(self.num_veh)]
        self.transit_cost_CIL = [self.routing.RegisterTransitCallback(partial(self._vehicle_transit_cost_C, veh_no=n)) for n in range(self.num_veh)]
        self.delivery_time_CIL = [self.routing.RegisterTransitCallback(partial(self._vehicle_delivery_time_C, veh_no=n)) for n in range(self.num_veh)]
        self.delivery_cost_CIL = [self.routing.RegisterTransitCallback(partial(self._vehicle_delivery_cost_C, veh_no=n)) for n in range(self.num_veh)]

        for v_no in range(self.num_veh): 
            self.routing.SetFixedCostOfVehicle(self.vehicles[v_no].fix_cost, v_no)
            self.routing.SetArcCostEvaluatorOfVehicle(self.delivery_cost_CIL[v_no], v_no)

        if self.spec.v_vol_cap:
            self.routing.AddDimensionWithVehicleCapacity(self.volume_CI, 
                                                         0,              # null capacity slack
                                                         [veh.vol_cap for veh in self.vehicles],   # vehicle maximum volume capacities
                                                         True,           # start cumul to zero
                                                         'VolumeCap')

        if self.spec.v_weight_cap:
            self.routing.AddDimensionWithVehicleCapacity(self.weight_CI, 
                                                         0,              # null capacity slack
                                                         [veh.weight_cap for veh in self.vehicles],   # vehicle maximum weight capacities
                                                         True,           # start cumul to zero
                                                         'WeightCap')

        if self.spec.v_dist_cap:
            self.routing.AddDimensionWithVehicleTransitAndCapacity(self.distance_CIL, 
                                                         0,              # null capacity slack
                                                         [veh.dist_cap for veh in self.vehicles],   # vehicle maximum distance capacities
                                                         True,           # start cumul to zero
                                                         'DistCap')

        if self.spec.v_transit_time_cap:
            self.routing.AddDimensionWithVehicleTransitAndCapacity(self.transit_time_CIL, 
                                                         0,              # null capacity slack
                                                         [veh.transit_time_cap for veh in self.vehicles],   # vehicle maximum transit time capacities
                                                         True,           # start cumul to zero
                                                         'TransitTimeCap')

        if self.spec.v_delivery_time_cap:
            self.routing.AddDimensionWithVehicleTransitAndCapacity(self.delivery_time_CIL, 
                                                         0,              # null capacity slack
                                                         [veh.delivery_time_cap for veh in self.vehicles],   # vehicle maximum transit time capacities
                                                         True,           # start cumul to zero
                                                         'DeliveryTimeCap')

        if self.spec.allow_miss: 
            for fac_no in [fac_no for fac_no in range(self.num_facs) if fac_no != self.depot_no]:
                self.routing.AddDisjunction([self.manager.NodeToIndex(fac_no)], self.facilities[fac_no].miss_penalty)

        if len(self.vf_exclusions) > 0: 
            for fac_no in [fac_no for fac_no in range(self.num_facs) if fac_no != self.depot_no]:
                fac_index = self.manager.NodeToIndex(fac_no)
                allowed_vehs = [veh_no for veh_no in range(self.num_veh) if (veh_no, fac_no) not in self.vf_exclusions]
                if len(allowed_vehs) != self.num_veh: 
                    self.routing.SetAllowedVehiclesForIndex(allowed_vehs, fac_index)

        if len(self.ff_pairs) > 0: 
            for (f1, f2) in self.ff_pairs: 
                f1_index = self.manager.NodeToIndex(f1)
                f2_index = self.manager.NodeToIndex(f2)
                self.routing.AddPickupAndDelivery(f1_index, f2_index)
                self.routing.solver().Add(self.routing.VehicleVar(f1_index) == self.routing.VehicleVar(f2_index))

        # fst_strategies = [routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
        #                     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC, 
        #                   routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        #                   routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        #                   #routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        #                   routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        #                   ]
        # random.shuffle(fst_strategies)
        # for s in fst_strategies: 
        #     search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        #     search_parameters.first_solution_strategy = s
        #     self.solution = self.routing.SolveWithParameters(search_parameters)
        #     if self.solution:
        #         self.log.info(f"Strategy: {s}, Objective Value: {self.solution.ObjectiveValue()}")
        #     else:
        #         self.log.info(f"Strategy: {s}, No solution")

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        if self.spec.time_limit <= 0 or presolve: 
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            search_parameters.time_limit.seconds = 60
        else: 
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            search_parameters.time_limit.seconds = self.spec.time_limit
            search_parameters.solution_limit = 9999

        # Solve the problem
        self.solution = self.routing.SolveWithParameters(search_parameters)
        self.status = ['ROUTING_NOT_SOLVED', 'ROUTING_SUCCESS', 'ROUTING_FAIL', 'ROUTING_FAIL_TIMEOUT', 'ROUTING_INVALID'][self.routing.status()]
        self.log.info(f"Solution Status: {self.status}")

        if self.solution: 
            self.process_solution()
            self.log.info(f"Objective Value: {self.solution.ObjectiveValue()}")
        else:
            self.log.info("Failed to obtain a solution")

    def process_data(self):
        """ Process data with validation and automatic scaling
        """
        self.validate_data()

        ### TODO: Scaling Data if needed

        if self.dist_factor != 1: 
            self.log.info(f"Distance Factor: {self.dist_factor}")
        if self.volume_factor != 1: 
            self.log.info(f"Volume Factor  : {self.volume_factor}")
        if self.weight_factor != 1: 
            self.log.info(f"Weight Factor  : {self.weight_factor}")
        if self.time_factor != 1: 
            self.log.info(f"Time Factor    : {self.time_factor}")
        if self.cost_factor != 1: 
            self.log.info(f"Cost Factor    : {self.cost_factor}")

    def process_solution(self, verify_solution=False): 
        """ Process solution and verify solution
        """
        self.solution_routes = []
        for veh_no in range(self.num_veh):
            route = []
            index = self.routing.Start(veh_no)
            route.append(self.manager.IndexToNode(index))
            while not self.routing.IsEnd(index):
                index = self.solution.Value(self.routing.NextVar(index))
                route.append(self.manager.IndexToNode(index))
            self.solution_routes.append(route)

        if verify_solution: 
            total_cost = 0
            sites_delivered = set()
            for veh_no in range(self.num_veh): 
                route = self.solution_routes[veh_no]
                sites_delivered = sites_delivered.union(route)
                if route == [0, 0]: 
                    self.log.info(f"Vehicle {veh_no} not assigned a route")
                    continue
                self.log.info(f"Vehicle {veh_no}: {route}")
                volume = 0
                weight = 0
                distance = 0
                transit_time = 0
                delivery_time = 0
                var_cost = 0
                for n in range(len(route)-1): 
                    curr_index = self.manager.NodeToIndex(route[n])
                    next_index = self.manager.NodeToIndex(route[n+1])
                    volume += self._volume_C(curr_index)
                    weight += self._weight_C(curr_index)
                    distance += self._vehicle_distance_C(curr_index, next_index, veh_no, adj_return_arc=False)
                    transit_time += self._vehicle_transit_time_C(curr_index, next_index, veh_no, adj_return_arc=False)
                    delivery_time += self._vehicle_delivery_time_C(curr_index, next_index, veh_no)
                    var_cost += self._vehicle_transit_cost_C(curr_index, next_index, veh_no)
                
                fix_cost = self.routing.GetFixedCostOfVehicle(veh_no)
                total_cost += var_cost
                total_cost += fix_cost

                self.log.info(f"""   Volume       : {volume/self.volume_factor} {f"({self.vehicles[veh_no].vol_cap})" if self.spec.v_vol_cap else ""}""")
                self.log.info(f"""   Weight       : {weight/self.weight_factor} {f"({self.vehicles[veh_no].weight_cap})" if self.spec.v_weight_cap else ""}""")
                self.log.info(f"""   Distance     : {distance/self.dist_factor} {f"({self.vehicles[veh_no].dist_cap})" if self.spec.v_dist_cap else ""}""")
                self.log.info(f"""   Transit Time : {transit_time/self.time_factor} {f"({self.vehicles[veh_no].transit_time_cap})" if self.spec.v_transit_time_cap else ""}""")
                self.log.info(f"""   Delivery Time: {delivery_time/self.time_factor} {f"({self.vehicles[veh_no].delivery_time_cap})" if self.spec.v_delivery_time_cap else ""}""")
                self.log.info(f"""   Variable Cost: {var_cost/self.cost_factor}""")
                self.log.info(f"""   Fixed Cost   : {fix_cost/self.cost_factor}""")

            self.log.info(f"Total Variable + Fixed Cost: {total_cost/self.cost_factor}")
            dropped_sites = set(range(self.num_facs)).difference(sites_delivered)
            if len(dropped_sites) == 0:
                self.log.info(f"All sites were delivered")
            else: 
                total_volume_dropped = sum(self.facilities[fac_no].vol for fac_no in dropped_sites)
                total_weight_dropped = sum(self.facilities[fac_no].weight for fac_no in dropped_sites)
                self.log.info(f"Sites Missed : {len(dropped_sites)} {dropped_sites}")
                self.log.info(f"Volume Missed: {total_volume_dropped}")
                self.log.info(f"Weight Missed: {total_weight_dropped}")
    