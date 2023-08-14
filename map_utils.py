"""This module contains functions used to create map objects."""

import folium
from pandas.core.frame import DataFrame
from mapbox import Directions
import pandas as pd


def colors(n):
    """Return a color.
    Args:
        colors (float): a float value denoting a color
    Returns:
        (str): a string containing a color name
    """  

    colors = ['black',
              'red',
              'purple',
              'cadetblue',
              'pink',
              'green',
              'blue',
              'gray',
              'darkred',
              'darkgreen',
              'darkblue',
              'orange',
              'lightgreen',
              'lightgray',
              'lightblue',
              ]

    return colors[n%len(colors)]


def draw_route_in_map(m, route_detail, color="black", line_style="solid", draw_routes=False):
    """Draw scenario route information on map.
    Args:
        m (folium.folium.Map): a folium map object
        
        route_detail (pandas.core.frame.DataFrame): a dataframe object containing routes information with lat/long coordinates
        
        color (str, optional):  a string containing a color name (default is 'black')
        
        line_style (str, optional):  a string containing a line type ('solid' or 'dash', default is 'solid')
        
        draw_routes (bool, optional):  a boolean that allows routes to be drawn using open street maps. False draws bird's eye view (default is False)             
    
    Returns:
        m (folium.folium.Map): a folium map object
    """    

    start = route_detail.iloc[0].facility
    route_detail = route_detail.set_index("facility")
    for i in range(1, len(route_detail)):
        stop = route_detail.index[i]

        if isinstance(route_detail.loc[start, "longitude"], float):
            start_coord = [route_detail.loc[start, "longitude"],
                           route_detail.loc[start, "latitude"]]
        else:
            """ Warehouse could be listed twice in the dataframe """
            start_coord = [pd.unique(route_detail.loc[start, "longitude"])[
                0], pd.unique(route_detail.loc[start, "latitude"])[0]]

        if isinstance(route_detail.loc[stop, "longitude"], float):
            stop_coord = [route_detail.loc[stop, "longitude"],
                          route_detail.loc[stop, "latitude"]]
        else:
            stop_coord = [pd.unique(route_detail.loc[stop, "longitude"])[
                0], pd.unique(route_detail.loc[stop, "latitude"])[0]]

        folium.Marker([stop_coord[1], stop_coord[0]],
                      popup=folium.Popup(stop + "<br>" + str(route_detail.iloc[i].route) + "<br>Stop: " + str(route_detail.iloc[i]["stop_no"]) + "<br>Truck type: " + str("None") + "<br>Volume delivered: " + str(
                          route_detail.iloc[i].vol) + " m<sup>3</sup>" + "<br>Distance from previous site: " + str(route_detail.iloc[i]["distance"]) + " km", max_width=350, min_width=250),
                      tooltip=stop,
                      icon=folium.Icon(color=color, icon_color=color)
                      ).add_to(m)

        if draw_routes:
            TOKEN = 'pk.eyJ1IjoiZGFzaGFzYXZpbmEiLCJhIjoiY2tjemVseHY0MDJ5YTJybXRqOWV3ZzY4eSJ9.kJJ10O9yEsg8xoCn2Wo1Lw'
            service = Directions(access_token=TOKEN)

            origin = {
                'type': 'Feature',
                'properties': {'name': 'Start'},
                'geometry': {
                    'type': 'Point',
                    'coordinates': start_coord
                }
            }
            destination = {
                'type': 'Feature',
                'properties': {'name': 'End'},
                'geometry': {
                    'type': 'Point',
                    'coordinates': stop_coord
                }
            }

            response = service.directions(
                [origin, destination], 'mapbox/driving')

            directions = response.geojson()
            points = []

            if len(directions['features']) != 0:  # adding this due to empty list
                for k in range(1, len(directions['features'][0]['geometry']['coordinates'])):
                    points.append(tuple([directions['features'][0]['geometry']['coordinates'][k][1],
                                         directions['features'][0]['geometry']['coordinates'][k][0]]))
                if line_style == "dash":
                    folium.PolyLine(points, popup=str(
                        route_detail.iloc[i].route), color=color, weight=3, opacity=1, dash_array="4").add_to(m)
                else:
                    folium.PolyLine(points, popup=str(
                        route_detail.iloc[i].route), color=color, weight=3, opacity=1).add_to(m)
        else:
            if line_style == "dash":
                folium.PolyLine([tuple([start_coord[1], start_coord[0]]), tuple([stop_coord[1], stop_coord[0]])], popup=str(
                    route_detail.iloc[i].route), color=color, weight=3, opacity=1, dash_array="4").add_to(m)
            else:
                folium.PolyLine([tuple([start_coord[1], start_coord[0]]), tuple([stop_coord[1], stop_coord[0]])], popup=str(
                    route_detail.iloc[i].route), color=color, weight=3, opacity=1).add_to(m)

        start = stop

    return m


def draw_links_in_map(m, facs, links, opt: str, draw_routes=False, color='Black'):
    """Draw link data on map.
    Args:
        m (folium.folium.Map): a folium map object
        
        facs (pandas.core.frame.DataFrame): a dataframe object containing routes information with lat/long coordinates
        
        links (pandas.core.frame.DataFrame): a dataframe object containing link data
        
        opt (str): a string denoting the type links to drawn (full, partial, or path) 
        
        draw_routes (bool, optional):  a boolean that allows routes to be drawn using open street maps. False draws bird's eye view (default is False)             
        
        color (str, optional):  a string containing a color name (default is 'Black')
    """      


    if opt == 'full' or opt == 'partial':
        col_name = 'Distance (km)'
    elif opt == 'path':
        col_name = 'Leg Distance (km)'

    for idx, row in links.iterrows():
        start_lon = facs.loc[facs['Code'] ==
                             row['Origin Facility Code'], 'Lon'].item()
        start_lat = facs.loc[facs['Code'] ==
                             row['Origin Facility Code'], 'Lat'].item()
        start_name = facs.loc[facs['Code'] ==
                              row['Origin Facility Code'], 'Facility'].item()
        start_coord = [start_lon, start_lat]

        stop_lon = facs.loc[facs['Code'] ==
                            row['Destination Facility Code'], 'Lon'].item()
        stop_lat = facs.loc[facs['Code'] ==
                            row['Destination Facility Code'], 'Lat'].item()
        stop_name = facs.loc[facs['Code'] ==
                             row['Destination Facility Code'], 'Facility'].item()
        stop_coord = [stop_lon, stop_lat]

        if opt == 'path':
            if idx == 0:
                folium.Marker([start_coord[1], start_coord[0]],
                            popup=folium.Popup("Facility Name: " + start_name + "<br>Facility Code: " + str(
                                row['Origin Facility Code']) + "<br>Latitude: " + str(start_lat)
                    + "<br>Longitude: " + str(start_lon), max_width=350, min_width=250),
                    # icon_size=(15,15), icon_anchor=(stop_coord[1], stop_coord[0]), shadow_size=(0,0))
                    icon=folium.Icon(color=color, icon_color=color),
                ).add_to(m)

                m.location = [start_lat, start_lon]

            folium.Marker(
                [stop_coord[1], stop_coord[0]],
                popup=folium.Popup("Facility Name: " + stop_name + "<br>Facility Code: " + str(
                    row['Destination Facility Code']) + "<br>Latitude: " + str(start_lat) + "<br>Longitude: " + str(stop_lon), x_width=350, min_width=250),
                    # icon_size=(15,15), icon_anchor=(stop_coord[1], stop_coord[0]), shadow_size=(0,0))
                icon=folium.Icon(color=color, icon_color=color),
            ).add_to(m)
        elif opt == 'partial':
            folium.Marker(
                [start_coord[1], start_coord[0]],
                popup=folium.Popup("Facility Name: " + start_name + "<br>Facility Code: " + str(
                    row['Destination Facility Code']) + "<br>Latitude: " + str(start_lat) + "<br>Longitude: " + str(start_lon), x_width=350, min_width=250),
                    # icon_size=(15,15), icon_anchor=(stop_coord[1], stop_coord[0]), shadow_size=(0,0))
                icon=folium.Icon(color=color, icon_color=color),
            ).add_to(m)

            folium.Marker(
                [stop_coord[1], stop_coord[0]],
                popup=folium.Popup("Facility Name: " + stop_name + "<br>Facility Code: " + str(
                    row['Destination Facility Code']) + "<br>Latitude: " + str(start_lat) + "<br>Longitude: " + str(stop_lon), x_width=350, min_width=250),
                    # icon_size=(15,15), icon_anchor=(stop_coord[1], stop_coord[0]), shadow_size=(0,0))
                icon=folium.Icon(color=color, icon_color=color),
            ).add_to(m)

        if draw_routes:
            TOKEN = 'pk.eyJ1IjoiZGFzaGFzYXZpbmEiLCJhIjoiY2tjemVseHY0MDJ5YTJybXRqOWV3ZzY4eSJ9.kJJ10O9yEsg8xoCn2Wo1Lw'
            service = Directions(access_token=TOKEN)

            origin = {
                'type': 'Feature',
                'properties': {'name': 'Start'},
                'geometry': {
                    'type': 'Point',
                    'coordinates': start_coord
                }
            }
            destination = {
                'type': 'Feature',
                'properties': {'name': 'End'},
                'geometry': {
                    'type': 'Point',
                    'coordinates': stop_coord
                }
            }

            response = service.directions(
                [origin, destination], 'mapbox/driving')

            directions = response.geojson()
            points = []

            if len(directions['features']) != 0:  # adding this due to empty list
                for k in range(1, len(directions['features'][0]['geometry']['coordinates'])):
                    points.append(tuple([directions['features'][0]['geometry']['coordinates'][k][1],
                                         directions['features'][0]['geometry']['coordinates'][k][0]]))

                folium.PolyLine(points, popup=folium.Popup("Origin: " + start_name + "<br>Destination: " + stop_name +
                                                           "<br>Distance: " + str(row[col_name]), max_width=350, min_width=250),
                                color='black', weight=3, opacity=1).add_to(m)
        else:
            folium.PolyLine([tuple([start_coord[1], start_coord[0]]), tuple([stop_coord[1], stop_coord[0]])],
                            popup=folium.Popup("Origin: " + str(row['Origin Facility Code']) + "<br>Destination: " + str(
                                row['Destination Facility Code']) + "<br>Distance: " + str(row[col_name]), max_width=350, min_width=250),
                            color='black', weight=3, opacity=1).add_to(m)
