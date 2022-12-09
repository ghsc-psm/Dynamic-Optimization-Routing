"""This module contains functions used to create matrices."""

from re import M
import pandas as pd
import numpy as np
import streamlit as st
from openpyxl import load_workbook
import matrix_utils as mu
import itertools
from streamlit_folium import folium_static
import map_utils as mu
import folium
import os

DROINF = 999999

def get_way_points(pmatrix, i, j, points):
    """Get location of an intermediate point on map.
    Args:
        pmatrix (pandas.core.frame.DataFrame): a dataframe object containing a predecessor matrix
        
        i (int): an integer value denoting a facility
        
        j (int): an integer value denoting a facility
        
        points (list): a list containing points     
    """

    points.append(j)
    if i != j and j != DROINF:
        get_way_points(pmatrix, i, int(pmatrix[i][j]), points)


def get_path(pmatrix, dmatrix, i, j):
    """Get location of an intermediate point on map.
    Args:
        pmatrix (pandas.core.frame.DataFrame): a dataframe object containing a predecessor matrix
        
        dmatrix (pandas.core.frame.DataFrame): a dataframe object containing a distance matrix
        
        i (int): an integer value denoting a facility
        
        j (int): an integer value denoting a facility
    
    Returns:
        if criteria is met:
            path (list): a list containing path information
            
            total_dist (float): a float value denoting the total distance of a route
            
        if criteria is not met:
            (None): a datatype with no value
            
            (None): a datatype with no value
    """     

    points = []
    get_way_points(pmatrix, i, j, points)
    points.reverse()
    if points[0] != i:
        return None, None

    path = []
    total_dist = 0
    for p in range(len(points)-1):
        s = points[p]
        t = points[p+1]
        dist = dmatrix[s][t]
        if dist == DROINF:
            return None, None
        path.append((s, t, dist))
        total_dist += dist
    return path, total_dist


def link_mat_to_full(mat_links, num_facs):
    """Create full matrix using network links.
    Args:
        mat_links (pandas.core.frame.DataFrame): a dataframe object containing matrix links
        
        num_facs (pandas.core.frame.DataFrame): a dataframe object containing a matrix
    Returns:
        d_matrix (pandas.core.frame.DataFrame): a dataframe object containing a distance matrix
            
    """     

    d_matrix = mat_links.copy(deep=True)
    for k in range(num_facs):
        for i in range(num_facs):
            for j in range(num_facs):
                d_matrix[i][j] = min(
                    d_matrix[i][j], d_matrix[i][k]+d_matrix[k][j])
    return d_matrix


def link_list_to_mat(link_list, facs, time: bool, full=True):
    """Create a full or non full matrix.
    Args:
        link_list (pandas.core.frame.DataFrame): a dataframe object containing link data
        
        facs (pandas.core.frame.DataFrame): a dataframe object containing FACS data
        
        time (bool, optional): a boolean denoting wether Link Time should be used in calculations (default is False)
        
        full (bool, optional): a boolean denoting wether a full or non full matrix should be created (default is True)        
    
    Returns:
        if full is True:
            rebuilt (pandas.core.frame.DataFrame): a dataframe object containing a full matrix
        
        if full is False:        
            output (pandas.core.frame.DataFrame): a dataframe object containing a non full matrix
        
    """       

    
    link_list['Origin Facility Code'] = link_list['Origin Facility Code'].astype(str)
    link_list['Destination Facility Code'] = link_list['Destination Facility Code'].astype(str)

    col = 'Link Time' if time else 'Link Distance'
    fac_list = facs['facility_id'].astype(str).to_list()
    output = pd.DataFrame(index=fac_list, columns=fac_list)
    # zero out diagonal
    np.fill_diagonal(output.values, 0)
    # update values
    for idx, row in link_list.iterrows():
        output[row['Origin Facility Code']][row['Destination Facility Code']] = row[col]
        output[row['Destination Facility Code']][row['Origin Facility Code']] = row[col]
        #output.at[row['Origin Facility Code'], row['Destintation Facility Code']] = row[col]
        #output.at[row['Destination Facility Code'], row['Origin Facility Code']] = row[col]
    # fill remaining with NA
    output.fillna(DROINF, inplace=True)
    output.reset_index(drop=True, inplace=True)
    output.columns = list(range(len(fac_list)))

    if full:
        rebuilt = link_mat_to_full(output, len(fac_list))
        return rebuilt
    else:
        return output