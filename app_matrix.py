from re import M
from PIL.Image import FASTOCTREE
import pandas as pd
import numpy as np
import streamlit as st
from openpyxl import load_workbook
import matrix_utils as matu
import itertools
from streamlit_folium import folium_static
import map_utils as mapu
import folium
import os


def app_matrix(session_state):

    expand_upload = session_state.data_file == ""
    with st.expander("Upload Data", expanded=expand_upload):
        data_file = st.file_uploader("P1 Data", type=["xlsx"])

        if data_file is not None:
            if st.button("Upload"):
                with open(os.path.join("./filestore", data_file.name), 'wb') as f:
                    f.write(data_file.getbuffer())
                    session_state.data_file = os.path.join(
                        "./filestore", data_file.name)

    if session_state.data_file != "":
        data = read_data_file(session_state.data_file)
        col_left, col_data, space, col_map, col_right = st.columns(
            [0.1, 4, 0.1, 5, 0.1])

        # formatting function - needs visibility to fac list
        def display_facility(option):
            if option == '':
                return ''

            fac_name = data['facs'].loc[data['facs']
                                        ['Code'] == str(option), 'Facility'].item()
            return (fac_name + ' (' + option + ')')

        with col_data:
            fac_list = data['facs']['Code'].to_list()
            fac_list.insert(0, '')
            view = st.radio(label='Displayed Data', options=[
                            'Distance Matrix', 'Matrix Links'])
            if view == 'Matrix Links':
                origin = st.selectbox(
                    label='Origin', options=fac_list, format_func=display_facility, key='og')
                destination = st.selectbox(
                    label='Destination', options=fac_list, format_func=display_facility, key='dest')
                if origin and not destination:
                    st.markdown(
                        "<div style='text-align: center'><h4 style='text-align: center;'><strong>Select a destination to generate path</strong></h4></div>", unsafe_allow_html=True)
                elif destination and not origin:
                    st.markdown(
                        "<div style='text-align: center'><h4 style='text-align: center;'><strong>Please select an origin</strong></h4></div>", unsafe_allow_html=True)
                elif origin == destination and origin == '':
                    st.markdown(
                        "<div style='text-align: center'><h4 style='text-align: center;'><strong>Select origin/destination to view path</strong></h4></div>", unsafe_allow_html=True)

        with col_map:
            use_road = st.checkbox('Use Road Network?', key='001')
            if view == 'Distance Matrix':
                m = folium.Map(location=[-15.416, 28.283], zoom_start=8)
                for idx, row in data['facs'].iterrows():
                    folium.CircleMarker(
                        location=[row['Lat'], row['Lon']],
                        popup=folium.Popup("Facility Name: " + str(row['Facility']) + "<br>Facility Code: " + str(
                            row['Code']) + "<br>Latitude: " + str(row['Lat']) + "<br>Longitude: " + str(row['Lon']), max_width=350, min_width=250),
                        color='black',
                        radius=5
                    ).add_to(m)
            elif view == 'Matrix Links':
                ref = data['facs']['Code'].astype(str).to_list()
                if origin and destination:
                    # processing to get path
                    ref = data['facs']['Code'].astype(str).to_list()
                    orig_idx = pd.Index(ref).get_loc(origin)
                    dest_idx = pd.Index(ref).get_loc(destination)
                    path, total_dist = matu.get_path(
                        data['predecessor'], data['rebuilt_mat'], orig_idx, dest_idx)
                    path_df = pd.DataFrame(path, columns=[
                                           'Origin Facility Code', 'Destination Facility Code', 'Leg Distance (km)'])
                    path_df['Origin Facility Code'] = path_df['Origin Facility Code'].apply(
                        lambda x: ref[x])
                    path_df['Destination Facility Code'] = path_df['Destination Facility Code'].apply(
                        lambda x: ref[x])
                    # write path and distances to df
                    with col_data:
                        st.markdown('#### Total Path Distance: ' +
                                    str("{:.2f}".format(total_dist)))
                        st.dataframe(path_df)

                    # map links
                    m = folium.Map(location=[-15.416, 28.283], zoom_start=8)
                    mapu.draw_links_in_map(m, data['facs'], path_df, 'path', use_road, False)
                elif origin and not destination:
                    # filter links - include origin
                    lon = data['facs'].loc[data['facs']['Code'] == origin, 'Lon'].item()
                    lat = data['facs'].loc[data['facs']['Code'] == origin, 'Lat'].item()

                    df = data['new_links'][(data['new_links']['Origin Facility Code'] == origin) |
                                           (data['new_links']['Destination Facility Code'] == origin)]

                    m = folium.Map(location=[lat, lon], zoom_start=8)
                    mapu.draw_links_in_map(m, data['facs'], df, 'partial', use_road)

                else:
                    m = folium.Map(location=[-15.416, 28.283], zoom_start=8)
                    mapu.draw_links_in_map(m, data['facs'], data['new_links'], 'full', use_road)
                    for idx, row in data['facs'].iterrows():
                        folium.Marker(
                            location=[row['Lat'], row['Lon']],
                            popup=folium.Popup("Facility Name: " + str(row['Facility']) + "<br>Facility Code: " + str(
                                row['Code']) + "<br>Latitude: " + str(row['Lat']) + "<br>Longitude: " + str(row['Lon']), max_width=350, min_width=250),
                            icon=folium.Icon(color='red', icon_color='black'),
                        ).add_to(m)

            folium_static(m, width=925)

        # od matrix or link matrix generation
        use_links = st.checkbox('Toggle Link Matrix', key='003')
        if use_links:
            data['link_matrix'].index = data['link_matrix'].index.map(str)
            st.dataframe(data['link_matrix'].style.format("{:.1f}"))
        else:
            data['matrix'].index = data['matrix'].index.map(str)
            st.dataframe(data['matrix'].style.format("{:.1f}"))

        rec_expander = st.expander(label='Reconciliation')
        with rec_expander:
            subset = data['output'][~data['output']['New Link Path'].isna()]
            st.dataframe(subset)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def read_data_file(data_file, ver=0):
    data = {}
    data['facs'] = pd.read_excel(data_file, sheet_name='Facility List')
    data['facs']['Code'] = data['facs']['Code'].astype(str)

    data['new_links'] = pd.read_excel(data_file, sheet_name='New Links')
    data['new_links']['Origin Facility Code'] = data['new_links']['Origin Facility Code'].astype(
        str)
    data['new_links']['Destination Facility Code'] = data['new_links']['Destination Facility Code'].astype(
        str)

    data['matrix'] = pd.read_excel(
        data_file, sheet_name='Facility Distance Matrix', index_col=0)
    data['link_matrix'] = pd.read_excel(
        data_file, sheet_name='Link Distance Matrix', index_col=0)
    data['link_matrix'].index = data['link_matrix'].index.astype(str)

    data['predecessor'] = pd.read_excel(
        data_file, sheet_name='Predecessor Matrix')
    data['rebuilt_mat'] = pd.read_excel(data_file, sheet_name='Rebuilt Matrix')
    data['output'] = pd.read_excel(data_file, sheet_name='Output')

    #data['link_matrix'].replace(999999, '', inplace=True)

    return data


def draw_link_in_map(m):
    pass
