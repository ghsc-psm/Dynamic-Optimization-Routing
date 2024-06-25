"""This module contains functions used to analyze scenarios."""

import streamlit as st
import folium
from streamlit_folium import folium_static
import configparser

from scenario import read_scenario
from map_utils import colors, draw_route_in_map

TABLE_HACK = """
<style>
table td:nth-child(1) {
    display: none
}
table th:nth-child(1) {
    display: none
}
</style>
"""

country_config = configparser.ConfigParser()
country_config.read("./country_cfg.toml")
local_currency  = country_config.get("country", "currency").upper()

def get_stats_row_html(stats): 
    """Construct html table to display statistics.
    Args:
        stats (pandas.core.frame.DataFrame): a dataframe object containing statistics
    
    Returns:
        html_str (str): a string containing html table specifications   
    """      
    ncol = len(stats)
    wdt_perc = f"{1/ncol:.4%}"
    html_str = """
        <table style="height: 78px; width: 100%; border-collapse: collapse;" border="0">
        <tbody>
        <tr style="height: 61px;">
        <td style="width: 0%; height: 61px; border-style: none; text-align: center;">
        <h2></h2>
        </td>"""
    for n in range(ncol): 
        html_str += f"""
            <td style="width: {wdt_perc}; height: 61px; border-style: none; text-align: center;">
            <h2>{stats[n][0]}</h2>
            </td>"""
    html_str += f"""
        </tr>
        <tr style="height: 17px;">
        <td style="width: 0%; height: 17px; border-style: none; text-align: center;">
        </td>"""
    for n in range(ncol):
        html_str += f"""
            <td style="width: {wdt_perc}; height: 17px; border-style: none; text-align: center;">{stats[n][1]}</td>"""
    html_str += f"""
        </tr>
        </tbody>
        </table>"""
    return html_str

def display_route(route, route_detail, pick_waves, roads): 
    """Display a map with scenario route information.
    Args:
        route (pandas.core.frame.DataFrame): a dataframe object containing route data
        
        route_detail (pandas.core.frame.DataFrame): a dataframe object containing routing details
        
        pick_waves (pandas.core.frame.DataFrame): a dataframe object containing pick wave data
        
        roads (pandas.core.frame.DataFrame): a dataframe object containing road data
    """     

    col_left, col_smry, space, col_map, col_right = st.columns([0.1, 4, 0.1, 5, 0.1])
    with col_smry: 
        st.subheader(f"{route['route']} ({route['truck_type']}): {route['path']}")
        stats = [(f"{route['truck_type']}", "Vehicle"), 
                (f"{route['vol']:.2f}", "Volume (m\u00B3)"),
                (f"{route['vol_utilization']:.1%}", "Utilization")]
        st.markdown(get_stats_row_html(stats), unsafe_allow_html=True)
        stats = [(f"{route['num_stops']}", "Stops"), 
                (f"{int(round(route['distance'], 0))}", "Distance (KM)"),
                (f"{int(round(route['fuel_usage'], 0)):,}", "Fuel (L)"),
                (f"{int(round(route['cost'], 0)):,}", f"Cost ({local_currency})")]
        st.markdown(get_stats_row_html(stats), unsafe_allow_html=True)

        st.markdown(TABLE_HACK, unsafe_allow_html=True)
        detail_info_df = pick_waves[['Route', 'District', 'Customer']].drop_duplicates(ignore_index=True)
        st.markdown("#### Suggested Loading Order")
        st.table(detail_info_df)
    
    with col_map:
        route_no = int(route['route'].split(' ')[-1])
        m = folium.Map(location = [route_detail.iloc[0].latitude, route_detail.iloc[0].longitude], zoom_start = 10)
        folium.Marker([route_detail.iloc[0].latitude, route_detail.iloc[0].longitude], tooltip = route_detail.iloc[0].facility).add_to(m)
        m = draw_route_in_map(m, route_detail, colors(route_no-1), "solid", roads)
        folium_static(m, width = 800)


def app_review(session_state): 
    """Display a map with scenario route information.
    Args:
        route (pandas.core.frame.DataFrame): a dataframe object containing route data
        
        route_detail (pandas.core.frame.DataFrame): a dataframe object containing routing details
        
        pick_waves (pandas.core.frame.DataFrame): a dataframe object containing pick wave data
        
        roads (pandas.core.frame.DataFrame): a dataframe object containing road data
    """  

    scenario = session_state.scenario_data
    st.markdown(f"## Review Dispatch Solution to Scenario ({scenario['Scenario']}) ✨", unsafe_allow_html=True)
    st.markdown("***")
    
    with st.expander("Overall Summary", expanded=True):
        col_left, col_smry, space, col_map, col_right = st.columns([0.1, 4, 0.1, 5, 0.1])
        with col_smry:
            st.markdown("""<h2 style="text-align: center;"><strong>Summary Statistics</strong></h2>""", unsafe_allow_html=True)
            stats = [(f"{len(scenario['SolSummary_DF'])}", "Dispatches"), 
                    (f"{int(round(scenario['SolSummary_DF']['vol'].sum(), 0)):,}", "Volume (m\u00B3)"),
                    (f"{scenario['SolSummary_DF']['vol'].sum()/scenario['SolSummary_DF']['vol_cap'].sum():.1%}", "Utilization"),]
            st.markdown(get_stats_row_html(stats), unsafe_allow_html=True)

            stats = [(f"{scenario['SolSummary_DF']['num_stops'].sum()}", "Stops"),
                    (f"{int(round(scenario['SolSummary_DF']['distance'].sum(), 0)):,}", "Distance (KM)"),
                    (f"{int(round(scenario['SolSummary_DF']['fuel_usage'].sum(), 0)):,}", "Fuel (L)"),
                    (f"{int(round(scenario['SolSummary_DF']['cost'].sum(), 0)):,}", f"Cost ({local_currency})")]
            st.markdown(get_stats_row_html(stats), unsafe_allow_html=True)
        
        with col_map:
            """ Instantiate map with warehouse """
            # roads = st.checkbox('Display dispatches using road network (approximate)?', key='002')
            m = folium.Map(location = [scenario['Facility_DF'].iloc[0].latitude, scenario['Facility_DF'].iloc[0].longitude], zoom_start = 10)
            folium.Marker([scenario['Facility_DF'].iloc[0].latitude, scenario['Facility_DF'].iloc[0].longitude], tooltip = scenario['Facility_DF'].iloc[0].facility).add_to(m)
            for _, route in scenario['SolSummary_DF'].iterrows(): 
                route_no = int(route['route'].split(' ')[-1])
                m = draw_route_in_map(m, scenario['SolDetail_DF'][scenario['SolDetail_DF']["route"] == route["route"]], colors(route_no-1), "dash"
                                      #,roads
                                      )
            folium_static(m, width = 800)
            #Need to add the download option here!
            if st.button("Save Map", key = 'Overall'):
                m.save(f"./data/{session_state.country}/{scenario['Scenario']}_Routes.html")
                st.markdown(f"**Map saved as an HTML: {scenario['Scenario']}_Routes.html**")
        # st.dataframe(scenario["SolDetail_DF"])

    with st.expander("Dispatch Details", expanded=False):
        for _, route in scenario['SolSummary_DF'].iterrows(): 
            display_route(route, 
                            scenario['SolDetail_DF'][scenario['SolDetail_DF']["route"] == route["route"]], 
                            scenario['Loading Plan'][scenario['Loading Plan']["Dispatch No"] == route["route"]]
                            #,roads
                            )
            if st.button("Save Map", key = route):
                m.save(f"./data/{session_state.country}/{scenario['Scenario']}_{route}_Route.html")
                st.markdown(f"**Map saved as an HTML: {scenario['Scenario']}__{route}_Route.html**")
            st.markdown("***")

    if len(scenario["SolMiss_DF"]) > 0: 
        with st.expander("Missed Deliveries", expanded=False):
            missed_df = scenario["SolMiss_DF"][["facility_id", 'facility', 'type', 'vol']]
            missed_df.columns = ['Facility ID', 'Facility', 'Type', 'Volume (m\u00B3)']
            st.table(missed_df)

if __name__ == "__main__":

    from PIL import Image
    import SessionState

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    st.sidebar.title("Routing Tool")
    
    st.image(Image.open('./images/app_banner.jpg'), use_column_width=True)
    st.markdown('<style>' + open('icons.css').read() + '</style>', unsafe_allow_html=True)
    
    session_state = SessionState.get(ref_file = "Reference Tables.xlsx", 
                                     user_file = "", 
                                     scenario_file =  r"./filestore/Scenario BUX72X_v3.xlsx", 
                                     scenario_data = None,
                                     scenario_ver_no = 0,) 
    session_state.scenario_data = read_scenario(session_state.scenario_file, session_state.scenario_ver_no) 
    app_review(session_state)