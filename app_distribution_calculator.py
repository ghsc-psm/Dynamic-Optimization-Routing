import streamlit as st
import pandas as pd
import os
from math import ceil
import traceback
from zipfile import ZipFile
import base64

from ortools.linear_solver import pywraplp

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def estimate_volume_opt(row, product_volume): 
    if row['SKU'] in product_volume: 
        unit_vol = product_volume[row['SKU']]
        solver = pywraplp.Solver('Estimator', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        x = {unit:solver.IntVar(0.0, ceil(row['Qty Modified']/unit), f"{unit}") for unit in unit_vol}    
        solver.Add(sum(unit*x[unit] for unit in unit_vol) >= row['Qty Modified'])
        solver.Minimize(sum(unit_vol[unit]*x[unit] for unit in unit_vol))
        if solver.Solve() == pywraplp.Solver.OPTIMAL: 
            return solver.Objective().Value()/1e9

def estimate_volume_each(row, product_volume): 
    if row['SKU'] in product_volume: 
        return row['Qty Modified']*product_volume[row['SKU']][1]/1e9

@st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
def load_data(order_file, master_file): 
    try: 
        master_vol_df = pd.read_excel(master_file, sheet_name='Volumetric')
        master_truck_df = pd.read_excel(master_file, sheet_name='Truck')
    except Exception as e:
        st.error(f"Master Data Loading Error: {e}\n{traceback.format_exc()}")
        st.stop()

    master_truck_df = master_truck_df[master_truck_df['Capacity (m3)'] > 0]
    trucks = {row['Truck Type']: (row['Capacity (m3)'], row['Cost']) for _, row in master_truck_df.iterrows()}

    master_vol_df['SKU'] = master_vol_df['SKU'].apply(lambda s: s.strip().upper())
    master_vol_df = master_vol_df[master_vol_df['Volume'] > 1]

    try: 
        if order_file[-4:].upper() == 'XLSX': 
            order_df = pd.read_excel(order_file)
            order_df = order_df[['SKU', 'SKU Description', 'Inventory Status', 'Original Qty', 'Qty Modified']]
            order_df['SKU'] = order_df['SKU'].apply(lambda s: s.strip().upper())
        elif order_file[-3:].upper() == 'ZIP':
            df_list = []
            files_skipped = []
            with ZipFile(order_file, 'r') as zip:
                for f in zip.namelist(): 
                    if f[-3:].upper() == 'CSV': 
                        try: 
                            df = pd.read_csv(zip.open(f, 'r'), sep='\t', encoding='UTF-16')
                        except Exception as e: 
                            try: 
                                df = pd.read_csv(zip.open(f, 'r'), sep=',', encoding='UTF-8')
                            except Exception as e:
                                st.error("Unknow format. Files zipped are expected to be UTF-16 with tab separator or UTF-8 with comma separator")
                                st.stop()
                                
                        if 'SKU' not in df.columns or 'Qty Modified' not in df.columns: 
                            files_skipped.append(f)
                            continue
                        df['FILE'] = f
                        df['SKU'] = df['SKU'].apply(lambda s: s.replace('=', '').replace('"', '').strip().upper())
                        df['SKU Description'] = df['SKU Description'].apply(lambda s: s.replace('=', '').replace('"', '').strip())
                        df['Inventory Status'] = df['Inventory Status'].apply(lambda s: s.replace('=', '').replace('"', '').strip())
                        df = df[['FILE', 'SKU', 'SKU Description', 'Inventory Status', 'Original Qty', 'Qty Modified']]
                        df_list.append(df)
            
            order_df = pd.concat(df_list, axis=0) if len(df_list) > 0 else None
            assert order_df is not None
            st.warning(f"{len(files_skipped)} files were skipped due to inconsistent format")
    except Exception as e:
        st.error(f"Order Data Loading Error: {e}\n{traceback.format_exc()}")
        st.stop()
    
    # order_df = order_df[order_df['Qty Modified'] > 0]

    product_volume = {}
    for _, row in master_vol_df.iterrows(): 
        sku = row['SKU']
        uom = row['Units of Measure (UOM)'].strip().upper()
        unit = 1 if uom == 'EACH' else int(uom.replace('CASE', '')) if 'CASE' in uom else None
        if unit is not None and unit > 0: 
            if sku not in product_volume: 
                product_volume[sku] = {}
            product_volume[sku][unit] = row['Volume']

    for sku in product_volume: 
        if 1 not in product_volume[sku]: 
            product_volume[sku][1] = max(product_volume[sku][unit]/unit for unit in product_volume[sku])

    order_df['Total Volume OPT'] = order_df.apply(estimate_volume_opt, args=(product_volume,), axis=1 )
    order_df['Total Volume EACH'] = order_df.apply(estimate_volume_each, args=(product_volume,), axis=1 )
    order_df['Total Volume EST'] = (2*order_df['Total Volume OPT']+order_df['Total Volume EACH'])/3

    return order_df, trucks

def app_distribution_calculator(session_state):

    expand_upload = session_state.calc_order_file == ""
    with st.expander("Upload Data", expanded=expand_upload):
        order_file = st.file_uploader("Order Data", type=["xlsx", "zip"], key="DistCalculator")

        master_file = None
        if st.checkbox("Use an Updated Master File", key='004'): 
            master_file = st.file_uploader("Master Data", type="xlsx", key="DistCalculatorMaster")

        if order_file is not None: 
            if st.button("Upload"): 
                with open(os.path.join("./filestore", order_file.name), 'wb') as f:
                    f.write(order_file.getbuffer())
                    session_state.calc_order_file = os.path.join("./filestore", order_file.name)

                if master_file is not None:
                    with open(os.path.join("./filestore", master_file.name), 'wb') as f:
                        f.write(master_file.getbuffer())
                        session_state.calc_master_file = os.path.join("./filestore", master_file.name)

    if session_state.calc_order_file != "": 
        order_name = os.path.splitext(os.path.basename(session_state.calc_order_file))[0]
        with st.expander(f"Calculator - {order_name}", expanded=True):
            order_df, trucks = load_data(session_state.calc_order_file, session_state.calc_master_file)
            order_m_df = order_df[pd.notna(order_df['Total Volume EACH'])]
            order_u_df = order_df[pd.isna(order_df['Total Volume EACH'])]

            # Weighted volume, 2 for OPT-based, 1 for EACH-based
            total_vol_matched = order_m_df['Total Volume EST'].sum()

            stats_html = f"""
            <table style="width: 100%; border-collapse: collapse; background-color: #ffffff;" border="1">
            <tbody>
            <tr style="background-color: #130441;">
            <td style="width: 33.3333%; text-align: center;">&nbsp;</td>
            <td style="width: 33.3333%; text-align: center;"><strong><span style="color: #ffffff;">Matched</span></strong></td>
            <td style="width: 33.3333%; text-align: center;"><strong><span style="color: #ffffff;">Unmatched</span></strong></td>
            </tr>
            <tr>
            <td style="width: 33.3333%; text-align: center;">Product</td>
            <td style="width: 33.3333%; text-align: center;">{len(order_m_df):,}</td>
            <td style="width: 33.3333%; text-align: center;">{len(order_u_df):,} ({len(order_u_df)/len(order_df):.1%})</td>
            </tr>
            <tr>
            <td style="width: 33.3333%; text-align: center;">Quantity</td>
            <td style="width: 33.3333%; text-align: center;">{order_m_df['Qty Modified'].sum():,}</td>
            <td style="width: 33.3333%; text-align: center;">{order_u_df['Qty Modified'].sum():,} ({order_u_df['Qty Modified'].sum()/(1e-9+order_df['Qty Modified'].sum()):.1%})</td>
            </tr>
            <tr>
            <td style="width: 33.3333%; text-align: center;">Volume (m\u00B3)</td>
            <td style="width: 33.3333%; text-align: center;">{total_vol_matched:.1f}</td>
            <td style="width: 33.3333%; text-align: center;"> N/A </td>
            </tr>
            </tbody>
            </table>
            """

            st.markdown("***")
            col_left, col1, col_mid, col2, col_right = st.columns([0.5, 3, 0.5, 2, 0.5])
            with col1: 
                st.markdown("### Volume Matching & Estimate")
                st.markdown(stats_html, unsafe_allow_html=True)

                vol_factor = 1
                if len(order_u_df) > 0: 
                    um_ratio = order_u_df['Qty Modified'].sum()/(1e-9+order_m_df['Qty Modified'].sum())
                    init_buffer_val = min(1000, int(um_ratio*20)*5)
                    max_value = min(1000, 100*(int(init_buffer_val/100)+1))
                    val = st.slider("Extra Buffer % for Unmatched Products", min_value=0, max_value=max_value, value=init_buffer_val, step=5)
                    vol_factor = 1 + val/100
                    if val != session_state.calc_buffer: 
                        session_state.calc_buffer = val
                        session_state.calc_solution = []

                total_vol = total_vol_matched * vol_factor
                st.markdown(f"### Total {'' if vol_factor==1 else 'Estimated '}Volume: {total_vol:.1f} m\u00B3")
            
            with col2:
                st.markdown("### Select Available Truck Types")
                sel_t_type = {} 
                for t_type in trucks: 
                    sel_t_type[t_type] = st.checkbox(f"{t_type} ({trucks[t_type][0]:.1f} m\u00B3, {trucks[t_type][1]:.2f} ZK/km)", value=True, key=t_type)

                type_sel = [t_type for t_type in sel_t_type if sel_t_type[t_type]]
                if type_sel != session_state.calc_type_sel: 
                    session_state.calc_type_sel = type_sel
                    session_state.calc_solution = []

                if len(type_sel) > 0 and st.button("Optimize"):
                    session_state.calc_solution = []
                    solver = pywraplp.Solver('Calculator', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
                    x = {t_type:solver.IntVar(0.0, ceil(total_vol/(1e-9+trucks[t_type][0])), f"{t_type}") for t_type in type_sel}    
                    solver.Add(sum(trucks[t_type][0]*x[t_type] for t_type in type_sel) >= total_vol)
                    solver.Minimize(sum(trucks[t_type][1]*x[t_type] for t_type in type_sel))
                    status = solver.Solve() 

                    if status == pywraplp.Solver.OPTIMAL:
                        for t_type in type_sel: 
                            if x[t_type].solution_value() > 0: 
                                session_state.calc_solution.append([t_type, trucks[t_type][0], trucks[t_type][1], int(x[t_type].solution_value())])

            st.markdown("***")
            if len(session_state.calc_solution): 
                tbl_html = """
                    <table style="width: 100%; border-collapse: collapse; background-color: #ffffff; height: 68px;" border="1">
                    <tbody>
                    <tr style="background-color: #130441;">
                    <td style="width: 20%; text-align: center; height: 17px;"><strong><span style="color: #ffffff;">Truck Type</span></strong></td>
                    <td style="width: 20%; text-align: center; height: 17px;"><strong><span style="color: #ffffff;">Capacity (m\u00B3)</span></strong></td>
                    <td style="width: 20%; text-align: center; height: 17px;"><strong><span style="color: #ffffff;">Cost (ZM/km)</span></strong></td>
                    <td style="width: 20%; text-align: center;"><span style="color: #ffffff;"><strong># Needed</strong></span></td>
                    <td style="width: 20%; text-align: center; height: 17px;"><strong><span style="color: #ffffff;">Total</span><span style="color: #ffffff;"> (m\u00B3)</span></strong></td>
                    </tr>"""
                total_cap = 0
                total_cnt = 0
                total_cost = 0
                for s in session_state.calc_solution: 
                    tbl_html += f"""
                        <tr style="height: 17px;">
                        <td style="width: 20%; text-align: center; height: 17px;">{s[0]}</td>
                        <td style="width: 20%; text-align: center; height: 17px;">{s[1]:.1f}</td>
                        <td style="width: 20%; text-align: center; height: 17px;">{s[2]:.2f}</td>
                        <td style="width: 20%; text-align: center;">{s[3]:d}</td>
                        <td style="width: 20%; text-align: center; height: 17px;">{s[1]*s[3]:.1f}</td>
                        </tr>"""
                    total_cap += s[1]*s[3]
                    total_cost += s[2]*s[3]
                    total_cnt += s[3]

                tbl_html += f"""
                    <tr style="height: 17px; background-color: #e6e652;">
                    <td style="width: 20%; text-align: center; height: 17px;"><strong>Summary</strong></td>
                    <td style="width: 20%; text-align: center; height: 17px;">&nbsp;</td>
                    <td style="width: 20%; text-align: center; height: 17px;"><strong>{total_cost:.2f}</strong></td>
                    <td style="width: 20%; text-align: center;"><strong>{total_cnt:d}</strong></td>
                    <td style="width: 20%; text-align: center; height: 17px;"><strong>{total_cap:.1f} ({total_vol/total_cap:.1%})</strong></td>
                    </tr>
                    </tbody>
                    </table>"""

                st.markdown("""<p style="text-align: center;"><strong>Optimized Decision Recommendation</strong></p>""", unsafe_allow_html=True)

                col_left, col_sol, col_right = st.columns([1, 8, 1])
                with col_sol: 
                    st.markdown(tbl_html, unsafe_allow_html=True)

                st.text("")

                vol_f = f"./filestore/Volume Status {order_name}.xlsx"
                with pd.ExcelWriter(vol_f) as writer:
                    order_m_df.to_excel(writer, sheet_name="Product Matched", index=False)
                    order_u_df.to_excel(writer, sheet_name="Product Not Matched", index=False)

                st.markdown(get_binary_file_downloader_html(vol_f, 'Volume Satus'), unsafe_allow_html=True)

if __name__ == "__main__":

    import SessionState
    st.set_page_config(layout="wide", initial_sidebar_state='auto')
    st.title("Volumetric Calculator for Distribution Planning")

    session_state = SessionState.get(calc_master_file = r"./data/Masterdata Weights and Volume.xlsx", 
                                     calc_order_file = "", 
                                     calc_solution = [],
                                     calc_buffer=0,
                                     calc_type_sel=[]
                                     ) 

    app_distribution_calculator(session_state)
