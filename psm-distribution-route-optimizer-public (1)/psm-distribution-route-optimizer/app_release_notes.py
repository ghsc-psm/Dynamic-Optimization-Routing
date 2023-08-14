import streamlit as st

ReleaseNotes = [
    ['2021/09/24 v1.0.0.0', [
        "Release at the end of Zambia STTA"
     ]],
    ['2021/09/27 v1.1.1.1', [
         "(N) Release Notes Functionality", 
         "(N) Enable consistent release verison capturing in UI and scenario meta data",
         "(F) Disabled caching for reference data to prevent crashes with reference update",
         "(D) Updated order details to 2021/09/26"
     ]],
    ['2021/09/28 v1.1.2.2', [
         "(F) Avoid crash when no order details are matched",
         "(D) Updated order details to 2021/09/28"
     ]],
    ['2021/10/04 v1.1.3.3', [
         "(F) Improved the download performance, leveraging binary downloads",
         "(D) Updated order details to 2021/10/03",
         "(D) Updated fleet data utilizing different volume factor for trucks"
     ]],
    ['2021/10/07 v1.1.4.3', [
         "(F) Expand recognized order prefix list",
     ]],
    ['2021/10/12 v1.1.5.4', [
         "(F) Facility ID truncated from the order evaluation file due to xlsm file format",
         "(F) Increased the robustness to retrieve orders from the route status file",
         "(D) Updated order details to 2021/10/11",
     ]],
    ['2021/11/09 v1.2.5.5', [
         "(N) Added details of the dispatch to the review view",
         "(N) Adjusted the sort order for the pick wave creation",
         "(D) Updated order details to 2021/11/09",
     ]],
    ['2021/11/11 v1.3.5.6', [
         "(N) Added validations for admin uploading facility mapping and order data",
         "(D) Updated order details to 2021/11/11",
     ]],
    ['2021/11/15 v1.4.5.7', [
         "(N) Display the last updated timestamp for order details",
         "(D) Updated order details to 2021/11/15",
     ]],
    ['2021/11/15 v1.4.6.7', [
         "(F) Fixed missing display in the review page",
     ]],
     ['2022/01/26 v1.4.7.8', [
         "(F) fixed loading of orders from the sheet ",
         "(D) Updated order details to 2022/01/26",
     ]],
]

def app_release_notes(session_state): 
    """Show technical documentation produced with any updates to the application (e.g., recent changes, feature enhancements, or bug fixes).
    Args:
        session_state (object): a new SessionState object 
    """    

    st.header("OPEX Toolkit Release Notes")
    ReleaseNotes.reverse()
    for r_note in ReleaseNotes: 
        with st.expander(f"Release Notes for {r_note[0]}"): 
            for note in r_note[1]: 
                st.markdown(f"- {note}")
