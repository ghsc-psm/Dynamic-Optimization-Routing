import streamlit as st
import configparser
import re
from psm.db import decrypt

def validate_password_format(password): 
    if len(password) > 6 and len(re.findall(r'[A-Z]', password)) > 0 and len(re.findall(r'[a-z]', password)) > 0 and len(re.findall(r'[0-9]', password)) > 0: 
        return True
    return False

def verify_emails(emails_str): 
    regex = '[^@]+@[^@]+\.[^@]+'        # email regular expression check
    good_emails = []
    for email in [v.strip() for v in emails_str.replace(" ", "").split(";")]:
        if email != '' and not re.search(regex, email):
            return False, email
        else:
            good_emails.append(email)
    return True, ";".join(good_emails)


def check_credentials(session_state):
    st.markdown("### Login")

    cred_name = st.empty()
    cred_pass = st.empty()

    uname = cred_name.text_input("Username (Email):", value = session_state.username)
    upass = cred_pass.text_input("Password:", value="", type="password")


    login = st.button("Login")

    if login:
        read_config = configparser.ConfigParser()
        read_config.read("credentials.ini")
        credentials_dict = dict()
        for (usertype, password) in read_config.items("COUNTRY CREDENTIALS"):
            credentials_dict[decrypt(password)] = usertype

        email_check, email = verify_emails(uname)
        if email_check:
            if upass != "" and not validate_password_format(upass):
                st.error("Password must be longer than 6 characters, and contain at least one upper, one lower and one number.")
                st.stop()
            elif upass in credentials_dict:
                session_state.username = uname
                credentials_list = credentials_dict[upass].split("_")
                session_state.country = credentials_list[0]
                session_state.user_type = credentials_list[1]
                data_path = f'./data/{session_state.country}/'
                session_state.calc_master_file = f"{data_path}{session_state.country}"+ r" DC.xlsx"
                session_state.dro_order_details_file = f"{data_path}{session_state.country}"+ r" DRO Order Details.xlsx"                     
                session_state.dro_facility_mapping_file = f"{data_path}{session_state.country}"+ r" ALL Facility Mapping.xlsx"
                session_state.dro_order_evaluation_template_file = f"{data_path}{session_state.country}"+ r" DRO Order Evaluation Template.xlsm"                                       
                st.experimental_rerun()
            else:
                st.error("Wrong password. Please double-check")
        else:
            st.error("Not a valid email. Please double check")

        
        
        
        
