"""
Shared sidebar styling for all Streamlit pages.
Provides consistent sidebar appearance across the application.
"""

import streamlit as st

def apply_sidebar_style():
    """
    Apply consistent sidebar styling across all pages.
    - Reduces sidebar width by 15% (from ~245px to ~208px)
    - Reduces font sizes for better space utilization
    - Makes sidebar elements more compact
    """
    st.markdown("""
        <style>
        /* Reduce sidebar width to ~208px (15% smaller than default) */
        section[data-testid="stSidebar"] {
            width: 208px !important;
            min-width: 208px !important;
        }
        
        /* Adjust main content to use more space */
        .main > div {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* Reduce global top spacing without negative offsets */
        div[data-testid="stAppViewContainer"] > .main {
            padding-top: 0.5rem !important;
        }

        div[data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 0.5rem !important;
            margin-top: 0 !important;
        }

        div[data-testid="stAppViewContainer"] > .main .block-container > div:first-child {
            padding-top: 0.25rem !important;
            margin-top: 0 !important;
        }

        div[data-testid="stAppViewContainer"] h1:first-child,
        div[data-testid="stAppViewContainer"] h2:first-child {
            margin-top: 0.25rem !important;
        }
        
        /* Reduce font size in sidebar */
        section[data-testid="stSidebar"] * {
            font-size: 0.9rem !important;
        }
        
        /* Page navigation links - make them smaller */
        section[data-testid="stSidebar"] a[href] {
            font-size: 0.85rem !important;
        }
        
        /* Streamlit's page navigation list items */
        section[data-testid="stSidebar"] ul[data-testid="stSidebarNavItems"] li {
            margin-bottom: 0.2rem !important;
        }
        
        section[data-testid="stSidebar"] ul[data-testid="stSidebarNavItems"] span {
            font-size: 0.85rem !important;
        }
        
        /* Target the page navigation specifically */
        section[data-testid="stSidebarNav"] a,
        section[data-testid="stSidebarNav"] span {
            font-size: 0.85rem !important;
        }
        
        /* Make page link container more compact */
        section[data-testid="stSidebarNav"] > ul {
            gap: 0.2rem !important;
        }
        
        section[data-testid="stSidebarNav"] li {
            margin: 0.1rem 0 !important;
            padding: 0.3rem 0.25rem !important;
        }
        
        /* Reduce left padding for navigation items */
        section[data-testid="stSidebarNav"] {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        
        /* Sidebar headers - slightly larger but still reduced */
        section[data-testid="stSidebar"] h1 {
            font-size: 1.3rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        section[data-testid="stSidebar"] h2 {
            font-size: 1.1rem !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0.5rem !important;
        }
        
        section[data-testid="stSidebar"] h3 {
            font-size: 1.0rem !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0.5rem !important;
        }
        
        /* Make radio buttons more compact */
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 0.25rem !important;
        }
        
        /* Make selectboxes more compact */
        section[data-testid="stSidebar"] .stSelectbox > div {
            margin-bottom: 0.5rem !important;
        }
        
        section[data-testid="stSidebar"] .stSelectbox label {
            font-size: 0.9rem !important;
            margin-bottom: 0.25rem !important;
        }
        
        /* Make multiselect more compact */
        section[data-testid="stSidebar"] .stMultiSelect > div {
            margin-bottom: 0.5rem !important;
        }
        
        section[data-testid="stSidebar"] .stMultiSelect label {
            font-size: 0.9rem !important;
            margin-bottom: 0.25rem !important;
        }
        
        /* Reduce padding in sidebar - half of original */
        section[data-testid="stSidebar"] > div:first-child {
            padding: 1rem 0.5rem 2rem 0.5rem !important;
        }
        
        /* Also reduce padding for inner containers */
        section[data-testid="stSidebar"] .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        
        /* Reduce indent for all sidebar content */
        section[data-testid="stSidebar"] .element-container {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        
        /* Make date input more compact */
        section[data-testid="stSidebar"] .stDateInput > div {
            margin-bottom: 0.5rem !important;
        }
        
        section[data-testid="stSidebar"] .stDateInput label {
            font-size: 0.9rem !important;
            margin-bottom: 0.25rem !important;
        }
        
        /* Make sliders more compact */
        section[data-testid="stSidebar"] .stSlider > div {
            margin-bottom: 0.5rem !important;
        }
        
        section[data-testid="stSidebar"] .stSlider label {
            font-size: 0.9rem !important;
            margin-bottom: 0.25rem !important;
        }
        
        /* Reduce gap between sidebar elements */
        section[data-testid="stSidebar"] .element-container {
            margin-bottom: 0.5rem !important;
        }
        
        /* Make checkbox more compact */
        section[data-testid="stSidebar"] .stCheckbox {
            margin-bottom: 0.5rem !important;
        }
        
        section[data-testid="stSidebar"] .stCheckbox label {
            font-size: 0.9rem !important;
        }
        
        /* Button styling in sidebar */
        section[data-testid="stSidebar"] .stButton button {
            font-size: 0.9rem !important;
            padding: 0.25rem 0.5rem !important;
        }
        
        /* Info/warning/error messages in sidebar */
        section[data-testid="stSidebar"] .stAlert {
            font-size: 0.85rem !important;
            padding: 0.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
