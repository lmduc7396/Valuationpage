import streamlit as st

def apply_google_font():
    """Apply Montserrat Google Font to the current Streamlit page."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Montserrat', sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
        }
        
        .stMarkdown {
            font-family: 'Montserrat', sans-serif;
        }
        
        /* Additional selectors for complete coverage */
        .stButton > button {
            font-family: 'Montserrat', sans-serif;
        }
        
        .stSelectbox > div > div {
            font-family: 'Montserrat', sans-serif;
        }
        
        .stTextInput > div > div > input {
            font-family: 'Montserrat', sans-serif;
        }
        
        .stTextArea > div > div > textarea {
            font-family: 'Montserrat', sans-serif;
        }
        
        div[data-baseweb="select"] {
            font-family: 'Montserrat', sans-serif;
        }
        
        /* Sidebar */
        .css-1d391kg {
            font-family: 'Montserrat', sans-serif;
        }
        
        /* Data tables */
        .dataframe {
            font-family: 'Montserrat', sans-serif;
        }
        
        /* Metrics */
        [data-testid="metric-container"] {
            font-family: 'Montserrat', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)