# --- 2. App UI & Styling ---
st.markdown("""
    <style>
    /* HIDE TOP-RIGHT MENU BUT KEEP SIDEBAR TOGGLE ALIVE */
    [data-testid="stHeader"] { 
        background-color: transparent !important; 
    }
    [data-testid="stToolbar"] { 
        visibility: hidden !important; /* Hides Deploy, Star, and Menu */
    }
    footer { visibility: hidden !important; }

    /* Ensure the sidebar toggle arrow is dark and visible */
    [data-testid="collapsedControl"] {
        color: #1a1a1a !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #1a1a1a !important;
    }

    /* Force high contrast text on main page */
    .stApp { background-color: #f5f1e6; color: #1a1a1a !important; }
    h1, h2, h3, h4, p, span, label { color: #1a1a1a !important; }
    
    /* Target the Sidebar explicitly to ensure a light background */
    [data-testid="stSidebar"] {
        background-color: #e3dcc9 !important; 
    }
    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    
    /* FIX: The CLOSED state of the dropdown */
    div[data-baseweb="select"] > div {
        background-color: #f5f1e6 !important; 
        border-color: #8c7b6c !important; 
        color: #1a1a1a !important;
    }
    div[data-baseweb="select"] span {
        color: #1a1a1a !important; 
    }
    div[data-baseweb="select"] svg {
        fill: #1a1a1a !important; 
    }
    
    /* FIX: The OPEN state of the dropdown menus (popovers) */
    [data-baseweb="popover"] > div {
        background-color: #f5f1e6 !important;
    }
    ul[role="listbox"] {
        background-color: #f5f1e6 !important;
    }
    li[role="option"] {
        background-color: transparent !important;
    }
    li[role="option"] span {
        color: #1a1a1a !important;
    }
    li[role="option"]:hover {
        background-color: #e3dcc9 !important;
    }
    
    .story-text { 
        font-size: 1.15rem; 
        color: #1a1a1a !important; 
        line-height: 1.6; 
        font-family: 'Georgia', serif; 
        margin-bottom: 25px; 
        border-left: 4px solid #8c7b6c; 
        padding-left: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
