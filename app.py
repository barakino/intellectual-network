import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random

# Ensure the app starts wide and the sidebar starts open
st.set_page_config(
    page_title="Intellectual Network Explorer", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 1. Embedded Data Generation ---
@st.cache_data
def load_data():
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    n_total = 120
    disciplines = ['Literature', 'Natural Philosophy', 'Political Economy']
    cities = ['Paris'] * 60 + ['London'] * 60
    
    first_names = ["Arthur", "Émile", "Julian", "Clara", "Beatrix", "Silas", "Léonide", "Percival", "Henrietta", "Gaston", "Adelaide", "Victorine", "Theodore", "Florence", "Auguste", "Leopold"]
    last_names = ["Penhaligon", "Rougier", "Sterling", "Montrose", "von Essen", "Thorne", "Barrat", "Finch", "Vance", "Moretti", "Hugo", "Lyell", "Babbage", "Mill", "Ricardo", "Sand"]
    
    data = []
    for i in range(n_total):
        city = cities[i]
        discipline = random.choice(disciplines)
        name = f"{random.choice(first_names)} {random.choice(last_names)} {chr(65 + (i % 26))}" 
        
        age = np.random.randint(25, 85)
        social_class = np.clip(np.random.normal(50, 20), 1, 100)
        wealth = np.clip(np.random.normal(social_class, 15), 1, 100)
        
        if discipline == 'Literature':
            avg_sent = np.random.normal(15, 3)     
            sentiment = np.random.uniform(-0.8, 0.8)
            pubs = np.random.poisson(10)
            travel = np.random.poisson(5)
            radicalism = np.clip(np.random.normal(0.7, 0.2), 0, 1)
            patronage = np.random.exponential(2000)
            letters = np.random.poisson(40)
            vocab = np.clip(np.random.normal(0.85, 0.05), 0, 1)
            speeches = np.random.poisson(5)
            polymath = np.clip(np.random.normal(0.4, 0.2), 0, 1)
            censorship = np.random.poisson(2)
            tenure = np.clip(np.random.normal(2, 5), 0, 40)
            
        elif discipline == 'Natural Philosophy':
            avg_sent = np.random.normal(26, 4)     
            sentiment = np.random.normal(0.1, 0.15)  
            pubs = np.random.poisson(25)
            travel = np.random.poisson(2)
            radicalism = np.clip(np.random.normal(0.3, 0.15), 0, 1)
            patronage = np.random.exponential(500)
            letters = np.random.poisson(20)
            vocab = np.clip(np.random.normal(0.6, 0.1), 0, 1)
            speeches = np.random.poisson(15)
            polymath = np.clip(np.random.normal(0.8, 0.15), 0, 1)
            censorship = np.random.poisson(0.2)
            tenure = np.clip(np.random.normal(25, 10), 0, 40)
            
        else: # Political Economy
            avg_sent = np.random.normal(38, 4)     
            sentiment = np.random.normal(0.0, 0.05)   
            pubs = np.random.poisson(8)
            travel = np.random.poisson(1)
            radicalism = np.clip(np.random.normal(0.5, 0.25), 0, 1)
            patronage = np.random.exponential(100)
            letters = np.random.poisson(60)
            vocab = np.clip(np.random.normal(0.5, 0.1), 0, 1)
            speeches = np.random.poisson(30)
            polymath = np.clip(np.random.normal(0.5, 0.2), 0, 1)
            censorship = np.random.poisson(1)
            tenure = np.clip(np.random.normal(15, 12), 0, 40)
            
        data.append({
            'ID': i, 'Name': name, 'City': city, 'Discipline': discipline,
            'Age': int(age), 'Social_Class_Index': round(social_class, 1), 'Wealth_Index': round(wealth, 1),
            'Avg_Sentence_Length': round(max(5, avg_sent), 1), 'Sentiment_Polarity': round(sentiment, 2),
            'Publications_Count': int(pubs), 'Travel_Frequency': int(travel), 'Radicalism_Score': round(radicalism, 2),
            'Patronage_Income': int(patronage), 'Correspondence_Volume': int(letters), 'Vocabulary_Richness': round(vocab, 2),
            'Public_Speeches': int(speeches), 'Polymath_Score': round(polymath, 2), 'Censorship_Incidents': int(censorship),
            'University_Tenure': int(tenure)
        })
    
    nodes_df = pd.DataFrame(data)
    base_cols = ['ID', 'Name', 'City', 'Discipline']
    numeric_cols = [col for col in nodes_df.columns if col not in base_cols]
    random.shuffle(numeric_cols)
    nodes_df = nodes_df[base_cols + numeric_cols]

    edges = []
    INTRA_CITY_PROB = 0.12
    INTER_CITY_PROB = 0.003 

    for i in range(n_total):
        for j in range(i + 1, n_total):
            node_i = nodes_df.iloc[i]
            node_j = nodes_df.iloc[j]
            connected = False
            
            if node_i['City'] == node_j['City']:
                if node_i['City'] == 'Paris':
                    if random.random() < INTRA_CITY_PROB: connected = True
                else: 
                    if node_i['Discipline'] == node_j['Discipline']:
                        if random.random() < INTRA_CITY_PROB: connected = True
            elif random.random() < INTER_CITY_PROB: connected = True

            if connected:
                edges.append({'Source': node_i['ID'], 'Target': node_j['ID'], 'Weight': np.random.randint(5, 100)})

    edges_df = pd.DataFrame(edges)
    return nodes_df, edges_df

# --- 2. App UI & Styling ---
# A safer, less aggressive CSS approach that respects Streamlit's core layout
st.markdown("""
    <style>
    /* Base App Colors */
    .stApp {
        background-color: #f5f1e6;
    }
    
    /* Ensure all main text is dark */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp span {
        color: #1a1a1a !important;
    }
    
    /* Style the Sidebar Container explicitly but safely */
    section[data-testid="stSidebar"] {
        background-color: #e3dcc9 !important;
        border-right: 1px solid #c9c1ae;
    }
    
    /* Fix Selectbox Input Area (Closed State) */
    div[data-baseweb="select"] > div {
        background-color: #f5f1e6 !important;
        border-color: #8c7b6c !important;
    }
    
    /* Fix Selectbox Text (Closed State) */
    div[data-baseweb="select"] span {
        color: #1a1a1a !important;
    }
    
    /* Fix Selectbox Dropdown Icon */
    div[data-baseweb="select"] svg {
        fill: #1a1a1a !important;
    }
    
    /* Fix Dropdown Menu Popover (Open State) */
    ul[role="listbox"] {
        background-color: #f5f1e6 !important;
    }
    li[role="option"] span {
        color: #1a1a1a !important;
    }
    li[role="option"]:hover {
        background-color: #e3dcc9 !important;
    }

    /* Custom Story Text Class */
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

st.title("📜 The 19th-Century Intellectual Network")

st.markdown("""
<div class="story-text">
    <p><b>The Year is 1888.</b> Across the English Channel, two vastly different intellectual worlds have taken shape. In the hazy, absinthe-soaked cafés of <b>Paris</b>, a chaotic salon culture thrives. Here, romantic poets passionately debate with stoic political economists and natural philosophers—ideas flowing freely across disciplines regardless of one's academic background.</p>
    <p>Meanwhile, behind the heavy mahogany doors of <b>London's</b> exclusive gentlemen's clubs, a different society operates. The British thinkers are strictly siloed. Natural philosophers speak only to other natural philosophers, and economists associate solely with their own kind. It is an era of rigid categorization, where crossing the aisle is a social faux pas.</p>
    <p><b>Your Mission:</b> You have intercepted a cache of biographical ledgers and postal records detailing the lives and correspondences of 120 prominent thinkers. The records are messy; 15 different numerical traits—from their wealth and radicalism scores to the average length of their sentences—have been mixed together randomly. Your target is to act as a historical data detective. Use the mathematical tools on the left to cut through the noise, discover which hidden variables truly define these disciplines, and visually prove the stark contrast between the Parisian salon and the London silo.</p>
</div>
""", unsafe_allow_html=True)

# Build Sidebar Controls
st.sidebar.header("Settings & Tools")

df_nodes, df_edges = load_data()
features = [col for col in df_nodes.columns if col not in ['ID', 'Name', 'City', 'Discipline']]

viz_type = st.sidebar.selectbox(
    "Select Visualization", 
    ["Network Graph", "PCA Projection", "t-SNE Projection", "UMAP Projection", "Dendrogram"]
)

color_options = ['Discipline', 'City'] + features
random.Random(42).shuffle(color_options) 
color_by = st.sidebar.selectbox("Color Nodes By", color_options)

is_categorical = color_by in ['Discipline', 'City']

# --- 3. Visualization Logic ---
if viz_type == "PCA Projection":
    x = StandardScaler().fit_transform(df_nodes[features])
    
    pca = PCA(n_components=3)
    components = pca.fit_transform(x)
    viz_df = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
    viz_df = pd.concat([viz_df, df_nodes], axis=1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("PCA Axes Selection")
    pc_options = ['PC1', 'PC2', 'PC3']
    x_ax = st.sidebar.selectbox("X-Axis", pc_options, index=0)
    y_ax = st.sidebar.selectbox("Y-Axis", pc_options, index=1)
    
    var_x = pca.explained_variance_ratio_[int(x_ax[-1])-1] * 100
    var_y = pca.explained_variance_ratio_[int(y_ax[-1])-1] * 100
    
    fig = px.scatter(viz_df, x=x_ax, y=y_ax, color=color_by,
                     hover_name='Name', hover_data=['City', 'Discipline'] + features[:3], 
                     title=f"PCA: {x_ax} vs {y_ax}", template="none")
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(font=dict(color='#1a1a1a'), plot_bgcolor='#f5f1e6', paper_bgcolor='#f5f1e6')
    fig.update_xaxes(title_text=f"{x_ax} ({var_x:.1f}% Variance)")
    fig.update_yaxes(title_text=f"{y_ax} ({var_y:.1f}% Variance)")
    st.plotly_chart(fig, use_container_width=True)

elif viz_type in ["t-SNE Projection", "UMAP Projection"]:
    x = StandardScaler().fit_transform(df_nodes[features])
    
    if viz_type == "t-SNE Projection":
        components = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(x)
        title_suffix = ""
    else: 
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        components = reducer.fit_transform(x)
        title_suffix = " (Preserves Global & Local Structure)"
        
    viz_df = pd.DataFrame(components, columns=['Dim 1', 'Dim 2'])
    viz_df = pd.concat([viz_df, df_nodes], axis=1)
    
    fig = px.scatter(viz_df, x='Dim 1', y='Dim 2', color=color_by,
                     hover_name='Name', hover_data=['City', 'Discipline'] + features[:3], 
                     title=f"{viz_type}{title_suffix}", template="none")
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(font=dict(color='#1a1a1a'), plot_bgcolor='#f5f1e6', paper_bgcolor='#f5f1e6')
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Network Graph":
    G = nx.from_pandas_edgelist(df_edges, 'Source', 'Target', ['Weight'])
    pos = nx.spring_layout(G, k=0.18, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.8, color='rgba(136, 136, 136, 0.4)'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color = [], [], [], []
    
    for node in G.nodes():
        x_pos, y_pos = pos[node]
        node_x.append(x_pos)
        node_y.append(y_pos)
        node_info = df_nodes[df_nodes['ID'] == node].iloc[0]
        node_text.append(f"<b>{node_info['Name']}</b><br>{color_by}: {node_info[color_by]}")
        node_color.append(node_info[color_by])

    if is_categorical:
        unique_labels = list(set(node_color))
        palette = px.colors.qualitative.Plotly 
        color_strings = [palette[unique_labels.index(c) % len(palette)] for c in node_color]
        marker_dict = dict(showscale=False, size=14, color=color_strings, line=dict(width=1.5, color='white'))
    else:
        marker_dict = dict(showscale=True, colorscale='Viridis', size=14, color=node_color, 
                           colorbar=dict(title=color_by, thickness=15, tickfont=dict(color='#1a1a1a')), 
                           line=dict(width=1.5, color='white'))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, marker=marker_dict
    )

    fig = go.Figure(data=[edge_trace, node_trace],
         layout=go.Layout(
             title=dict(text=f'Interactive Network (Colored by {color_by})', font=dict(color='#1a1a1a')), 
             showlegend=False, 
             font=dict(color='#1a1a1a'),
             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
             plot_bgcolor='#f5f1e6', paper_bgcolor='#f5f1e6',
             margin=dict(l=0, r=0, t=40, b=0)
         )
    )
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Dendrogram":
    st.subheader(f"Hierarchical Clustering (Leaves colored by {color_by})")
    
    linked = linkage(df_nodes[features], 'ward')
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#f5f1e6')
    ax.set_facecolor('#f5f1e6')
    
    d = dendrogram(linked, labels=df_nodes['Name'].values, ax=ax, leaf_rotation=90, leaf_font_size=8)
    
    xlbls = ax.get_xmajorticklabels()
    
    if is_categorical:
        unique_vals = df_nodes[color_by].unique()
        cmap = plt.get_cmap('tab10')
        cat_color_map = {val: cmap(i) for i, val in enumerate(unique_vals)}
        
        for lbl in xlbls:
            name = lbl.get_text()
            val = df_nodes[df_nodes['Name'] == name][color_by].values[0]
            lbl.set_color(cat_color_map[val])
    else:
        min_val = df_nodes[color_by].min()
        max_val = df_nodes[color_by].max()
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        cmap = cm.viridis
        
        for lbl in xlbls:
            name = lbl.get_text()
            val = df_nodes[df_nodes['Name'] == name][color_by].values[0]
            lbl.set_color(cmap(norm(val)))
            
    ax.tick_params(axis='y', colors='#1a1a1a')
    ax.spines['bottom'].set_color('#1a1a1a')
    ax.spines['left'].set_color('#1a1a1a') 
            
    st.pyplot(fig)
