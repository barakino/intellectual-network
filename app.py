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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random

st.set_page_config(page_title="Intellectual Network Explorer", layout="wide")

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
        name = f"{random.choice(first_names)} {random.choice(last_names)} {chr(65 + (i % 26))}" # Added suffix to ensure unique names for dendrogram
        
        if discipline == 'Literature':
            avg_sent = np.random.normal(15, 3)     
            sentiment = np.random.uniform(-0.8, 0.8) 
        elif discipline == 'Natural Philosophy':
            avg_sent = np.random.normal(26, 4)     
            sentiment = np.random.normal(0.1, 0.15)  
        else: 
            avg_sent = np.random.normal(38, 4)     
            sentiment = np.random.normal(0.0, 0.05)   
            
        data.append({
            'ID': i, 'Name': name, 'City': city, 'Discipline': discipline,
            'Avg_Sentence_Length': round(max(5, avg_sent), 2),
            'Sentiment_Polarity': round(sentiment, 3),
            'Wealth_Index': round(np.random.uniform(1, 100), 2)
        })
    nodes_df = pd.DataFrame(data)

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
st.markdown("""
    <style>
    .main { background-color: #f5f1e6; }
    </style>
    """, unsafe_allow_html=True)

st.title("📜 19th-Century Intellectual Network Explorer")
st.sidebar.header("Settings & Tools")

df_nodes, df_edges = load_data()
features = ['Avg_Sentence_Length', 'Sentiment_Polarity', 'Wealth_Index']

viz_type = st.sidebar.selectbox("Select Visualization", ["Network Graph", "PCA Projection", "t-SNE Projection", "Dendrogram"])

# UPDATED: Allow selecting any valid column for coloring
color_options = ['Discipline', 'City', 'Avg_Sentence_Length', 'Sentiment_Polarity', 'Wealth_Index']
color_by = st.sidebar.selectbox("Color Nodes By", color_options)

# Helper to determine if the selected color is categorical or continuous
is_categorical = color_by in ['Discipline', 'City']

# --- 3. Visualization Logic ---
if viz_type in ["PCA Projection", "t-SNE Projection"]:
    x = StandardScaler().fit_transform(df_nodes[features])
    
    if viz_type == "PCA Projection":
        components = PCA(n_components=2).fit_transform(x)
    else:
        components = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(x)
        
    viz_df = pd.DataFrame(components, columns=['Dim 1', 'Dim 2'])
    viz_df = pd.concat([viz_df, df_nodes], axis=1)
    
    # Plotly Express handles continuous/categorical natively
    fig = px.scatter(viz_df, x='Dim 1', y='Dim 2', color=color_by,
                     hover_name='Name', hover_data=['City', 'Discipline'],
                     title=f"{viz_type} (Colored by {color_by})", template="none")
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Network Graph":
    G = nx.from_pandas_edgelist(df_edges, 'Source', 'Target', ['Weight'])
    pos = nx.spring_layout(G, k=0.15, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color = [], [], [], []
    
    # Generate node positions and text
    for node in G.nodes():
        x_pos, y_pos = pos[node]
        node_x.append(x_pos)
        node_y.append(y_pos)
        node_info = df_nodes[df_nodes['ID'] == node].iloc[0]
        node_text.append(f"<b>{node_info['Name']}</b><br>{color_by}: {node_info[color_by]}")
        node_color.append(node_info[color_by])

    # UPDATED: Handle Plotly Network Graph Coloring (Discrete vs Continuous)
    if is_categorical:
        # Convert categorical labels to numeric indices for Plotly's coloring engine
        unique_labels = list(set(node_color))
        color_indices = [unique_labels.index(c) for c in node_color]
        marker_dict = dict(showscale=False, size=12, color=color_indices, colorscale='Set1', line=dict(width=1, color='white'))
    else:
        # Map continuous variables to a Viridis color gradient
        marker_dict = dict(showscale=True, colorscale='Viridis', size=12, color=node_color, 
                           colorbar=dict(title=color_by, thickness=15), line=dict(width=1, color='white'))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, marker=marker_dict
    )

    fig = go.Figure(data=[edge_trace, node_trace],
         layout=go.Layout(title=f'Interactive Network (Colored by {color_by})', showlegend=False, 
         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
         plot_bgcolor='#f5f1e6', paper_bgcolor='#f5f1e6')
    )
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Dendrogram":
    st.subheader(f"Hierarchical Clustering (Leaves colored by {color_by})")
    linked = linkage(df_nodes[features], 'ward')
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#f5f1e6')
    ax.set_facecolor('#f5f1e6')
    
    d = dendrogram(linked, labels=df_nodes['Name'].values, ax=ax, leaf_rotation=90, leaf_font_size=8)
    
    # UPDATED: Color Dendrogram Leaves based on User Selection
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
        # Normalize continuous values for the colormap
        min_val = df_nodes[color_by].min()
        max_val = df_nodes[color_by].max()
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        cmap = cm.viridis
        
        for lbl in xlbls:
            name = lbl.get_text()
            val = df_nodes[df_nodes['Name'] == name][color_by].values[0]
            lbl.set_color(cmap(norm(val)))
            
    st.pyplot(fig)
