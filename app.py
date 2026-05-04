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
import random

st.set_page_config(page_title="Intellectual Network Explorer", layout="wide")

# --- 1. Embedded Data Generation (Cached for speed) ---
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
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
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

# Load the data automatically
df_nodes, df_edges = load_data()
features = ['Avg_Sentence_Length', 'Sentiment_Polarity', 'Wealth_Index']

# Sidebar Controls
viz_type = st.sidebar.selectbox("Select Visualization", ["Network Graph", "PCA Projection", "t-SNE Projection", "Dendrogram"])
color_by = st.sidebar.selectbox("Color Nodes By", ["Discipline", "City"])

# --- 3. Visualization Logic ---
if viz_type in ["PCA Projection", "t-SNE Projection"]:
    x = StandardScaler().fit_transform(df_nodes[features])
    
    if viz_type == "PCA Projection":
        components = PCA(n_components=2).fit_transform(x)
    else:
        components = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(x)
        
    viz_df = pd.DataFrame(components, columns=['Dim 1', 'Dim 2'])
    viz_df = pd.concat([viz_df, df_nodes], axis=1)
    
    fig = px.scatter(viz_df, x='Dim 1', y='Dim 2', color=color_by,
                     hover_name='Name', hover_data=['City', 'Discipline'],
                     title=viz_type, template="none")
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
    
    # Create a color map for dynamic coloring
    unique_labels = df_nodes[color_by].unique()
    colors = px.colors.qualitative.Plotly[:len(unique_labels)]
    color_map = dict(zip(unique_labels, colors))

    for node in G.nodes():
        x_pos, y_pos = pos[node]
        node_x.append(x_pos)
        node_y.append(y_pos)
        node_info = df_nodes[df_nodes['ID'] == node].iloc[0]
        node_text.append(f"<b>{node_info['Name']}</b><br>{node_info['City']} | {node_info['Discipline']}")
        node_color.append(color_map[node_info[color_by]])

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(showscale=False, size=12, color=node_color, line=dict(width=1, color='white'))
    )

    fig = go.Figure(data=[edge_trace, node_trace],
         layout=go.Layout(title=f'Interactive Network (Colored by {color_by})', showlegend=False, 
         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
         plot_bgcolor='#f5f1e6', paper_bgcolor='#f5f1e6')
    )
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Dendrogram":
    st.subheader("Hierarchical Clustering based on Language & Sentiment")
    linked = linkage(df_nodes[features], 'ward')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#f5f1e6')
    ax.set_facecolor('#f5f1e6')
    dendrogram(linked, labels=df_nodes['Name'].values, ax=ax, leaf_rotation=90, leaf_font_size=8)
    st.pyplot(fig)