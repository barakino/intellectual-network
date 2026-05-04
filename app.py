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
        name = f"{random.choice(first_names)} {random.choice(last_names)} {chr(65 + (i % 26))}" 
        
        # Base independent features
        age = np.random.randint(25, 85)
        social_class = np.clip(np.random.normal(50, 20), 1, 100)
        wealth = np.clip(np.random.normal(social_class, 15), 1, 100)
        
        # Correlated features based on discipline
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
    
    # Shuffle the numeric columns so the "giveaway" features are hidden among the noise
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
st.markdown("""
    <style>
    .main { background-color: #f5f1e6; }
    .story-text { font-size: 1.15rem; color: #3b3a36; line-height: 1.6; font-family: 'Georgia', serif; margin-bottom: 25px; border-left: 4px solid #8c7b6c; padding-left: 20px;}
    </style>
    """, unsafe_allow_html=True)

st.title("📜 The 19th-Century Intellectual Network")

# --- Narrative Story / Game Target ---
st.markdown("""
<div class="story-text">
    <p><b>The Year is 1888.</b> Across the English Channel, two vastly different intellectual worlds have taken shape. In the hazy, absinthe-soaked cafés of <b>Paris</b>, a chaotic salon culture thrives. Here, romantic poets passionately debate with stoic political economists and natural philosophers—ideas flowing freely across disciplines regardless of one's academic background.</p>
    <p>Meanwhile, behind the heavy mahogany doors of <b>London's</b> exclusive gentlemen's clubs, a different society operates. The British thinkers are strictly siloed. Natural philosophers speak only to other natural philosophers, and economists associate solely with their own kind. It is an era of rigid categorization, where crossing the aisle is a social faux pas.</p>
    <p><b>Your Mission:</b> You have intercepted a cache of biographical ledgers and postal records detailing the lives and correspondences of 120 prominent thinkers. The records are messy; 15 different numerical traits—from their wealth and radicalism scores to the average length of their sentences—have been mixed together randomly. Your target is to act as a historical data detective. Use the mathematical tools on the left to cut through the noise, discover which hidden variables truly define these disciplines, and visually prove the stark contrast between the Parisian salon and the London silo.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Settings & Tools")

df_nodes, df_edges = load_data()
features = [col for col in df_nodes.columns if col not in ['ID', 'Name', 'City', 'Discipline']]

viz_type = st.sidebar.selectbox("Select Visualization", ["Network Graph", "PCA Projection", "t-SNE Projection", "Dendrogram"])

color_options = ['Discipline', 'City'] + features
color_by = st.sidebar.selectbox("Color Nodes By", color_options)

is_categorical = color_by in ['Discipline', 'City']

# --- 3. Visualization Logic ---
if viz_type in ["PCA Projection", "t-SNE Projection"]:
    x = StandardScaler().fit_transform(df_nodes[features])
    
    if viz_type == "PCA Projection":
        pca = PCA(n_components=2)
        components = pca.fit_transform(x)
        var_explained = sum(pca.explained_variance_ratio_) * 100
        title_suffix = f" (Captures {var_explained:.1f}% of variance)"
    else:
        components = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(x)
        title_suffix = ""
        
    viz_df = pd.DataFrame(components, columns=['Dim 1', 'Dim 2'])
    viz_df = pd.concat([viz_df, df_nodes], axis=1)
    
    fig = px.scatter(viz_df, x='Dim 1', y='Dim 2', color=color_by,
                     hover_name='Name', hover_data=['City', 'Discipline'] + features[:3], 
                     title=f"{viz_type}{title_suffix}", template="none")
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
        marker_dict = dict(showscale=False, size=12, color=color_strings, line=dict(width=1, color='white'))
    else:
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
            
    st.pyplot(fig)
