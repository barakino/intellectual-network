# Generate node positions and text
    for node in G.nodes():
        x_pos, y_pos = pos[node]
        node_x.append(x_pos)
        node_y.append(y_pos)
        node_info = df_nodes[df_nodes['ID'] == node].iloc[0]
        node_text.append(f"<b>{node_info['Name']}</b><br>{color_by}: {node_info[color_by]}")
        node_color.append(node_info[color_by])

    # --- FIX APPLIED HERE ---
    if is_categorical:
        # Map each category directly to a valid hex color string
        unique_labels = list(set(node_color))
        palette = px.colors.qualitative.Plotly  # Use Plotly's default discrete palette
        color_strings = [palette[unique_labels.index(c) % len(palette)] for c in node_color]
        
        marker_dict = dict(showscale=False, size=12, color=color_strings, line=dict(width=1, color='white'))
    else:
        # Map continuous variables to a Viridis color gradient (Plotly accepts 'Viridis' natively)
        marker_dict = dict(showscale=True, colorscale='Viridis', size=12, color=node_color, 
                           colorbar=dict(title=color_by, thickness=15), line=dict(width=1, color='white'))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, marker=marker_dict
    )
