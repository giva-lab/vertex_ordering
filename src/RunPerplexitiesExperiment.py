import sys
import os
import networkx as nx
import numpy as np
import osmnx as ox
import shapely
import geopandas as gpd
import pandas as pd
import umap
import pickle
import statistics
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs
import osmnx as ox
import math 
from sklearn.preprocessing import StandardScaler
import openTSNE 
import folium
import branca
import branca.colormap as cm
from sklearn.metrics.pairwise import pairwise_distances,haversine_distances
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from src.utils import get_topological_measure, get_topological_measure_optimized
import CityHub
from sklearn.neighbors import NearestNeighbors


# Loading pre-computed city data (if it exists) for 'forward' and 'inverse' mappings.
# This helps to avoid recomputation in future runs.
measures = {perp : {} for perp in [10,25,50,75,100,150,200]}

# Defining EPSG (coordinate reference systems) for the city.
# EPSG codes are used to properly handle geospatial data, ensuring correct map projections.
city = 'São Paulo'
epsg = 'EPSG:31983'


print(city) 
cg = CityHub.CityHub(city)  # Instantiate a CityHub object for the current city
G = cg.city_street_graph  # Retrieve the city’s street graph (nodes and edges)

print("TSNE -----------------------------------")
nodes2d = np.array(cg.city_vert_list)  # Retrieve the 2D coordinates of nodes

# Create a GeoDataFrame for the nodes and reproject to the correct EPSG coordinate system
gpdnodes = gpd.GeoDataFrame(geometry=gpd.points_from_xy(nodes2d[:, 0], nodes2d[:, 1]), crs="EPSG:4326")
gpdnodes = gpdnodes.to_crs(epsg)  # Reproject to the specific city's CRS
nodesproj = gpdnodes.geometry.apply(lambda g: pd.Series(g.coords[0]))  # Extract projected coordinates

# Normalize the projected coordinates using StandardScaler
scaler = StandardScaler()
scaler.fit(nodesproj)
nodesprojsca = scaler.transform(nodesproj)

for perp in measures:
    print("Calculating TSNE with perplexity", perp)
    # Apply t-SNE with 1 component to reduce the 2D node coordinates to 1D
    X_embedding = openTSNE.TSNE(
        n_components=1,
        perplexity=perp,
        n_jobs=30,
        random_state=42,
        verbose=True,

    ).fit(nodesprojsca)


    print("Gerando dataframe")

    nodesdf = pd.DataFrame(nodes2d)  # Create a DataFrame with the node coordinates
    nodesdf['y'] = nodesproj[0]  # Add the y-coordinates (latitude) to the DataFrame
    nodesdf['x'] = nodesproj[1]  # Add the x-coordinates (longitude) to the DataFrame
    nodesdf["tsne"] = X_embedding  # Store the t-SNE 1D embedding in the DataFrame
    nodesdf = nodesdf.sort_values('tsne').reset_index()  # Sort the DataFrame by t-SNE values and reset index
    nodesdf["new_index"] = nodesdf.index  # Add a new column with the new sorted index

    # Set up a color map to visualize the t-SNE embedding results on a map
    mi = 0
    ma = nodesdf.shape[0] - 1  # Maximum index in the DataFrame
    colormap = cm.LinearColormap(colors=['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red'],
                                    index=[mi, ma/5, ma/5*2, ma/5*3, ma/5*4, ma], vmin=mi, vmax=ma,
                                    caption='t-SNE perplexity 2000')  # Define a color map to show on the map

    # Get the latitude and longitude information for mapping
    Lats = nodes2d[:, 0]  # Extract latitudes
    Longs = nodes2d[:, 1]  # Extract longitudes
    med_lat = 0.5 * (np.max(Lats) + np.min(Lats))  # Calculate the median latitude for centering the map
    med_lon = 0.5 * (np.max(Longs) + np.min(Longs))  # Calculate the median longitude for centering the map

    # Create a folium map centered on the city, zoomed in to level 12
    m = folium.Map(location=[med_lat, med_lon], zoom_start=12, tiles='CartoDB positron')

    # Add CircleMarkers to the map for each node, colored by the t-SNE embedding
    for index, row in nodesdf.iterrows():
        marker = folium.CircleMarker([row[0], row[1]], radius=1, color=colormap(index))
        popup = folium.Popup(str(index))  # Add index as popup text
        marker.add_child(popup)
        m.add_child(marker)  # Add marker to the map

    # Add the color map legend and save the map as an HTML file
    m.add_child(colormap)
    m.save('tsne_perp'+str(perp)+'map_' + city + '.html')

    # Create the inverse mapping for the t-SNE sorted indices
    df = nodesdf.sort_values(by=['index'])  # Sort the DataFrame back by the original index
    tsne_sorted_index_list = np.array(df['new_index'])  # Get the new sorted indices based on t-SNE
    inverse_tsne_sorted_index_list = [0] * len(cg.city_vert_list)
    for i in range(len(tsne_sorted_index_list)):
        inverse_tsne_sorted_index_list[tsne_sorted_index_list[i]] = i  # Inverse map of t-SNE indices

    # Metric for quality: embedding -> latlong (FORWARD METRIC)
    print("Calculating metrics for embedding -> latlong")

    # Define window size as 1% of the total nodes in the city
    window_size = int(len(cg.city_vert_list) / 100)

    # best estimation for normalization purposes
    window_size = int(len(cg.city_vert_list)/100)
    originaldf = df[[0,1,'x','y']].copy()
    originaldf.index = df['index'].values

    def calculate_diagonal(list_of_indices):
        Lats = [cg.city_vert_list[j][0] for j in list_of_indices]
        Longs = [cg.city_vert_list[j][1] for j in list_of_indices]
        min_rad = [math.radians(_) for _ in [min(Lats),min(Longs)]]
        max_rad = [math.radians(_) for _ in [max(Lats),max(Longs)]]
        return haversine_distances([min_rad,max_rad])[0,1]* 6371000/1000

    knn = NearestNeighbors(n_neighbors=window_size, algorithm='auto')
    knn.fit(originaldf[['x','y']])
    neighbors = knn.kneighbors(originaldf[['x','y']], return_distance = False)
    best_diagonal = np.apply_along_axis(calculate_diagonal, 1, neighbors)

    # --- t-SNE Embedding Diagonals ---
    diagonals_tsne = []
    for i in range(0, len(cg.city_vert_list) - window_size, 1):
        Lats = [cg.city_vert_list[j][0] for j in inverse_tsne_sorted_index_list[i:i+window_size]]
        Longs = [cg.city_vert_list[j][1] for j in inverse_tsne_sorted_index_list[i:i+window_size]]
        min_rad = [math.radians(_) for _ in [min(Lats),min(Longs)]]
        max_rad = [math.radians(_) for _ in [max(Lats),max(Longs)]]
        diag = haversine_distances([min_rad,max_rad])[0,1]* 6371000/1000
        norm_diag = diag / best_diagonal[inverse_tsne_sorted_index_list[i + int(window_size/2)]]
        diagonals_tsne.append(norm_diag)

    # Save the forward metrics
    measures[perp]['forward'] = diagonals_tsne

    # Topological Measure
    print("Calculating Metrics for Topological Measure")
    topological_tsne = get_topological_measure_optimized(nx.adjacency_matrix(G), tsne_sorted_index_list)

    # Save the forward metrics
    measures[perp]['topological'] = topological_tsne


    # Metric for quality: latlong -> embedding (INVERSE METRIC)
    print("Calculating metrics for latlong -> embedding")

    max_inverse_ind_dist_tsne = []

    # Loop through each node in the city's vertex list
    for i in range(0, len(cg.city_vert_list)):
        # Calculate the shortest paths from the source node to others within a cutoff distance (0.5 km)
        ps = nx.single_source_dijkstra_path(cg.city_street_graph, cg.city_vert_ind_to_nxind_dict[i], cutoff=0.5 * 1000.0, weight='length')
        result_nodes = [cg.city_vert_nxind_to_ind_dict[k] for k in ps.keys()]
        
        # Get the sorted indices t-SNE
        inds_sorted_tsne = [tsne_sorted_index_list[j] for j in result_nodes]
        
        # Calculate the maximum inverse index distance and normalize by the number of result nodes
        max_inverse_ind_dist_tsne.append((np.max(inds_sorted_tsne) - np.min(inds_sorted_tsne)) / len(inds_sorted_tsne))

    # Save the inverse metrics
    measures[perp]['inverse'] = max_inverse_ind_dist_tsne
    
# Save the forward and inverse metrics as pickle files
with open("perps_"+city+".pkl", "wb") as file:
    pickle.dump(measures, file)