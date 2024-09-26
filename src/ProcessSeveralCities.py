import sys
import os
import CityHub
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


# Loading pre-computed city data (if it exists) for 'forward' and 'inverse' mappings.
# This helps to avoid recomputation in future runs.
if os.path.exists("forward.pkl"):
    with open("forward.pkl", "rb") as file:
        cities_forward = pickle.load(file)

    with open("inverse.pkl", "rb") as file:
        cities_inverse = pickle.load(file)
else:
    # If no pre-computed data is found, initialize empty dictionaries
    cities_forward = {'fiedler': {}, 'tsne': {}, 'umap':{}, 'original':{}, 'random':{}}
    cities_inverse = {'fiedler': {}, 'tsne': {}, 'umap':{}, 'original':{}, 'random':{}}

# Defining EPSG (coordinate reference systems) for each city.
# EPSG codes are used to properly handle geospatial data, ensuring correct map projections.
epsg = {
    'Busan': 'EPSG:4612',
    'Mumbai': "EPSG:7767",
    'Barcelona': 'EPSG:2062',
    'Nairobi': 'EPSG:4210',
    'Santiago': 'EPSG:9147',
    'Bogota': 'EPSG:21897'
}

# Loop through each city to process its street graph
for city in epsg:
    print(city) 
    cg = CityHub.CityHub(city)  # Instantiate a CityHub object for the current city
    G = cg.city_street_graph  # Retrieve the city’s street graph (nodes and edges)

    sigma = 5000  # Sigma parameter (for inverse Laplacian matrix calculations)

    # Add 'inv_length' (inverse of edge length) as an attribute to each edge in the graph.
    # This will be used to calculate the Laplacian matrix.
    for u in G.edges:
        G[u[0]][u[1]][0]["inv_length"] = 1 / G[u[0]][u[1]][0]["length"]

    # Calculate the Laplacian matrix of the graph using the 'inv_length' as the weight
    laplacian = nx.laplacian_matrix(G, weight='inv_length')

    # Compute the eigenvalues and eigenvectors of the Laplacian matrix.
    # The Fiedler vector corresponds to the second smallest eigenvalue.
    eig_values, eig_vectors = eigs(laplacian.astype(float), k=2, which='LM', sigma=0)
    fiedler_pos = np.where(eig_values.real == np.sort(eig_values.real)[1])[0][0]  # Find the Fiedler position
    fiedler_vector = np.transpose(eig_vectors)[fiedler_pos]  # Retrieve the Fiedler vector

    print("Fiedler vector: " + str(fiedler_vector.real))
    fiedler_vector_real = fiedler_vector.real
    fiedler_sorted_index_list = np.argsort(np.argsort(fiedler_vector_real))  # Sorting the Fiedler vector

    # Assign the 'fiedler_index' attribute to each node in the graph based on the sorted Fiedler vector
    i = 0
    for u in G.nodes:
        G.nodes[u]['fiedler_index'] = fiedler_sorted_index_list[i]
        i += 1

    # Create an inverse mapping of the sorted Fiedler vector
    inverse_fiedler_sorted_index_list = [0] * len(cg.city_vert_list)
    for i in range(len(fiedler_sorted_index_list)):
        inverse_fiedler_sorted_index_list[fiedler_sorted_index_list[i]] = i

    # Generate a random permutation of indices (for comparison with other methods)
    random_index_list = np.random.permutation(len(cg.city_vert_list))

    # t-SNE dimensionality reduction to 1D

    print("TSNE -----------------------------------")
    nodes2d = np.array(cg.city_vert_list)  # Retrieve the 2D coordinates of nodes

    # Create a GeoDataFrame for the nodes and reproject to the correct EPSG coordinate system
    gpdnodes = gpd.GeoDataFrame(geometry=gpd.points_from_xy(nodes2d[:, 0], nodes2d[:, 1]), crs="EPSG:4326")
    gpdnodes = gpdnodes.to_crs(epsg[city])  # Reproject to the specific city's CRS
    nodesproj = gpdnodes.geometry.apply(lambda g: pd.Series(g.coords[0]))  # Extract projected coordinates

    # Normalize the projected coordinates using StandardScaler
    scaler = StandardScaler()
    scaler.fit(nodesproj)
    nodesprojsca = scaler.transform(nodesproj)

    print("Calculating TSNE")
    # Apply t-SNE with 1 component to reduce the 2D node coordinates to 1D
    X_embedding = openTSNE.TSNE(
        n_components=1,
        perplexity=2000,
        n_jobs=10,
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
        marker = folium.CircleMarker([row['y'], row['x']], radius=1, color=colormap(index))
        popup = folium.Popup(str(index))  # Add index as popup text
        marker.add_child(popup)
        m.add_child(marker)  # Add marker to the map

    # Add the color map legend and save the map as an HTML file
    m.add_child(colormap)
    m.save('tsne_map_' + city + '.html')

    # Create the inverse mapping for the t-SNE sorted indices
    df = nodesdf.sort_values(by=['index'])  # Sort the DataFrame back by the original index
    tsne_sorted_index_list = np.array(df['new_index'])  # Get the new sorted indices based on t-SNE
    inverse_tsne_sorted_index_list = [0] * len(cg.city_vert_list)
    for i in range(len(tsne_sorted_index_list)):
        inverse_tsne_sorted_index_list[tsne_sorted_index_list[i]] = i  # Inverse map of t-SNE indices

    # ---- UMAP Embedding ----
    print("UMAP --------------------------------------------------")

    # Create a UMAP model for 1D embedding with high n_neighbors (2000) to preserve global structure
    umap_model = umap.UMAP(n_components=1, n_neighbors=2000, min_dist=0.5, random_state=42)
    umap_embedding = umap_model.fit_transform(nodesprojsca)  # Apply UMAP to the normalized projected coordinates

    # Create another DataFrame to hold the UMAP results
    nodesdf_umap = pd.DataFrame(nodes2d)  # Recreate DataFrame with original node coordinates
    nodesdf_umap['y'] = nodesproj[0]  # Add y-coordinates (latitude)
    nodesdf_umap['x'] = nodesproj[1]  # Add x-coordinates (longitude)
    nodesdf_umap["umap"] = umap_embedding  # Store the UMAP embedding in the DataFrame
    nodesdf_umap = nodesdf_umap.sort_values('umap').reset_index()  # Sort the DataFrame by UMAP values
    nodesdf_umap["new_index"] = nodesdf_umap.index  # Assign new indices after sorting

    # Reuse the same color map for UMAP visualization
    mi = 0
    ma = nodesdf.shape[0] - 1
    colormap = cm.LinearColormap(colors=['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red'],
                                 index=[mi, ma/5, ma/5*2, ma/5*3, ma/5*4, ma], vmin=mi, vmax=ma,
                                 caption='UMAP neighbors 2000')

    # Recenter the map for UMAP results and create a new folium map
    m = folium.Map(location=[med_lat, med_lon], zoom_start=12, tiles='CartoDB positron')

    # Add CircleMarkers for UMAP embedding
    for index, row in nodesdf_umap.iterrows():
        marker = folium.CircleMarker([row['y'], row['x']], radius=1, color=colormap(index))
        popup = folium.Popup(str(index))
        marker.add_child(popup)
        m.add_child(marker)

    # Add the color map legend and save the UMAP map as an HTML file
    m.add_child(colormap)
    m.save('umap_map_' + city + '.html')

    # Create the inverse mapping for UMAP sorted indices
    df = nodesdf_umap.sort_values(by=['index'])  # Sort by original indices
    umap_sorted_index_list = np.array(df['new_index'])  # Get the new UMAP-sorted indices
    inverse_umap_sorted_index_list = [0] * len(cg.city_vert_list)
    for i in range(len(umap_sorted_index_list)):
        inverse_umap_sorted_index_list[umap_sorted_index_list[i]] = i  # Inverse map of UMAP indices


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

    # --- Original Embedding Diagonals ---
    diagonals_original = []
    for i in range(0, len(cg.city_vert_list) - window_size, 1):
        Lats = [cg.city_vert_list[j][0] for j in range(i,i+window_size)]
        Longs = [cg.city_vert_list[j][1] for j in range(i,i+window_size)]
        min_rad = [math.radians(_) for _ in [min(Lats),min(Longs)]]
        max_rad = [math.radians(_) for _ in [max(Lats),max(Longs)]]
        diag = haversine_distances([min_rad,max_rad])[0,1]* 6371000/1000
        norm_diag = diag / best_diagonal[i + int(window_size/2)]
        diagonals_original.append(norm_diag)

    # --- Fiedler Embedding Diagonals ---
    diagonals_fiedler = []
    for i in range(0, len(cg.city_vert_list) - window_size, 1):
        Lats = [cg.city_vert_list[j][0] for j in inverse_fiedler_sorted_index_list[i:i+window_size]]
        Longs = [cg.city_vert_list[j][1] for j in inverse_fiedler_sorted_index_list[i:i+window_size]]
        min_rad = [math.radians(_) for _ in [min(Lats),min(Longs)]]
        max_rad = [math.radians(_) for _ in [max(Lats),max(Longs)]]
        diag = haversine_distances([min_rad,max_rad])[0,1]* 6371000/1000
        norm_diag = diag / best_diagonal[inverse_fiedler_sorted_index_list[i + int(window_size/2)]]
        diagonals_fiedler.append(norm_diag)

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

    # --- UMAP Embedding Diagonals ---
    diagonals_umap = []
    for i in range(0, len(cg.city_vert_list) - window_size, 1):
        Lats = [cg.city_vert_list[j][0] for j in inverse_umap_sorted_index_list[i:i+window_size]]
        Longs = [cg.city_vert_list[j][1] for j in inverse_umap_sorted_index_list[i:i+window_size]]
        min_rad = [math.radians(_) for _ in [min(Lats),min(Longs)]]
        max_rad = [math.radians(_) for _ in [max(Lats),max(Longs)]]
        diag = haversine_distances([min_rad,max_rad])[0,1]* 6371000/1000
        norm_diag = diag / best_diagonal[inverse_umap_sorted_index_list[i + int(window_size/2)]]
        diagonals_umap.append(norm_diag)

    # --- Random Embedding Diagonals ---
    diagonals_random = []
    for i in range(0, len(cg.city_vert_list) - window_size, 1):
         Lats = [cg.city_vert_list[j][0] for j in random_index_list[i:i+window_size]]
        Longs = [cg.city_vert_list[j][1] for j in random_index_list[i:i+window_size]]
        min_rad = [math.radians(_) for _ in [np.min(Lats),np.min(Longs)]]
        max_rad = [math.radians(_) for _ in [np.max(Lats),np.max(Longs)]]
        diag = haversine_distances([min_rad,max_rad])[0,1]* 6371000/1000
        norm_diag = diag / best_diagonal[random_index_list[i + int(window_size/2)]]
        diagonals_random.append(norm_diag)

    # Output the average and median values for each method
    print("FORWARD")
    print('Fiedler - Mean:', statistics.mean(diagonals_fiedler))
    print('Fiedler - Median:', statistics.median(diagonals_fiedler))
    print('t-SNE - Mean:', statistics.mean(diagonals_tsne))
    print('t-SNE - Median:', statistics.median(diagonals_tsne))
    print('UMAP - Mean:', statistics.mean(diagonals_umap))
    print('UMAP - Median:', statistics.median(diagonals_umap))
    print('Original - Mean:', statistics.mean(diagonals_original))
    print('Original - Median:', statistics.median(diagonals_original))
    print('Random - Mean:', statistics.mean(diagonals_random))
    print('Random - Median:', statistics.median(diagonals_random))

    # Create a boxplot comparing the diagonal distances for each method
    plt.figure(figsize=(4, 3))
    plt.boxplot([diagonals_fiedler, diagonals_tsne, diagonals_umap, diagonals_original, diagonals_random],
                labels=['Fiedler', 't-SNE', 'UMAP', 'Original', 'Random'], showfliers=False)
    plt.xlabel("Index Ordering")
    plt.ylabel("Forward Approach (Diagonal Distance)")
    plt.savefig(city + '_forward.pdf', dpi=300, format='pdf', bbox_inches='tight')

    # Save the forward metrics for each city
    cities_forward['fiedler'][city] = diagonals_fiedler
    cities_forward['tsne'][city] = diagonals_tsne
    cities_forward['umap'][city] = diagonals_umap
    cities_forward['original'][city] = diagonals_original
    cities_forward['random'][city] = diagonals_random


    # Metric for quality: latlong -> embedding (INVERSE METRIC)
    print("Calculating metrics for latlong -> embedding")

    # Initialize lists to store the maximum inverse index distance for each method
    max_inverse_ind_dist_fiedler = []
    max_inverse_ind_dist_tsne = []
    max_inverse_ind_dist_umap = []
    max_inverse_ind_dist_original = []
    max_inverse_ind_dist_random = []

    # Loop through each node in the city's vertex list
    for i in range(0, len(cg.city_vert_list)):
        # Calculate the shortest paths from the source node to others within a cutoff distance (0.5 km)
        ps = nx.single_source_dijkstra_path(cg.city_street_graph, cg.city_vert_ind_to_nxind_dict[i], cutoff=0.5 * 1000.0, weight='length')
        result_nodes = [cg.city_vert_nxind_to_ind_dict[k] for k in ps.keys()]
        
        # Get the sorted indices for Fiedler, t-SNE, UMAP, and Random embeddings
        inds_sorted_fiedler = [fiedler_sorted_index_list[j] for j in result_nodes]
        inds_sorted_tsne = [tsne_sorted_index_list[j] for j in result_nodes]
        inds_sorted_umap = [umap_sorted_index_list[j] for j in result_nodes]
        inds_random = [random_index_list[j] for j in result_nodes]
        
        # Calculate the maximum inverse index distance and normalize by the number of result nodes
        max_inverse_ind_dist_fiedler.append((np.max(inds_sorted_fiedler) - np.min(inds_sorted_fiedler)) / len(inds_sorted_fiedler))
        max_inverse_ind_dist_tsne.append((np.max(inds_sorted_tsne) - np.min(inds_sorted_tsne)) / len(inds_sorted_tsne))
        max_inverse_ind_dist_umap.append((np.max(inds_sorted_umap) - np.min(inds_sorted_umap)) / len(inds_sorted_umap))
        max_inverse_ind_dist_original.append((np.max(result_nodes) - np.min(result_nodes)) / len(result_nodes))
        max_inverse_ind_dist_random.append((np.max(inds_random) - np.min(inds_random)) / len(inds_random))

    # Output the average and median values for each method
    print("INVERSE")
    print('Fiedler - Mean:', statistics.mean(max_inverse_ind_dist_fiedler))
    print('Fiedler - Median:', statistics.median(max_inverse_ind_dist_fiedler))
    print('t-SNE - Mean:', statistics.mean(max_inverse_ind_dist_tsne))
    print('t-SNE - Median:', statistics.median(max_inverse_ind_dist_tsne))
    print('UMAP - Mean:', statistics.mean(max_inverse_ind_dist_umap))
    print('UMAP - Median:', statistics.median(max_inverse_ind_dist_umap))
    print('Original - Mean:', statistics.mean(max_inverse_ind_dist_original))
    print('Original - Median:', statistics.median(max_inverse_ind_dist_original))
    print('Random - Mean:', statistics.mean(max_inverse_ind_dist_random))
    print('Random - Median:', statistics.median(max_inverse_ind_dist_random))

    # Create a boxplot comparing the inverse index distances for each method
    plt.figure(figsize=(4, 3))
    plt.boxplot([max_inverse_ind_dist_fiedler, max_inverse_ind_dist_tsne, max_inverse_ind_dist_umap],
                labels=['Fiedler', 't-SNE', 'UMAP'], showfliers=False)
    plt.xlabel("Index Ordering")
    plt.ylabel("Inverse Approach (Max Index Distance)")
    plt.savefig(city + '_inverse.pdf', dpi=300, format='pdf', bbox_inches='tight')

    # Save the inverse metrics for each city
    cities_inverse['fiedler'][city] = max_inverse_ind_dist_fiedler
    cities_inverse['tsne'][city] = max_inverse_ind_dist_tsne
    cities_inverse['umap'][city] = max_inverse_ind_dist_umap
    cities_inverse['original'][city] = max_inverse_ind_dist_original
    cities_inverse['random'][city] = max_inverse_ind_dist_random

    # Save the forward and inverse metrics as pickle files
    with open("forward.pkl", "wb") as file:
        pickle.dump(cities_forward, file)

    with open("inverse.pkl", "wb") as file:
        pickle.dump(cities_inverse, file)