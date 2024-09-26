## A visual methodology to assess spatial graph vertex ordering algorithms

Graph vertex ordering is crucial for various graph-related applications, especially in spatial and urban data analysis where graphs represent real-world locations and their connections. The task is to arrange vertices along a single axis while preserving spatial relationships, but this often results in distortions due to the complexity of spatial data. Existing methods mostly assess ordering quality using a global metric, which may not capture specific use case needs or localized variations.

This work proposes a new methodology to visually evaluate and compare vertex ordering techniques on spatial graphs. Two quantitative comparison mechanisms are proposed. Experiments on urban data from various cities demonstrate the methodology's effectiveness in tuning hyperparameters and comparing well-known vertex ordering techniques. The visual approach reveals nuanced spatial patterns that global metrics might miss, providing deeper insights into the behavior of different vertex ordering methods.

### Project Structure

In the **src** folder, you will find two key methods for calculating forward and inverse metrics for sorted spatial graph nodes:

1. The Jupyter notebook **GenerateMapsMaceioExample.ipynb**:

  * Applies different vertex ordering techniques to a city's spatial graph.
  * Calculates both forward and inverse metrics for these orderings.
  * Generates maps and visualizations of the results.
   
2. The Python script **ProcessSeveralCities.py**:
   
  * Iterates over multiple cities, applying the same vertex ordering techniques.
  * Measures the forward and inverse metrics automatically for each city.
  * To customize the cities processed, simply modify the **epsg** dictionary by adding or removing city names and their corresponding EPSG codes.

### Dependencies
Both files depend on [CityHub](https://github.com/victorhb/CityHub) for obtaining and manipulating city street graphs.
