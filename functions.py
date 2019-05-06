# Imports

import folium
import numpy as np
import pandas as pd
import seaborn as sns
from folium import plugins
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Functions

def get_nan_columns(df, country_indexed=False):
    """ This function get the name of the columns that contain NaN values
        and the number of NaN values in that colum. """
    cols = pd.DataFrame(df.isna().any(), columns=['hasnan']).reset_index()
    cols.columns = ['name', 'hasnan']
    cols['number'] = cols['name'].apply(lambda x: df.loc[df[x].isna()].shape[0])
    if(country_indexed):
        cols['number country'] = cols['name'].apply(lambda x: np.unique(df.loc[df[x].isna()].index).size)
    else:
        cols['number country'] = cols['name'].apply(lambda x: np.unique(df.loc[df[x].isna()]['Country'].values).size)
    nan_columns = cols.loc[cols['number'] > 0]
    return nan_columns

def show_map(df, feature, legend):
    # Geojson file
    world_geo = r'Data/world-countries.json'
    # Create a plain world map
    world_map = folium.Map(location=[30, 0], zoom_start=2, tiles='Mapbox Bright')
    # Generate choropleth map using the feature_name
    #bins = list(df[feature].quantile([0, 0.25, 0.5, 0.75, 1]))
    world_map.choropleth(
        geo_data=world_geo,
        data=df,
        columns=['id', feature],
        key_on='feature.id', 
        fill_color='BuPu',
        fill_opacity=0.75,
        line_opacity=0.2,
        legend_name=legend,
        #bins=None,
        reset=True)
    # Display map
    return world_map

def plot_corr_map(df):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(df.corr(), cmap='BuPu', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(df.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df.columns, fontsize='large')
    ax.set_yticklabels(df.columns, fontsize='large')
    plt.show()
    
def scatter_plot(df, scatter_x, scatter_y, x_label, y_label, x_log=False):
    continents = df['Continent'].values
    cdict = {'Europe': 'royalblue', 'America': 'red', 'Africa': 'orange', 'Asia': 'limegreen', 'Oceania': 'violet'}
    for g in np.unique(continents):
        ix = np.where(continents == g)
        plt.scatter(scatter_x[ix], scatter_y[ix], s=400, edgecolor='white', c=cdict[g], label=g, alpha=0.6)
    plt.legend(loc='lower right', fontsize='x-large')
    if(x_log):
        plt.xscale('log')
    plt.xlabel(x_label, fontsize='x-large')
    plt.ylabel(y_label, fontsize='x-large')
    plt.gcf().set_size_inches(18, 8)
    plt.show()
    
def plot_pred(y_test, y_pred, title, score):
    """ Plot the predicted values y_pred against
        the test values y_test. """
    plt.scatter(y_test, y_pred, c='royalblue', edgecolors='white', alpha=0.6, s=400)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=4, color='crimson')
    plt.title(title + (' - Score: {:.2f}%'.format(score*100)), fontsize='x-large')
    plt.xlabel('Original', fontsize='x-large')
    plt.ylabel('Predicted', fontsize='x-large')
    plt.gcf().set_size_inches(18, 8)
    plt.show()
    
def plot_features_coefs(features, weights, title):
    colors = ['crimson' if c < 0 else 'royalblue' for c in np.sort(weights)]
    plt.barh(features[np.argsort(weights)], np.sort(weights), color=colors)
    plt.grid(True)
    plt.title(title, fontsize='x-large')
    plt.gcf().set_size_inches(18, 8)
    plt.show()
    
def plot_dist(df, feature): 
    sns.distplot(df[feature])
    plt.xlabel(feature, fontsize='x-large')
    plt.gcf().set_size_inches(18, 5)
    plt.show()