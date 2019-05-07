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
    
def update_feature(df, feature, percentage, threshold):
    """ This function updates the column feature by +- percentage
        for all observations of df where the life expectancy is
        below threshold. """
    # Get observations where life expectation is below threshold
    life_exp_below_th = df[df['Life expectancy'] < threshold]
    indices = life_exp_below_th.index
    # Increase feature by percentage
    new_values = life_exp_below_th[feature].apply(lambda x: x*(1+percentage)).values
    # Update the life_expectancy data frame with the new values
    life_expectancy_updated = df.copy()
    life_expectancy_updated.at[indices, feature] = new_values
    return life_expectancy_updated

def predict_new_le(df, threshold, model):
    """ This functions predict the new life expectancy of all
        observations of df where the life expectancy was below
        threshold using the model model. """
    ids = df['id'].values
    # Get the observations where the life expectancy is below threshold
    df = df.drop(['Continent', 'id'], axis=1)
    indices = df[df['Life expectancy'] < threshold].index
    below_th = df.loc[indices]
    below_th_x = below_th.drop(['Life expectancy'], axis=1)
    X = below_th_x.values
    y_pred = model.predict(X)
    # Set back the values in the dataframe
    df.at[indices, 'Life expectancy'] = y_pred
    df['id'] = ids
    return df

def get_life_exp_pred_results(original_df, increases_per_feature, indices, threshold, model):
    """ This function returns the average life expectancy over all countries that
        had a life expectancy below threshold and the number of countries that obtained
        a life expectancy above threshold, after the improvements of the features in
        the dictionary increases_per_feature. """
    averages_le = []
    nb_countries = [] 
    for key in increases_per_feature:
        avg_key = []
        nb_countries_key = []
        for i in increases_per_feature[key]:
            updated_df = update_feature(original_df, key, i, threshold)
            predicted_df = predict_new_le(updated_df, threshold, model)
            life_exp = predicted_df.loc[indices]
            avg_key.append(np.mean(life_exp['Life expectancy'].values))
            nb_countries_key.append(life_exp[life_exp['Life expectancy'] >= threshold].shape[0]) 
        averages_le.append(avg_key)
        nb_countries.append(nb_countries_key)
    return averages_le, nb_countries