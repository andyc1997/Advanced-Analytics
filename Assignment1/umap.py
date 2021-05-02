# -*- coding: utf-8 -*-
"""
Created on Sun May  2 03:14:41 2021

@author: user
"""
# Packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

#%% 

import umap
from umap.umap_ import UMAP

# Print more rows and columns of pandas.DataFrame
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#%%
# Change path if needed
path = r'C:\Users\user\Desktop\KUL - Mstat\Big Data Platforms and Technologies\project'
data = pd.read_csv(path + r'\ctrain.csv')

#%%

# Weight of evidence & Information value
def get_information_value(data, features):
    # cross tab
    tab = pd.crosstab(data[features], data['fraud'])
    # weight of evidence
    tab['all'] = tab[['Y', 'N']].sum(axis = 1) 
    tab['share'] = tab['all'] / tab['all'].sum(axis = 0)
    tab['Y_rate'] = tab['Y'] / tab['all']
    tab['N_dist'] = tab['N'] / tab['N'].sum()
    tab['Y_dist'] = tab['Y'] / tab['Y'].sum()
    tab['WoE'] = np.log(tab['N_dist'] / tab['Y_dist'])
    tab = tab.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    # information value
    tab['IV'] = tab['WoE'] * (tab['N_dist'] - tab['Y_dist'])
    return tab[np.abs(tab['IV']) > 0.01].index.values # threshold 0.01

# apply get_information_value
claim_postal_code_list = get_information_value(data, 'claim_postal_code')
policy_holder_postal_code_list = get_information_value(data, 'policy_holder_postal_code')
driver_postal_code_list = get_information_value(data, 'driver_postal_code')
third_party_1_postal_code_list = get_information_value(data, 'third_party_1_postal_code')
third_party_2_postal_code_list = get_information_value(data, 'third_party_2_postal_code')
repair_postal_code_list = get_information_value(data, 'repair_postal_code')
claim_vehicle_brand_list = get_information_value(data, 'claim_vehicle_brand')
policy_coverage_type_list = get_information_value(data, 'policy_coverage_type')

def handle_age(value):
    # A simple program to discretize age
    if pd.isna(value):
        return 'unknown'
    else:
        if value <= 20:
            return '<=20'
#        elif value <= 30:
#            return '<=30'
        elif value <= 40:
            return '<=40'
#        elif value <= 50:
#            return '<=50'
        elif value <= 60:
            return '<=60'
 #       elif value <= 70:
 #           return '<=70'
        elif value <= 80:
            return '<=80'
        else:
            return '>80'

def handle_policy_coverage(value):
    # A simple program to discretize policy_coverage_1000
    if pd.isna(value):
        return 'unknown'
    else:
        if value <= 20:
            return '<=20'
        elif value <= 40:
            return '<=40'
        elif value <= 60:
            return '<=60'
        elif value <= 80:
            return '<=80'
        else:
            return '>80'

def handle_categorical_grouping(value, grouping_list):
        if value == 'unknown':
            return value
        elif value in grouping_list:
            return str(value)
        else:
            return 'other'
        
def transform(x_dataset):
        x_dataset['driver_age'] = x_dataset['driver_age'].apply(lambda x: handle_age(x))
        x_dataset['policy_holder_age'] = x_dataset['policy_holder_age'].apply(lambda x: handle_age(x))
        x_dataset['repair_age'] = x_dataset['repair_age'].apply(lambda x: handle_age(x))
        x_dataset['third_party_1_age'] = x_dataset['third_party_1_age'].apply(lambda x: handle_age(x))
        x_dataset['third_party_2_age'] = x_dataset['third_party_2_age'].apply(lambda x: handle_age(x))
        x_dataset['third_party_3_age'] = x_dataset['third_party_3_age'].apply(lambda x: handle_age(x))
        
        x_dataset['policy_coverage_1000'] = x_dataset['policy_coverage_1000'].apply(lambda x: handle_policy_coverage(x))
        
        x_dataset['claim_postal_code'] = x_dataset['claim_postal_code'].apply(lambda x: handle_categorical_grouping(x, claim_postal_code_list))
        x_dataset['policy_holder_postal_code'] = x_dataset['policy_holder_postal_code'].apply(lambda x: handle_categorical_grouping(x, policy_holder_postal_code_list))
        x_dataset['driver_postal_code'] = x_dataset['driver_postal_code'].apply(lambda x: handle_categorical_grouping(x, driver_postal_code_list))
        x_dataset['third_party_1_postal_code'] = x_dataset['third_party_1_postal_code'].apply(lambda x: handle_categorical_grouping(x, third_party_1_postal_code_list))
        x_dataset['third_party_2_postal_code'] = x_dataset['third_party_2_postal_code'].apply(lambda x: handle_categorical_grouping(x, third_party_2_postal_code_list))
        x_dataset['third_party_3_postal_code'] = x_dataset['third_party_3_postal_code'].apply(lambda x: x if x == 'unknown' else 'other')
        x_dataset['repair_postal_code'] = x_dataset['repair_postal_code'].apply(lambda x: handle_categorical_grouping(x, repair_postal_code_list))
        # x_dataset['claim_vehicle_brand'] = x_dataset['claim_vehicle_brand'].apply(lambda x: handle_categorical_grouping(x, claim_vehicle_brand_list))
        x_dataset['policy_coverage_type'] = x_dataset['policy_coverage_type'].apply(lambda x: handle_categorical_grouping(x, policy_coverage_type_list))        
        
        return x_dataset.drop(['third_party_1_id_known', 'third_party_2_id_known', 'third_party_3_id_known'], axis = 1)
    
#%%
# Some transformation, same steps as prediction
data = transform(data)

#%%
# Separate target and features
X, y = data.drop(['claim_id', 'fraud'], axis = 1), data['fraud'].apply(lambda x: 1 if x == 'Y' else 0)

#%%
# Umap for continuous features with euclidean distance metrics
cont_feature = X.columns[X.dtypes != np.dtype('O')]
cont_feature = cont_feature[~cont_feature.str.endswith('_id_known')]

pipeline = Pipeline([('impute', SimpleImputer(strategy = 'median')),
                     ('scaler', StandardScaler()),
                     ('upsampling', SMOTE(random_state = 999))])
X_cont_resample, y_resample = pipeline.fit_resample(X[cont_feature], y)


#%%
# Umap for binary features with jaccard distance metrics
pipeline = Pipeline([('upsampling', SMOTE(random_state = 999))])
X_cat_resample, y_resample2 = pipeline.fit_resample(pd.get_dummies(X.drop(cont_feature, axis = 1), 
                              drop_first = True), y)

#%%
# Fit umap objects
fit_cont = umap.UMAP(metric = 'euclidean', n_neighbors = 30, n_components = 3).fit(X_cont_resample, y_resample)
fit_cat = umap.UMAP(metric = 'jaccard', n_neighbors = 30, n_components = 3).fit(X_cat_resample, y_resample)

#%%
# Get lower dimensional representation
umap_embedded_cont = fit_cont.transform(X_cont_resample)
umap_embedded_cat = fit_cat.transform(X_cat_resample)

#%%
claim_amount_smote, y_resample3 = SMOTE(random_state = 999).fit_resample(data[['claim_amount']], y)

#%%
# Plot 2d
def umap_plot_embedded(umap_obj, feature_type):
    fig, ax = plt.subplots(1, figsize=(14, 10))
    colors = ['blue', 'red']
    plt.scatter(*umap_obj.T, s = 0.1, c = y_resample, 
                cmap = matplotlib.colors.ListedColormap(colors), alpha = 1.0)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title('UMAP for ' + feature_type + ' data with n_neighbors = 30')
    cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
    cbar.set_ticks(np.arange(2))
    cbar.set_ticklabels(['N', 'Y'])
    return fig

fig_cat = umap_plot_embedded(umap_embedded_cat, 'categorical')
fig_cont = umap_plot_embedded(umap_embedded_cont, 'continuous')

#%% 
#Plot 3d
def umap_plot_embedded(umap_obj, feature_type):
    fig = matplotlib.pyplot.figure()
    ax = Axes3D(fig)
    ax.view_init(340, 90)
    colors = claim_amount_smote
    plt3d = ax.scatter(*umap_obj.T, s = 0.1, 
               c = colors, cmap = 'cool',
               alpha = 1.0)
    #plt.setp(ax, xticks=[], yticks=[])
    plt.title('UMAP for ' + feature_type + ' data with n_neighbors = 30')
    fig.colorbar(plt3d, ax = ax)
    return fig

fig_cat = umap_plot_embedded(umap_embedded_cat, 'categorical')
fig_cont = umap_plot_embedded(umap_embedded_cont, 'continuous')

#%% 
# Sanity check
print(np.sum(y_resample3 != y_resample2))
print(np.sum(y_resample != y_resample2))
