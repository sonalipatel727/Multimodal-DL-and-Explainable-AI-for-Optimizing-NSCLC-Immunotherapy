import numpy as np
import pandas as pd
import shap
import re
from array import *


def run_shap_on_model(model, X_train_data, X_test_data, features, target, save):
    explainer = shap.GradientExplainer(model, X_train_data) 
    shap_values = explainer(X_test_data) #specify which case this was on ps, need to do bs and po
    
    shap_values_array = shap_values.values.squeeze(axis=-1)
    shap_values_array = np.array(shap_values_array)
    #print("SHAP Values for: ", shap_values_array)
    #print(X_test_data.shape)

    shap.summary_plot(shap_values_array, X_test_data.numpy(), feature_names=features)
    shap_df = pd.DataFrame(shap_values_array, columns = features)

    feature_importance = shap_df.abs().mean().sort_values(ascending=False)

    #If want to save shap_df and feature importance list 
    if save == True:
        shap_df.to_csv(target + '_shap_values.csv', index=True)
        feature_importance.to_csv(target + '_shap_values_overall.csv', index = True)
    #print(feature_importance)
    return shap_df


def get_top_num_features(feature_num, feature_importance_df, target_label):
    feature_string = "The features with associated SHAP values are presented as the following for target label ", target_label, ": "

    for i in range(feature_num):
        feature_name = feature_importance_df.index[i]
        feature_name = feature_name.replace("{", "").replace("}", "")
        feature_shap_value = feature_importance_df.iloc[i]
        
        feature_string += f"{feature_name}: {feature_shap_value:.6f}"
        
        if i < feature_num - 1:
            feature_string += ", "

    return feature_string
    