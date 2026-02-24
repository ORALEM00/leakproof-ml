import os
import shap
import numpy as np
import matplotlib.pyplot as plt

from leakproof_ml.utils import  load_results_from_json
from leakproof_ml.plots import plot_predictions, histogram_errors, plot_interpretability_bar 
from leakproof_ml.plots import interpretability_comparison_plot

# Script absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
# Set working directory to the project root
os.chdir(base_dir) 

# Input path
input_path = "raw_results2"
input_interpertability_path = "raw_interpretability_results2"
# Output path
plot_path = "resulting_plots2"

# Models to be plotted
models_name = ['XGBRegressor', 'Ridge', 'CatboostRegressor', 'VotingRegressor_0', 'VotingRegressor_1']

results ={}
for model in models_name:
    # Create output directory   
    output_path = f"{plot_path}/{model}/"
    os.makedirs(os.path.dirname(output_path), exist_ok = True)

    # Metric plots
    for method in ["trainTest","trainTest_removed","randomCV","randomCV_removed","groupedCV", 
           "groupedCV_removed"]: 
        # Load results per methdology
        results = load_results_from_json(f"{input_path}/{model}/tuned/{method}.json")

        # Create metric directory
        metric_path = os.path.join(output_path, "metrics")
        os.makedirs(metric_path, exist_ok=True)

        # Store plots per methodology
        plot_predictions(results['y_true'], results['y_predict'], 
                         filename=f"{metric_path}/{method}_predictions.png")
        histogram_errors(results['y_true'], results['y_predict'], 
                         filename=f"{metric_path}/{method}_histogram_errors.png")

    # Interpretability plots
    for method in ["pi", "shap"]:
        for key in ['trainTest', 'trainTest_removed', 'randomCV', 'randomCV_removed', 'groupedCV', 'groupedCV_removed']:
            res = load_results_from_json(f"{input_interpertability_path}/{model}/{method}/{method}_{key}.json")
            results[f"{method}_{key}"] = res

            # Create method directory
            method_path = os.path.join(output_path, method)
            os.makedirs(method_path, exist_ok=True)

            # PLot individual bars
            plot_interpretability_bar(results[f"{method}_{key}"], 
                           title = f"{method} Feature Importance for {model}", 
                           filename=f"{method_path}/{method}_{key}.png",
                           method=method)
            
            if method == 'shap':
                plt.figure()
                # Also plot SHAP summary plot
                shap.summary_plot(
                    np.array(results[f"{method}_{key}"]["shap_values"]), 
                    np.array(results[f"{method}_{key}"]['X_test']), 
                    results[f"{method}_{key}"]['features'],
                    show=False
                )
                plt.tight_layout()
                plt.savefig(f"{method_path}/shap_summary_{key}.png", dpi=300)
                plt.close()
    
    # Plot comprehensive comparison plots shap
    interpretability_comparison_plot(
        results['shap_groupedCV'],
        results['shap_randomCV'],
        method='shap', labels = ['Grouped CV', 'Random CV'],
        title=f'SHAP comparision between protocols for {model}', 
        filename=f"{output_path}/shap_comparison_2bar.png"
        )

    # Plot comprehensive comparison plots permutation
    interpretability_comparison_plot(
        results['pi_groupedCV'],
        results['pi_randomCV'],
        filename=f"{output_path}/pi_comparison_2bar.png",
        method='perm', labels = ['Grouped CV', 'Random CV'],
        title=f'Permutation Importance comparision between protocols for {model}'
        )
    
    # Plot comprehensive comparison plots shap
    interpretability_comparison_plot(
        results['shap_groupedCV_removed'],
        results['shap_groupedCV'],
        results['shap_randomCV_removed'],
        results['shap_randomCV'],
        method='shap', labels = ['Grouped CV (removed)', 'Grouped CV', 'Random CV (removed)', 'Random CV'],
        title=f'SHAP comparision between protocols for {model}', 
        filename=f"{output_path}/shap_comparison_4bar.png"
        )

    # Plot comprehensive comparison plots permutation
    interpretability_comparison_plot(
        results['pi_groupedCV_removed'],
        results['pi_groupedCV'],
        results['pi_randomCV_removed'],
        results['pi_randomCV'],
        filename=f"{output_path}/pi_comparison_4bar.png",
        method='perm', labels = ['Grouped CV (removed)', 'Grouped CV', 'Random CV (removed)', 'Random CV'],
        title=f'Permutation Importance comparision between protocols for {model}'
        )
    
    # Plot comprehensive comparison plots shap
    interpretability_comparison_plot(
        results['shap_groupedCV_removed'],
        results['shap_groupedCV'],
        results['shap_randomCV_removed'],
        results['shap_randomCV'],
        results['shap_trainTest_removed'],
        results['shap_trainTest'],
        method='shap', labels = ['Grouped CV (removed)', 'Grouped CV', 'Random CV (removed)', 'Random CV', 'Train Test (removed)', 'Train Test'],
        title=f'SHAP comparision between protocols for {model}', 
        filename=f"{output_path}/shap_comparison_6bar.png"
        )

    # Plot comprehensive comparison plots permutation
    interpretability_comparison_plot(
        results['pi_groupedCV_removed'],
        results['pi_groupedCV'],
        results['pi_randomCV_removed'],
        results['pi_randomCV'],
        results['pi_trainTest_removed'],
        results['pi_trainTest'],
        filename=f"{output_path}/pi_comparison_6bar.png",
        method='perm', labels = ['Grouped CV (removed)', 'Grouped CV', 'Random CV (removed)', 'Random CV', 'Train Test (removed)', 'Train Test'],
        title=f'Permutation Importance comparision between protocols for {model}'
        )
    