import numpy as np
import matplotlib.pyplot as plt

from ._plots_utils import _shap_barPlot_dictionary, _align_interpretability_dicts


def plot_interpretability_bar(data, title, method = "perm", filename = None):
    """
    Generate a horizontal bar plot for feature importance.

    Visualizes mean importance scores with associated error bars (standard deviation). 
    Supports both Permutation Importance and SHAP values by automatically 
    converting SHAP results into a global importance format.

    Parameters
    ----------
    data : dict
        A dictionary containing 'features', 'importance_mean', and 'importance_std'.
    title : str
        The title of the plot.
    method : {'perm', 'shap'}, default='perm'
        The interpretability method used. If 'shap', raw values are converted 
        to global importance magnitudes.
    filename : str, optional
        Path to save the plot. If None, the plot is displayed interactively.

    Returns
    -------
    None
    """
    # Convert raw SHAP values into a global importance summary dictionary
    if method == "shap":
        data = _shap_barPlot_dictionary(data)
        
    feature_names = data['features']
    means = data['importance_mean']
    stds = data['importance_std']
    
    num_features = len(feature_names)
    
    # Increase height for readability based on number of features
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(10, max(5, num_features * 0.4))) 

    y = np.arange(num_features)
    bar_height = 0.6
    
    # Plot the bars
    ax.barh(y, means, bar_height,
           xerr=stds, capsize=5,
           color='#1f77b4', edgecolor='black')

    # Add plot details
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Mean Importance", fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names, fontsize=12)
    ax.set_xlim(left=0) # Importance should start at zero
    ax.invert_yaxis()

    # Customize the grid and spines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.grid(axis='y', visible=False)
    
    # Add text labels for the mean values next to the bars
    for i in range(num_features):
        offset = (means[0] + stds[0]) * 0.005
        x_pos = means[i] + stds[i] + offset
        ax.text(x_pos, y[i],
                f"{means[i]:.3f}",
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()



def _plot_n_bars(*dicts, feature_names, labels, title, filename = None):
    """
    Generate a grouped vertical bar plot comparing feature importance 
    across an arbitrary number of interpretability result sets, for every
    feature. 

    Each dictionary represents one protocol, methodology, or experimental condition. 
    Bars are grouped by feature, and each group contains one bar per dictionary. 
    Error bars represent the standard deviation of the importance scores.

    Parameters
    ----------
    *dicts : dict
        Variable number of aligned interpretability dictionaries.
        Each dictionary must contain:
            - 'importance_mean' : list or array of mean importance values
            - 'importance_std'  : list or array of standard deviations
        All dictionaries must be aligned to the same feature order.
    
    feature_names : list of str
        Ordered list of feature names corresponding to the aligned
        importance values.

    labels : list of str
        Labels for each dictionary (used in the legend). Must match
        the number of provided dictionaries.

    title : str
        Title of the figure.

    filename : str, optional
        If provided, the figure is saved at this path (300 dpi).
        If None, the figure is displayed interactively.

    """
    
    num_features = len(feature_names)
    num_protocols = len(dicts)
    
    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid')

    # Parameters for dynamic layout adjustments
    
    # Bar Width and Separation
    bar_width = 0.3 
    separation_factor = 0.08
    effective_width = bar_width + separation_factor
    
    # Horizontal Spacing
    gap_factor = 0.8 
    group_span = num_protocols * bar_width 
    x_step = group_span + gap_factor 
    x = np.arange(num_features) * x_step

    # Dynamic Figure Size
    fig_width = max(10, num_features * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    # Offsets centered around feature positions for n bars
    offsets = np.linspace(- (num_protocols - 1) / 2, (num_protocols - 1) / 2, num_protocols) * effective_width

    # Color Palette
    base_colors = plt.cm.tab10.colors  # Use a built-in color palette for up to 10 distinct colors
    
    # Plots n bars per feature, including error caps for standard deviation
    for i in range(len(dicts)):
        base_color = base_colors[i % len(base_colors)]
        
        ax.bar(x + offsets[i], dicts[i]['importance_mean'], bar_width, yerr=dicts[i]['importance_std'], capsize=3, linewidth=0.4, 
               label=labels[i], color=base_color, edgecolor='black', error_kw=dict(lw=0.7, capsize=2, capthick=0.7))

    # Add plot details 
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Mean Permutation Importance (Decrease in $R^2$ Score)", fontsize=12) 
    ax.set_xticks(x)
    # Rotate x-axis labels for readability
    ax.set_xticklabels(feature_names, rotation=45, ha='right')

    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', visible = False) 
    ax.grid(axis='x', visible=False) 
    
    # Add text labels above each bar for mean importance values
    all_means = [d["importance_mean"] for d in dicts]
    all_stds = [d["importance_std"] for d in dicts]
    
    for i in range(num_features):
        # Vertical divider for feature groups
        ax.axvline(x[i] - x_step/2, color="gray", linewidth=0.5, alpha=0.6)
        for j in range(len(dicts)):
            mean = all_means[j][i]
            std = all_stds[j][i]
            x_pos = x[i] + offsets[j] # Central position of the bar
            offset = (all_means[0][0] + all_stds[0][0]) * 0.02
            y_pos = mean + std + offset # Y position above the error bar
            
            # Only add label if mean importance is greater than 0.00
            if mean > 0.00:
                ax.text(x_pos, y_pos,
                        f"{mean:.3f}",
                        ha='center', va='bottom', fontsize=8, rotation=90) 
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()



def interpretability_comparison_plot(*dicts, labels, method = 'perm', title = None, filename = None):
    """
    Align and compare multiple interpretability result sets in a single
    grouped bar plot.

    This function accepts an arbitrary number of interpretability result
    dictionaries (e.g., different validation protocols, models, or 
    outlier configurations), aligns their feature order (based on the first dictionary 
    provided), and generates a comparative grouped bar visualization with each 
    feature with n vertical bars depending on the number of dictionaries provided. 
    Designed to be compatible with the provided intepretability functions on this package. 

    Parameters
    ----------
    *dicts : dict
        Variable number of interpretability result dictionaries.
        Each dictionary must correspond to a single experimental condition

    labels : list of str
        Legend labels corresponding to each dictionary. The number of
        labels must match the number of dictionaries.

    method : {'perm', 'shap'}, default='perm'
        Type of interpretability method used:
            - 'perm' : permutation importance
            - 'shap' : SHAP importance
        If 'shap' is selected, dictionaries are internally converted
        into global bar-plot format before alignment.

    title : str, optional
        Title of the resulting plot.

    filename : str, optional
        If provided, saves the plot at this location.
        If None, displays it interactively.

    Returns
    -------
    None
        Generates a grouped bar comparison plot.

    Raises
    ------
    ValueError
        If the number of labels does not match the number of dictionaries.

    """
     
    # Ensure labels and dictionaries provided are the same length
    if len(dicts) != len(labels):
        raise ValueError("Number of dictionaries and labels must be the same. For plotting purposes.")

    # Ensure all dictionaries are in global importance format
    if method == "shap":
        dicts = [_shap_barPlot_dictionary(d) for d in dicts]

    # Collect and align the six result dictionaries
    aligned_dicts = _align_interpretability_dicts(*dicts)
    
    # Features ordered consistently across all dictionaries
    features_ordered = aligned_dicts[0]['features']
    
    # Generate plot
    _plot_n_bars(*aligned_dicts, feature_names=features_ordered, 
                 labels=labels, title=title, filename=filename)