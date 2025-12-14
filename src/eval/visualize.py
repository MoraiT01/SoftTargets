import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import pandas as pd
from typing import Dict, Optional, Literal

from clearml import Task

def _extract_labels_and_values(metrics: Dict[str, float], metric_type: str) -> Dict[str, float]:
    """
    Extracts values for a specific metric type (e.g., 'accuracy' or 'loss') 
    from the flat results dictionary.
    
    Returns:
        Dict: {class_label: value}
    """
    # Pattern matches keys like "class_0_accuracy" or "class_cat_loss"
    # Captures the middle part as the label
    pattern = re.compile(fr"class_(.+)_{metric_type}")
    
    extracted = {}
    for key, value in metrics.items():
        match = pattern.match(key)
        if match:
            label = match.group(1)
            extracted[label] = value
            
    # Sort by label for consistent plotting
    return dict(sorted(extracted.items()))

def plot_metric_comparison(
    models_data: Dict[str, Dict[str, float]], 
    metric_name: Literal["accuracy", "loss"],
    title: str
) -> go.Figure:
    """
    Creates a grouped bar chart comparing a specific metric across different models and classes.
    
    Args:
        models_data: Dict mapping model names (e.g., "Trained") to their result dictionaries.
        metric_name: The suffix of the metric to plot (e.g., "accuracy", "loss").
        title: Chart title.
    """
    fig = go.Figure()
    eval_task = Task.init(project_name='softtargets', task_name='Metric Comparison')

    # Iterate over each model (Trained, Base, Unlearned)
    for model_name, results in models_data.items():
        # Extract the specific metric data (e.g., per-class accuracy)
        data_map = _extract_labels_and_values(results, metric_name)
        
        if not data_map:
            print(f"Warning: No data found for metric '{metric_name}' in model '{model_name}'")
            continue

        fig.add_trace(go.Bar(
            name=model_name,
            x=list(data_map.keys()),
            y=list(data_map.values()),
            text=[f"{v:.2f}" for v in data_map.values()],
            textposition='auto'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Class / Split",
        yaxis_title=metric_name.capitalize(),
        barmode='group',
        legend_title="Model Version"
    )
    
    return fig

def plot_parameter_changes(param_changes: Dict[str, float]):
    """
    Visualizes parameter differences (scalars).
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(param_changes.keys()),
        y=list(param_changes.values()),
        marker_color='indianred',
        text=[f"{v:.6f}" for v in param_changes.values()],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Average Parameter Change (vs Original Trained Model)",
        xaxis_title="Comparison",
        yaxis_title="Avg Abs Difference",
    )
    eval_task = Task.init(project_name='softtargets', task_name='Average Parameter Changes')

    eval_task.get_logger().report_scalar(title="Parameter Change", series="parameter_change", value=param_changes['avg_parameter_change'])

def plot_dataset_stats(df: pd.DataFrame, forget_col: str = "f1_split") -> None:
    """
    Generates visualizations for the dataset statistics:
    1. Stacked Histogram: Samples per class, colored by Forget/Retain status.
    2. Pie Chart: Total proportion of Forget vs Retain samples.
    
    Args:
        df: The dataframe containing the dataset metadata.
        forget_col: The column name indicating the forget split (1=Forget, 0=Retain).
    """
    eval_task = Task.init(project_name='softtargets', task_name='Dataset Stats')

    if forget_col not in df.columns:
        print(f"Warning: Column '{forget_col}' not found in dataframe. Skipping dataset stats plot.")
        return

    # Prepare data for plotting
    # Map 0/1 to descriptive labels
    df_plot = df.copy()
    df_plot['Status'] = df_plot[forget_col].map({1: 'Forget', 0: 'Retain'})
    
    # --- 1. Stacked Histogram (Samples per Class) ---
    # Group by Class and Status to get counts
    class_counts = df_plot.groupby(['Class_Label', 'Status']).size().reset_index(name='Count')
    
    fig_hist = go.Figure()
    
    for status, color in [('Retain', 'royalblue'), ('Forget', 'firebrick')]:
        subset = class_counts[class_counts['Status'] == status]
        fig_hist.add_trace(go.Bar(
            name=status,
            x=subset['Class_Label'],
            y=subset['Count'],
            marker_color=color,
            text=subset['Count'],
            textposition='auto'
        ))
        
    fig_hist.update_layout(
        title="Dataset Distribution: Samples per Class (Forget vs Retain)",
        xaxis_title="Class Label",
        yaxis_title="Number of Samples",
        barmode='stack', # Stacked histogram
    )
    eval_task.get_logger().report_plotly(title="Dataset Composition", series="dataset_composition", figure=fig_hist)
    
    # --- 2. Pie Chart (Total Unlearned vs Retained) ---
    total_counts = df_plot['Status'].value_counts()
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=total_counts.index, 
        values=total_counts.values,
        hole=.3, # Donut chart style looks nice
        marker_colors=['royalblue' if l == 'Retain' else 'firebrick' for l in total_counts.index]
    )])
    
    fig_pie.update_layout(
        title=f"Total Dataset Composition ({forget_col})",
        template="plotly_white"
    )
    eval_task.get_logger().report_plotly(title="Dataset Composition", series="dataset_composition", figure=fig_pie)

def visualize_all(
    trained_metrics: Dict[str, float], 
    base_metrics: Dict[str, float], 
    unlearned_metrics: Dict[str, float],
    param_diffs: Optional[Dict[str, float]] = None
):
    """
    Dispatcher function that generates and displays/saves all visualizations.
    """
    # 1. Organize data for the plotters
    models_map = {
        "Original (Trained)": trained_metrics,
        "Target (Retain-Only)": base_metrics,
        "Unlearned": unlearned_metrics
    }
    
    # 2. Dispatch to specific plotting functions
    print("Generating Accuracy Comparison Plot...")
    fig_acc = plot_metric_comparison(models_map, "accuracy", "Model Accuracy Comparison")
    fig_acc.show()
    
    print("Generating Loss Comparison Plot...")
    fig_loss = plot_metric_comparison(models_map, "loss", "Model Loss Comparison")
    fig_loss.show()
    
    # 3. Visualize Parameter Changes (if provided)
    if param_diffs:
        print("Generating Parameter Change Plot...")
        plot_parameter_changes(param_diffs)
        