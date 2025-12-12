import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from typing import Dict, Any, List, Optional

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
    metric_name: str, 
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
        template="plotly_white",
        legend_title="Model Version"
    )
    
    return fig

def plot_parameter_changes(param_changes: Dict[str, float]) -> go.Figure:
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
        template="plotly_white"
    )
    
    return fig

def visualize_all(
    trained_metrics: Dict[str, float], 
    base_metrics: Dict[str, float], 
    unlearned_metrics: Dict[str, float],
    param_diffs: Optional[Dict[str, float]] = None
):
    """
    Dispatcher function that generates and displays/saves all visualizations.
    
    Args:
        trained_metrics: Results from the original model.
        base_metrics: Results from the retrained (gold standard) model.
        unlearned_metrics: Results from the unlearned model.
        param_diffs: Dictionary containing parameter difference scalars.
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
        fig_params = plot_parameter_changes(param_diffs)
        fig_params.show()
        
    # Note: In a headless environment (like ClearML or Docker without display), 
    # you might want to use fig.write_html("path.html") instead of fig.show().