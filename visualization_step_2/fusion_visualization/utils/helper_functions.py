# Add these to your main script or create a new file called report_helpers.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel
import traceback
from typing import Tuple
def create_summary_dataframe(results: List[Dict], metrics: List[str]) -> pd.DataFrame:
    """
    Create overall summary dataframe from results
    """
    try:
        # Initialize data collection
        summary_data = []
        
        for result in results:
            if result is None:
                continue
                
            base_info = {
                'image': result['image'],
                'ground_truth_available': result['ground_truth_available']
            }
            
            # Add metrics for each method
            for method, values in result['metrics'].items():
                method_data = base_info.copy()
                method_data['method'] = method
                
                # Add each metric
                for metric in metrics:
                    method_data[metric] = values.get(metric, np.nan)
                
                # Add additional information
                if 'fusion' in method:
                    method_data['type'] = 'fusion'
                    method_data['combination'] = method.split('_')[1]
                    method_data['fusion_type'] = method.split('_')[-1]
                else:
                    method_data['type'] = 'individual'
                    method_data['combination'] = 'N/A'
                    method_data['fusion_type'] = 'N/A'
                
                summary_data.append(method_data)
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Add aggregate statistics
        df['mean_performance'] = df[metrics].mean(axis=1)
        df['std_performance'] = df[metrics].std(axis=1)
        
        return df
        
    except Exception as e:
        print(f"Error creating summary dataframe: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def create_method_analysis(results: List[Dict], metrics: List[str]) -> pd.DataFrame:
    """
    Create detailed method-wise analysis
    """
    try:
        # Collect method-wise statistics
        method_stats = {}
        
        for result in results:
            if result is None:
                continue
                
            for method, values in result['metrics'].items():
                if method not in method_stats:
                    method_stats[method] = {
                        'samples': 0,
                        'total_values': {metric: 0.0 for metric in metrics},
                        'squared_values': {metric: 0.0 for metric in metrics}
                    }
                
                # Update statistics
                stats = method_stats[method]
                stats['samples'] += 1
                
                for metric in metrics:
                    value = values.get(metric, 0.0)
                    stats['total_values'][metric] += value
                    stats['squared_values'][metric] += value * value
        
        # Calculate final statistics
        analysis_data = []
        
        for method, stats in method_stats.items():
            method_data = {'method': method}
            n = stats['samples']
            
            for metric in metrics:
                # Calculate mean
                mean = stats['total_values'][metric] / n
                method_data[f'{metric}_mean'] = mean
                
                # Calculate standard deviation
                squared_mean = stats['squared_values'][metric] / n
                variance = squared_mean - (mean * mean)
                method_data[f'{metric}_std'] = np.sqrt(max(0, variance))
                
                # Calculate confidence intervals
                ci = 1.96 * method_data[f'{metric}_std'] / np.sqrt(n)
                method_data[f'{metric}_ci_lower'] = mean - ci
                method_data[f'{metric}_ci_upper'] = mean + ci
            
            # Add method type
            method_data['type'] = 'fusion' if 'fusion' in method else 'individual'
            if 'fusion' in method:
                method_data['combination'] = method.split('_')[1]
                method_data['fusion_type'] = method.split('_')[-1]
            
            analysis_data.append(method_data)
        
        return pd.DataFrame(analysis_data)
        
    except Exception as e:
        print(f"Error creating method analysis: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def create_fusion_analysis(results: List[Dict]) -> pd.DataFrame:
    """
    Create detailed fusion analysis
    """
    try:
        fusion_data = []
        
        for result in results:
            if result is None or 'fusion_results' not in result:
                continue
                
            for fusion_name, fusion_result in result['fusion_results'].items():
                fusion_info = {
                    'image': result['image'],
                    'fusion_name': fusion_name,
                }
                
                # Add fusion type and combination
                parts = fusion_name.split('_')
                fusion_info['combination_type'] = parts[0]
                fusion_info['fusion_type'] = parts[-1]
                fusion_info['attention_types'] = '_'.join(parts[1:-1])
                
                # Add weights if available
                if 'weights' in fusion_result:
                    for att_type, weight in fusion_result['weights'].items():
                        fusion_info[f'weight_{att_type}'] = weight
                
                # Add other metrics
                if 'metrics' in result and fusion_name in result['metrics']:
                    metrics = result['metrics'][fusion_name]
                    fusion_info.update(metrics)
                
                fusion_data.append(fusion_info)
        
        df = pd.DataFrame(fusion_data)
        
        # Add performance ranking
        if 'f1_score' in df.columns:
            df['rank'] = df.groupby('image')['f1_score'].rank(ascending=False)
        
        return df
        
    except Exception as e:
        print(f"Error creating fusion analysis: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def create_statistical_analysis(results: List[Dict]) -> pd.DataFrame:
    """
    Create statistical analysis of methods
    """
    try:
        # Collect data for statistical tests
        method_data = {}
        metrics = ['precision', 'recall', 'f1_score', 'iou']
        
        for result in results:
            if result is None:
                continue
                
            for method, values in result['metrics'].items():
                if method not in method_data:
                    method_data[method] = {metric: [] for metric in metrics}
                
                for metric in metrics:
                    method_data[method][metric].append(values.get(metric, np.nan))
        
        # Perform statistical tests
        stats_data = []
        
        # Friedman test for each metric
        for metric in metrics:
            metric_values = []
            methods = []
            
            for method, values in method_data.items():
                metric_values.append(values[metric])
                methods.append(method)
            
            try:
                statistic, p_value = friedmanchisquare(*metric_values)
                stats_data.append({
                    'metric': metric,
                    'test': 'Friedman',
                    'statistic': statistic,
                    'p_value': p_value,
                    'methods': 'all'
                })
            except Exception as e:
                print(f"Error in Friedman test for {metric}: {str(e)}")
        
        # Pairwise Wilcoxon tests
        for metric in metrics:
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    try:
                        data1 = method_data[method1][metric]
                        data2 = method_data[method2][metric]
                        
                        # Remove NaN values
                        valid = ~(np.isnan(data1) | np.isnan(data2))
                        if sum(valid) < 2:
                            continue
                            
                        statistic, p_value = wilcoxon(
                            np.array(data1)[valid],
                            np.array(data2)[valid]
                        )
                        
                        stats_data.append({
                            'metric': metric,
                            'test': 'Wilcoxon',
                            'method1': method1,
                            'method2': method2,
                            'statistic': statistic,
                            'p_value': p_value
                        })
                    except Exception as e:
                        print(f"Error in Wilcoxon test for {method1} vs {method2}: {str(e)}")
        
        return pd.DataFrame(stats_data)
        
    except Exception as e:
        print(f"Error creating statistical analysis: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def generate_report_visualizations(results: List[Dict], report_dir: Path):
    """
    Generate comprehensive visualizations for the report
    """
    try:
        viz_dir = report_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Performance comparison plots
        generate_performance_plots(results, viz_dir)
        
        # Method comparison plots
        generate_method_comparison_plots(results, viz_dir)
        
        # Fusion analysis plots
        generate_fusion_analysis_plots(results, viz_dir)
        
        # Statistical visualization
        generate_statistical_plots(results, viz_dir)
        
    except Exception as e:
        print(f"Error generating report visualizations: {str(e)}")
        traceback.print_exc()

def generate_performance_plots(results: List[Dict], viz_dir: Path):
    """Generate performance comparison visualizations"""
    metrics = ['precision', 'recall', 'f1_score', 'iou']
    
    # Overall performance boxplot
    plt.figure(figsize=(15, 10))
    data = []
    
    for result in results:
        if result is None:
            continue
            
        for method, values in result['metrics'].items():
            for metric in metrics:
                data.append({
                    'Method': method,
                    'Metric': metric,
                    'Value': values.get(metric, np.nan)
                })
    
    df = pd.DataFrame(data)
    
    sns.boxplot(x='Metric', y='Value', hue='Method', data=df)
    plt.xticks(rotation=45)
    plt.title('Performance Comparison Across Methods')
    plt.tight_layout()
    plt.savefig(viz_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_method_comparison_plots(results: List[Dict], viz_dir: Path):
    """Generate method comparison visualizations"""
    # Method correlation heatmap
    plt.figure(figsize=(12, 10))
    
    method_correlations = calculate_method_correlations(results)
    sns.heatmap(method_correlations, annot=True, cmap='RdBu_r', center=0)
    plt.title('Method Correlation Matrix')
    plt.tight_layout()
    plt.savefig(viz_dir / 'method_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance over images
    plt.figure(figsize=(15, 8))
    
    performance_data = []
    for result in results:
        if result is None:
            continue
            
        for method, values in result['metrics'].items():
            performance_data.append({
                'Image': result['image'],
                'Method': method,
                'F1-Score': values.get('f1_score', np.nan)
            })
    
    df = pd.DataFrame(performance_data)
    sns.lineplot(data=df, x='Image', y='F1-Score', hue='Method')
    plt.xticks(rotation=45)
    plt.title('Performance Across Images')
    plt.tight_layout()
    plt.savefig(viz_dir / 'performance_across_images.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_fusion_analysis_plots(results: List[Dict], viz_dir: Path):
    """Generate fusion analysis visualizations"""
    # Fusion type comparison
    plt.figure(figsize=(12, 8))
    
    fusion_data = []
    for result in results:
        if result is None:
            continue
            
        for method, values in result['metrics'].items():
            if 'fusion' in method:
                fusion_type = method.split('_')[-1]
                fusion_data.append({
                    'Fusion Type': fusion_type,
                    'F1-Score': values.get('f1_score', np.nan)
                })
    
    df = pd.DataFrame(fusion_data)
    sns.violinplot(data=df, x='Fusion Type', y='F1-Score')
    plt.title('Performance by Fusion Type')
    plt.tight_layout()
    plt.savefig(viz_dir / 'fusion_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_statistical_plots(results: List[Dict], viz_dir: Path):
    """Generate statistical analysis visualizations"""
    # Performance distribution
    plt.figure(figsize=(15, 10))
    
    perf_data = []
    for result in results:
        if result is None:
            continue
            
        for method, values in result['metrics'].items():
            method_type = 'Fusion' if 'fusion' in method else 'Individual'
            perf_data.append({
                'Method Type': method_type,
                'Method': method,
                'F1-Score': values.get('f1_score', np.nan),
                'IoU': values.get('iou', np.nan)
            })
    
    df = pd.DataFrame(perf_data)
    
    plt.subplot(2, 1, 1)
    sns.violinplot(data=df, x='Method Type', y='F1-Score')
    plt.title('F1-Score Distribution by Method Type')
    
    plt.subplot(2, 1, 2)
    sns.violinplot(data=df, x='Method Type', y='IoU')
    plt.title('IoU Distribution by Method Type')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_method_correlations(results: List[Dict]) -> pd.DataFrame:
    """Calculate correlation between different methods"""
    method_values = {}
    
    for result in results:
        if result is None:
            continue
            
        for method, values in result['metrics'].items():
            if method not in method_values:
                method_values[method] = []
            method_values[method].append(values.get('f1_score', np.nan))
    
    # Create correlation matrix
    methods = list(method_values.keys())
    corr_matrix = np.zeros((len(methods), len(methods)))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            valid = ~(np.isnan(method_values[method1]) | np.isnan(method_values[method2]))
            if sum(valid) < 2:
                corr_matrix[i, j] = np.nan
                continue
                
            corr = np.corrcoef(
                np.array(method_values[method1])[valid],
                np.array(method_values[method2])[valid]
            )[0, 1]
            corr_matrix[i, j] = corr
    
    return pd.DataFrame
    
    
# Add these additional helper functions

def create_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create detailed performance comparison table
    """
    try:
        # Calculate key statistics for each method
        performance_stats = []
        metrics = ['precision', 'recall', 'f1_score', 'iou']
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            stats = {
                'method': method,
                'type': method_data['type'].iloc[0],
                'n_samples': len(method_data)
            }
            
            # Calculate statistics for each metric
            for metric in metrics:
                values = method_data[metric].dropna()
                stats.update({
                    f'{metric}_mean': values.mean(),
                    f'{metric}_std': values.std(),
                    f'{metric}_median': values.median(),
                    f'{metric}_min': values.min(),
                    f'{metric}_max': values.max(),
                    f'{metric}_ci_lower': values.mean() - 1.96 * values.std() / np.sqrt(len(values)),
                    f'{metric}_ci_upper': values.mean() + 1.96 * values.std() / np.sqrt(len(values))
                })
            
            performance_stats.append(stats)
        
        return pd.DataFrame(performance_stats)
        
    except Exception as e:
        print(f"Error creating performance table: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def create_ranking_analysis(results: List[Dict]) -> pd.DataFrame:
    """
    Create analysis of method rankings across images
    """
    try:
        ranking_data = []
        metrics = ['f1_score', 'iou']
        
        for result in results:
            if result is None:
                continue
                
            image_metrics = result['metrics']
            
            # Calculate rankings for each metric
            for metric in metrics:
                values = {method: metrics[metric] 
                         for method, metrics in image_metrics.items()}
                
                # Sort methods by metric value
                sorted_methods = sorted(values.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)
                
                # Assign rankings
                for rank, (method, value) in enumerate(sorted_methods, 1):
                    ranking_data.append({
                        'image': result['image'],
                        'method': method,
                        'metric': metric,
                        'value': value,
                        'rank': rank
                    })
        
        return pd.DataFrame(ranking_data)
        
    except Exception as e:
        print(f"Error creating ranking analysis: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def create_fusion_effectiveness_analysis(results: List[Dict]) -> pd.DataFrame:
    """
    Analyze effectiveness of different fusion strategies
    """
    try:
        fusion_effectiveness = []
        
        for result in results:
            if result is None:
                continue
                
            for method, metrics in result['metrics'].items():
                if 'fusion' not in method:
                    continue
                    
                # Parse fusion information
                parts = method.split('_')
                fusion_type = parts[-1]
                combination = '_'.join(parts[1:-1])
                
                # Get component methods
                component_methods = combination.split('-')
                component_scores = []
                
                # Get scores of component methods
                for comp_method in component_methods:
                    if comp_method in result['metrics']:
                        component_scores.append(
                            result['metrics'][comp_method]['f1_score']
                        )
                
                if component_scores:
                    effectiveness = {
                        'image': result['image'],
                        'fusion_type': fusion_type,
                        'combination': combination,
                        'fusion_score': metrics['f1_score'],
                        'mean_component_score': np.mean(component_scores),
                        'max_component_score': max(component_scores),
                        'improvement_over_mean': metrics['f1_score'] - np.mean(component_scores),
                        'improvement_over_max': metrics['f1_score'] - max(component_scores)
                    }
                    
                    fusion_effectiveness.append(effectiveness)
        
        return pd.DataFrame(fusion_effectiveness)
        
    except Exception as e:
        print(f"Error creating fusion effectiveness analysis: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def generate_detailed_visualizations(results: List[Dict], viz_dir: Path):
    """
    Generate additional detailed visualizations
    """
    try:
        # Create performance trend plots
        plot_performance_trends(results, viz_dir)
        
        # Create method comparison plots
        plot_method_comparisons(results, viz_dir)
        
        # Create fusion analysis plots
        plot_fusion_analysis(results, viz_dir)
        
        # Create statistical significance plots
        plot_statistical_significance(results, viz_dir)
        
    except Exception as e:
        print(f"Error generating detailed visualizations: {str(e)}")
        traceback.print_exc()

def plot_performance_trends(results: List[Dict], viz_dir: Path):
    """Plot performance trends across images"""
    plt.figure(figsize=(15, 10))
    
    # Collect data
    trend_data = []
    for result in results:
        if result is None:
            continue
            
        for method, metrics in result['metrics'].items():
            trend_data.append({
                'image': result['image'],
                'method': method,
                'f1_score': metrics['f1_score'],
                'type': 'fusion' if 'fusion' in method else 'individual'
            })
    
    df = pd.DataFrame(trend_data)
    
    # Create plot
    sns.lineplot(data=df, x='image', y='f1_score', 
                hue='method', style='type', markers=True)
    
    plt.xticks(rotation=45)
    plt.title('Performance Trends Across Images')
    plt.tight_layout()
    
    plt.savefig(viz_dir / 'performance_trends.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_method_comparisons(results: List[Dict], viz_dir: Path):
    """Plot detailed method comparisons"""
    metrics = ['precision', 'recall', 'f1_score', 'iou']
    
    plt.figure(figsize=(20, 15))
    
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, idx)
        
        data = []
        for result in results:
            if result is None:
                continue
                
            for method, metrics in result['metrics'].items():
                data.append({
                    'method': method,
                    'value': metrics[metric],
                    'type': 'fusion' if 'fusion' in method else 'individual'
                })
        
        df = pd.DataFrame(data)
        
        sns.boxplot(data=df, x='method', y='value', hue='type')
        plt.xticks(rotation=45)
        plt.title(f'{metric.upper()} Comparison')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'method_comparisons.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_fusion_analysis(results: List[Dict], viz_dir: Path):
    """Plot fusion analysis visualizations"""
    fusion_data = create_fusion_effectiveness_analysis(results)
    
    if fusion_data.empty:
        return
    
    # Plot improvement distributions
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    sns.boxplot(data=fusion_data, x='fusion_type', 
                y='improvement_over_mean')
    plt.title('Improvement Over Mean Component Score')
    
    plt.subplot(2, 1, 2)
    sns.boxplot(data=fusion_data, x='fusion_type', 
                y='improvement_over_max')
    plt.title('Improvement Over Max Component Score')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'fusion_improvements.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_statistical_significance(results: List[Dict], viz_dir: Path):
    """Plot statistical significance analysis"""
    stats_df = create_statistical_analysis(results)
    
    if stats_df.empty:
        return
    
    # Plot significance matrix
    plt.figure(figsize=(12, 10))
    
    significance_matrix = pd.pivot_table(
        stats_df[stats_df['test'] == 'Wilcoxon'],
        values='p_value',
        index='method1',
        columns='method2'
    )
    
    sns.heatmap(significance_matrix < 0.05, 
                annot=True, 
                cmap='RdYlGn_r',
                fmt='.2g')
    
    plt.title('Statistical Significance Matrix')
    plt.tight_layout()
    
    plt.savefig(viz_dir / 'significance_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Additional utility functions

def calculate_confidence_intervals(values: np.ndarray, 
                                confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence intervals"""
    mean = np.mean(values)
    std = np.std(values)
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * (std / np.sqrt(len(values)))
    return mean - margin, mean + margin

def format_metric_value(value: float, 
                       precision: int = 3, 
                       percentage: bool = True) -> str:
    """Format metric value for display"""
    if percentage:
        return f"{value*100:.{precision}f}%"
    return f"{value:.{precision}f}"

def calculate_effect_size(group1: np.ndarray, 
                         group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_se

def create_latex_tables(results_df: pd.DataFrame, 
                       output_dir: Path):
    """Create LaTeX tables for paper"""
    # Performance table
    performance_table = results_df.pivot_table(
        values=['f1_score', 'iou'],
        index='method',
        aggfunc=['mean', 'std']
    )
    
    with open(output_dir / 'performance_table.tex', 'w') as f:
        f.write(performance_table.to_latex(float_format="%.3f"))
    
    # Rankings table
    rankings_table = create_ranking_analysis(results_df)
    rankings_summary = rankings_table.groupby('method')['rank'].agg(['mean', 'std'])
    
    with open(output_dir / 'rankings_table.tex', 'w') as f:
        f.write(rankings_summary.to_latex(float_format="%.2f"))