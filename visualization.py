import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_allocation_pie(optimizer, weights):
    weights_to_plot = weights[weights > 0.001]
    tickers_to_plot = [optimizer.tickers[i] for i, w in enumerate(weights) if w > 0.001]
    
    fig = go.Figure(data=[go.Pie(
        labels=tickers_to_plot,
        values=weights_to_plot,
        hole=0.3,
        textinfo='label+percent',
        textposition='auto'
    )])
    fig.update_layout(title="Optimal portfolio allocation", height=500)
    return fig

def plot_weights_bar(optimizer, weights):
    fig = go.Figure(data=[go.Bar(
        x=optimizer.tickers,
        y=weights,
        marker_color=weights,
        marker_colorscale='Viridis',
        text=[f"{w*100:.1f}%" for w in weights],
        textposition='auto'
    )])
    fig.update_layout(
        title="Portfolio weights distribution",
        xaxis_title="Assets",
        yaxis_title="Weight",
        height=500,
        showlegend=False
    )
    return fig

def plot_performance(optimizer, best_weights, benchmark_weights):
    portfolio_value = optimizer.calculate_portfolio_value(best_weights)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value.values,
        mode='lines',
        name='Biased-Randomization',
        line=dict(color='darkblue', width=3)
    ))
    
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    for idx, (method_name, weights) in enumerate(benchmark_weights.items()):
        method_value = optimizer.calculate_portfolio_value(weights)
        fig.add_trace(go.Scatter(
            x=method_value.index,
            y=method_value.values,
            mode='lines',
            name=method_name,
            line=dict(color=colors[idx % len(colors)], width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Portfolio performance over time",
        xaxis_title="Date",
        yaxis_title="Cumulative value",
        height=600,
        hovermode='x unified'
    )
    return fig

def plot_rolling_metrics(optimizer, weights):
    portfolio_returns = (optimizer.returns * weights).sum(axis=1)
    rolling_returns = portfolio_returns.rolling(window=30).mean()
    rolling_std = portfolio_returns.rolling(window=30).std()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_returns.index,
        y=rolling_returns.values,
        mode='lines',
        name='30-day rolling mean',
        line=dict(color='darkgreen', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=rolling_returns.index,
        y=rolling_returns.values + rolling_std.values,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=rolling_returns.index,
        y=rolling_returns.values - rolling_std.values,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(0, 128, 0, 0.3)',
        fill='tonexty',
        name='± σ',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="Rolling performance analysis (30-day window)",
        xaxis_title="Date",
        yaxis_title="Rolling returns",
        height=500,
        hovermode='x unified'
    )
    return fig

def plot_efficient_frontier(optimizer, all_solutions, best_metrics, benchmark_weights):
    ret, std, sharpe, obj = best_metrics
    returns_data = [sol['return'] for sol in all_solutions]
    stds_data = [sol['std'] for sol in all_solutions]
    objectives_data = [sol['objective'] for sol in all_solutions]
    sharpes_data = [sol['sharpe'] for sol in all_solutions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=stds_data,
        y=returns_data,
        mode='markers',
        marker=dict(
            size=8,
            color=objectives_data,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Objective value"),
            line=dict(width=0.5, color='black')
        ),
        text=[f"Obj: {o:.3f}<br>Sharpe: {s:.3f}" for o, s in zip(objectives_data, sharpes_data)],
        hovertemplate='<b>Return:</b> %{y:.2%}<br><b>Risk:</b> %{x:.2%}<br>%{text}<extra></extra>',
        name='BR Solutions'
    ))
    
    fig.add_trace(go.Scatter(
        x=[std],
        y=[ret],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='black')),
        name='Best BR',
        hovertemplate='<b>Best BR</b><br>Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>'
    ))
    
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    symbols = ['square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'pentagon']
    
    for idx, (method_name, weights) in enumerate(benchmark_weights.items()):
        ret_b, std_b, _, _ = optimizer.portfolio_performance(weights)
        fig.add_trace(go.Scatter(
            x=[std_b],
            y=[ret_b],
            mode='markers',
            marker=dict(size=15, color=colors[idx % len(colors)], symbol=symbols[idx % len(symbols)], 
                       line=dict(width=2, color='black')),
            name=method_name,
            hovertemplate=f'<b>{method_name}</b><br>Return: %{{y:.2%}}<br>Risk: %{{x:.2%}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Efficient frontier comparison",
        xaxis_title="Volatility (risk)",
        yaxis_title="Expected return",
        height=600,
        hovermode='closest'
    )
    return fig

def plot_convergence(all_solutions, final_objective):
    iterations = [sol['iteration'] for sol in all_solutions]
    objectives = [sol['objective'] for sol in all_solutions]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations,
        y=objectives,
        mode='lines+markers',
        name='Objective value',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))
    fig.add_hline(y=final_objective, line_dash="dash", line_color="red", 
                  annotation_text=f"Best: {final_objective:.4f}", annotation_position="right")
    
    fig.update_layout(
        title="Convergence over iterations",
        xaxis_title="Iteration",
        yaxis_title="Objective value",
        height=500
    )
    return fig

def plot_return_distribution(optimizer, weights):
    portfolio_returns = (optimizer.returns * weights).sum(axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=portfolio_returns,
        nbinsx=50,
        name='Returns',
        marker_color='steelblue',
        opacity=0.7
    ))
    fig.add_vline(x=portfolio_returns.mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {portfolio_returns.mean():.4f}")
    fig.add_vline(x=portfolio_returns.median(), line_dash="dash", line_color="green",
                  annotation_text=f"Median: {portfolio_returns.median():.4f}")
    
    fig.update_layout(
        title="Portfolio returns distribution",
        xaxis_title="Daily returns",
        yaxis_title="Frequency",
        height=500
    )
    return fig

def plot_individual_sharpe(optimizer, portfolio_sharpe):
    individual_sharpes = []
    for i in range(len(optimizer.tickers)):
        w = np.zeros(len(optimizer.tickers))
        w[i] = 1.0
        _, _, sharpe_ind, _ = optimizer.portfolio_performance(w)
        individual_sharpes.append(sharpe_ind)
    
    colors_sharpe = ['green' if s > 0 else 'red' for s in individual_sharpes]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=optimizer.tickers,
        y=individual_sharpes,
        marker_color=colors_sharpe,
        text=[f"{s:.3f}" for s in individual_sharpes],
        textposition='auto'
    ))
    fig.add_hline(y=portfolio_sharpe, line_dash="dash", line_color="blue",
                  annotation_text=f"Portfolio: {portfolio_sharpe:.4f}")
    
    fig.update_layout(
        title="Individual asset Sharpe Ratios",
        xaxis_title="Assets",
        yaxis_title="Sharpe Ratio",
        height=500,
        showlegend=False
    )
    return fig

def plot_covariance_heatmap(optimizer):
    fig = go.Figure(data=go.Heatmap(
        z=optimizer.cov_matrix.values,
        x=optimizer.tickers,
        y=optimizer.tickers,
        colorscale='RdBu',
        zmid=0,
        text=optimizer.cov_matrix.values,
        texttemplate='%{text:.4f}',
        textfont={"size": 8},
        colorbar=dict(title="Covariance")
    ))
    fig.update_layout(title="Asset covariance matrix", height=600, xaxis={'side': 'bottom'})
    return fig

def plot_metrics_comparison(metrics_df, metric_col, color, title):
    fig = go.Figure(data=[go.Bar(
        x=metrics_df['Method'],
        y=metrics_df[metric_col],
        marker_color=color,
        text=metrics_df[metric_col].round(2),
        textposition='auto'
    )])
    fig.update_layout(
        title=title,
        xaxis_title="Method",
        yaxis_title=metric_col,
        height=400
    )
    return fig