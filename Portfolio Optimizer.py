import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from financial_model import BiasedRandomizedPortfolioOptimizer, GeneticAlgorithmOptimizer
import visualization as viz
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

def setup_page():
    st.set_page_config(
        page_title="Portfolio Optimizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
        .sub-header { font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin-top: 2rem; margin-bottom: 1rem; }
        .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    if 'optimizer' not in st.session_state: st.session_state.optimizer = None
    if 'best_weights' not in st.session_state: st.session_state.best_weights = None
    if 'all_solutions' not in st.session_state: st.session_state.all_solutions = None
    if 'benchmark_results' not in st.session_state: st.session_state.benchmark_results = None
    if 'benchmark_weights' not in st.session_state: st.session_state.benchmark_weights = {}
    if 'data_loaded_from_file' not in st.session_state: st.session_state.data_loaded_from_file = False

def main():
    setup_page()
    init_session_state()
    
    st.markdown('# 📊 Hybrid Portfolio Optimizer', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("1. Application Mode")
        app_mode = st.radio("Select Mode", ["Standard Optimization", "Algorithm Benchmark"])
        st.divider()

    st.sidebar.header("2. Data Source")
    data_source = st.sidebar.radio("Source", ["New/Existing configuration", "Load from saved file"])
    
    selected_tickers = []
    
    if data_source == "Load from saved file":
        temp_opt = GeneticAlgorithmOptimizer(['AAPL'], '2020-01-01', '2023-01-01')
        saved_files = temp_opt.list_saved_data_files()
        
        if saved_files:
            file_options = {f"{info['filename']} ({info['num_tickers']} stocks)": info for info in saved_files}
            selected_file_key = st.sidebar.selectbox("Select saved data file", options=list(file_options.keys()))
            selected_file_info = file_options[selected_file_key]
            
            if st.sidebar.button("Load this file", type="primary"):
                with st.spinner("Loading data..."):
                    optimizer = GeneticAlgorithmOptimizer(['AAPL'], '2020-01-01', '2023-01-01')
                    if optimizer.load_data_from_file(selected_file_info['filepath']):
                        st.session_state.optimizer = optimizer
                        st.session_state.data_loaded_from_file = True
                        st.session_state.best_weights = None
                        st.session_state.benchmark_weights = {}
                        st.session_state.benchmark_results = None
                        st.success(f"✅ Loaded {len(optimizer.tickers)} stocks.")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load data.")
        else:
            st.sidebar.warning("No saved files found.")
            data_source = "New/Existing configuration"

    years_back = 1
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years_back)
    
    if data_source == "New/Existing configuration":
        preset_portfolios = {
            "Tech giants": ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            "Diversified": ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'JPM', 'BAC', 'JNJ', 'UNH', 'PG', 'KO', 'XOM', 'CVX', 'BA', 'NEE', 'AMT', 'DIS', 'NFLX'],
            "Custom": []
        }
        portfolio_choice = st.sidebar.selectbox("Select preset", list(preset_portfolios.keys()))
        
        if portfolio_choice == "Custom":
            custom_tickers = st.sidebar.text_area("Tickers (comma-separated)", "AAPL, MSFT, GOOGL")
            selected_tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
        else:
            selected_tickers = preset_portfolios[portfolio_choice]
            
        years_back = st.sidebar.slider("Years of history", 1, 10, 3)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years_back)
    risk_free_rate = st.sidebar.number_input("Risk-free rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
    use_saved_data = st.sidebar.checkbox("Use saved data", value=True)
    
    if app_mode == "Standard Optimization":
        run_standard_optimization(selected_tickers, start_date, end_date, risk_free_rate, use_saved_data if data_source == "New/Existing configuration" else False, data_source)
    else:
        run_benchmark_mode(selected_tickers, start_date, end_date, risk_free_rate, use_saved_data if data_source == "New/Existing configuration" else False, data_source)

def run_standard_optimization(tickers, start_date, end_date, risk_free_rate, use_saved, data_source):
    st.sidebar.header("3. Objective & Method")
    
    sharpe_weight = st.sidebar.slider("Sharpe Weight (γ)", 0.0, 1.0, 1.0, 0.05)
    return_weight = st.sidebar.slider("Return Weight (ω)", 0.0, 1.0, 0.0, 0.05)
    
    total = sharpe_weight + return_weight
    if total > 0:
        sharpe_weight /= total
        return_weight /= total

    method = st.sidebar.radio("Optimization Method", ["Biased-Randomization", "Genetic Algorithm"])

    if method == "Biased-Randomization":
        st.sidebar.markdown("**BR Parameters**")
        n_iters = st.sidebar.slider("Iterations", 10, 2000, 500, 10)
        beta = st.sidebar.slider("Beta (β)", 0.01, 0.99, 0.25, 0.01)
        use_geo = st.sidebar.radio("Distribution", ["Geometric", "Triangular"]) == "Geometric"
        alloc_range = st.sidebar.slider("Allocation range", 0.0, 1.0, (0.05, 0.60))
        local_search = st.sidebar.checkbox("Local Search", value=True)
        ls_prop = st.sidebar.slider("LS Proportion", 0.0, 1.0, 0.2, 0.05) if local_search else 0.0
    else:
        st.sidebar.markdown("**GA Parameters**")
        n_gen = st.sidebar.slider("Generations", 10, 500, 50, 10)
        pop_size = st.sidebar.slider("Population Size", 20, 500, 100, 20)
        cx_type = st.sidebar.selectbox("Crossover", ["Arithmetic", "Single Point"])
        cx_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8, 0.05)
        mut_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1, 0.01)
        elitism = st.sidebar.slider("Elitism", 0, 10, 2, 1)

    st.sidebar.subheader("Benchmarks")
    benchmarks = {
        'Greedy': st.sidebar.checkbox("Greedy", value=True),
        'Equal Weight': st.sidebar.checkbox("Equal Weight", value=True),
        'Min Variance': st.sidebar.checkbox("Min Variance", value=True),
        'Risk Parity': st.sidebar.checkbox("Risk Parity", value=True),
        'HRP': st.sidebar.checkbox("HRP", value=True),
        'Max Diversification': st.sidebar.checkbox("Max Diversification", value=True)
    }

    run_disabled = (sharpe_weight == 0 and return_weight == 0)
    
    if st.sidebar.button("Run Standard Optimization", type="primary", disabled=run_disabled):
        with st.spinner("Initializing..."):
            try:
                if data_source == "New/Existing configuration":
                    optimizer = GeneticAlgorithmOptimizer(
                        tickers=tickers,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        risk_free_rate=risk_free_rate,
                        sharpe_weight=sharpe_weight,
                        return_weight=return_weight
                    )
                    optimizer.get_data(use_saved=use_saved)
                else:
                    optimizer = st.session_state.optimizer
                    optimizer.sharpe_weight = sharpe_weight
                    optimizer.return_weight = return_weight
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, current_best):
                    progress_bar.progress(progress)
                    status_text.text(f"Best objective: {current_best:.4f}")

                if method == "Biased-Randomization":
                    optimizer.allocation_low = alloc_range[0]
                    optimizer.allocation_high = alloc_range[1]
                    
                    best_weights, best_obj, all_sols = optimizer.multi_start_biased_randomization(
                        n_iterations=n_iters, beta=beta, use_geometric=use_geo,
                        apply_local_search=local_search, local_search_prop=ls_prop,
                        progress_callback=update_progress
                    )
                else:
                    best_weights, best_obj, all_sols = optimizer.genetic_algorithm_optimization(
                        n_generations=n_gen, population_size=pop_size,
                        crossover_rate=cx_rate, mutation_rate=mut_rate,
                        crossover_type=cx_type, elitism_count=elitism,
                        progress_callback=update_progress
                    )
                
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.optimizer = optimizer
                st.session_state.best_weights = best_weights
                st.session_state.all_solutions = all_sols
                st.session_state.benchmark_weights = {}
                
                with st.spinner("Calculating benchmarks..."):
                    if benchmarks['Greedy']:
                        w, _ = optimizer.greedy_heuristic_sharpe()
                        st.session_state.benchmark_weights['Greedy'] = w
                    if benchmarks['Equal Weight']:
                        st.session_state.benchmark_weights['Equal Weight'] = optimizer.equal_weight_portfolio()
                    if benchmarks['Min Variance']:
                        st.session_state.benchmark_weights['Min Variance'] = optimizer.minimum_variance_portfolio()
                    if benchmarks['Risk Parity']:
                        st.session_state.benchmark_weights['Risk Parity'] = optimizer.risk_parity_portfolio()
                    if benchmarks['HRP']:
                        st.session_state.benchmark_weights['HRP'] = optimizer.hierarchical_risk_parity()
                    if benchmarks['Max Diversification']:
                        st.session_state.benchmark_weights['Max Diversification'] = optimizer.maximum_diversification_portfolio()
                
                st.success(f"Optimization completed using {method}!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state.optimizer and st.session_state.best_weights is not None:
        opt = st.session_state.optimizer
        bw = st.session_state.best_weights
        sols = st.session_state.all_solutions
        bench_w = st.session_state.benchmark_weights
        
        ret, std, sharpe, objective = opt.portfolio_performance(bw)

        st.markdown('<p class="sub-header">📊 Optimization results</p>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Exp. Return", f"{ret*100:.2f}%")
        c2.metric("Volatility", f"{std*100:.2f}%")
        c3.metric("Sharpe", f"{sharpe:.4f}")
        c4.metric("Objective", f"{objective:.4f}")
        c5.metric("Solutions", f"{len(sols)}")
        
        comparison_data = [{
            'Method': f'🏆 {method if "method" in locals() else "Optimized"}',
            'Return (%)': ret * 100,
            'Volatility (%)': std * 100,
            'Sharpe Ratio': sharpe,
            'Objective': objective
        }]
        
        for m_name, w in bench_w.items():
            r_b, s_b, sh_b, o_b = opt.portfolio_performance(w)
            comparison_data.append({
                'Method': m_name,
                'Return (%)': r_b * 100,
                'Volatility (%)': s_b * 100,
                'Sharpe Ratio': sh_b,
                'Objective': o_b
            })
            
        metrics_df = pd.DataFrame(comparison_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Allocation", "📈 Performance", "🎯 Frontier", 
            "📉 Risk", "🔬 Comparison", "💾 Data"
        ])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(viz.plot_allocation_pie(opt, bw), use_container_width=True)
            with c2: st.plotly_chart(viz.plot_weights_bar(opt, bw), use_container_width=True)
            w_df = pd.DataFrame({'Ticker': opt.tickers, 'Weight': bw}).sort_values('Weight', ascending=False)
            st.dataframe(w_df[w_df['Weight'] > 0.001].style.format({'Weight': '{:.2%}'}), use_container_width=True, hide_index=True)
            
        with tab2:
            st.plotly_chart(viz.plot_performance(opt, bw, bench_w), use_container_width=True)
            st.plotly_chart(viz.plot_rolling_metrics(opt, bw), use_container_width=True)
            
        with tab3:
            st.plotly_chart(viz.plot_efficient_frontier(opt, sols, (ret, std, sharpe, objective), bench_w), use_container_width=True)
            st.plotly_chart(viz.plot_convergence(sols, objective), use_container_width=True)
            
        with tab4:
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(viz.plot_return_distribution(opt, bw), use_container_width=True)
            with c2: st.plotly_chart(viz.plot_individual_sharpe(opt, sharpe), use_container_width=True)
            st.plotly_chart(viz.plot_covariance_heatmap(opt), use_container_width=True)
            
        with tab5:
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(viz.plot_metrics_comparison(metrics_df, 'Return (%)', 'lightblue', 'Annual Returns'), use_container_width=True)
            with c2: st.plotly_chart(viz.plot_metrics_comparison(metrics_df, 'Sharpe Ratio', 'lightgreen', 'Sharpe Ratio'), use_container_width=True)
            st.plotly_chart(viz.plot_metrics_comparison(metrics_df, 'Volatility (%)', 'salmon', 'Volatility'), use_container_width=True)
            
        with tab6:
            st.write(f"Stocks: {len(opt.tickers)}")
            st.write(f"Days: {len(opt.returns)}")
            st.write(f"Period: {opt.start_date} to {opt.end_date}")
            with st.expander("Tickers"):
                st.write(", ".join(opt.tickers))

def run_benchmark_mode(tickers, start_date, end_date, risk_free_rate, use_saved, data_source):
    st.sidebar.header("3. Benchmark Configurations")
    
    st.sidebar.subheader("Include Standard Benchmarks")
    std_benchmarks = {
        'Equal Weight': st.sidebar.checkbox("Equal Weight", value=True),
        'Min Variance': st.sidebar.checkbox("Min Variance", value=True),
        'Risk Parity': st.sidebar.checkbox("Risk Parity", value=True),
        'HRP': st.sidebar.checkbox("HRP", value=False),
        'Max Diversification': st.sidebar.checkbox("Max Diversification", value=False)
    }

    st.sidebar.subheader("Include Biased-Randomization")
    include_br = st.sidebar.checkbox("Biased-Randomization", value=True)
    if include_br:
        with st.sidebar.expander("BR Settings", expanded=False):
            br_iters = st.number_input("BR Iterations", 50, 2000, 100)
            br_beta = st.slider("BR Beta", 0.01, 0.99, 0.25, 0.01)
            br_geo = st.radio("BR Distribution", ["Geometric", "Triangular"]) == "Geometric"
            br_alloc = st.slider("BR Alloc Range", 0.0, 1.0, (0.05, 0.60))
            br_ls = st.checkbox("BR Local Search", value=True)
            br_ls_prop = st.slider("BR LS Prop", 0.0, 1.0, 0.2) if br_ls else 0.0
    
    st.sidebar.subheader("Define GA Variants")
    
    default_configs = [
        {'name': 'GA-Standard', 'selection': 'Tournament', 'crossover': 'Arithmetic', 'mutation': 'Gaussian'},
        {'name': 'GA-Genetic', 'selection': 'Roulette', 'crossover': 'Single Point', 'mutation': 'Swap'}
    ]
    
    configs_to_run = []
    num_variants = st.sidebar.number_input("Number of Variants", 1, 5, 2)
    
    for i in range(num_variants):
        with st.sidebar.expander(f"Variant {i+1}", expanded=(i==0)):
            name = st.text_input(f"Name {i+1}", f"Config {i+1}" if i >= len(default_configs) else default_configs[i]['name'])
            sel = st.selectbox(f"Selection {i+1}", ["Tournament", "Roulette"], index=0)
            cx = st.selectbox(f"Crossover {i+1}", ["Arithmetic", "Single Point", "Uniform"], index=0)
            mut = st.selectbox(f"Mutation {i+1}", ["Gaussian", "Swap"], index=0)
            cx_rate = st.slider(f"Crossover Rate {i+1}", 0.0, 1.0, 0.8, 0.05)
            mut_rate = st.slider(f"Mutation Rate {i+1}", 0.0, 1.0, 0.1, 0.01)
            
            configs_to_run.append({
                'name': name,
                'selection': sel,
                'crossover': cx,
                'mutation': mut,
                'crossover_rate': cx_rate,
                'mutation_rate': mut_rate,
                'elitism': 2
            })
            
    n_gen = st.sidebar.slider("Generations", 10, 200, 50)
    pop_size = st.sidebar.slider("Population Size", 20, 200, 100)

    if st.sidebar.button("🚀 Run Benchmark", type="primary"):
        status_box = st.empty()
        progress_bar = st.progress(0)
        
        try:
            if data_source == "New/Existing configuration":
                opt = GeneticAlgorithmOptimizer(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                opt.get_data(use_saved=use_saved)
            else:
                opt = st.session_state.optimizer
            
            st.session_state.optimizer = opt
            
            def on_progress(pct, msg):
                progress_bar.progress(pct)
                status_box.text(msg)
            
            results = opt.run_benchmark(configs_to_run, n_gen, pop_size, on_progress)
            
            if include_br:
                on_progress(0.9, "Running Biased-Randomization...")
                opt.allocation_low = br_alloc[0]
                opt.allocation_high = br_alloc[1]
                br_weights, br_obj, _ = opt.multi_start_biased_randomization(
                    n_iterations=br_iters, beta=br_beta, use_geometric=br_geo,
                    apply_local_search=br_ls, local_search_prop=br_ls_prop
                )
                br_ret, br_std, br_sharpe, _ = opt.portfolio_performance(br_weights)
                results['Biased-Randomization'] = {
                    'metrics': {'Return': br_ret, 'Risk': br_std, 'Sharpe': br_sharpe},
                    'objective': br_obj,
                    'history': [br_obj]*n_gen,
                    'weights': br_weights
                }

            on_progress(0.95, "Calculating Standard Benchmarks...")
            for name, enabled in std_benchmarks.items():
                if enabled:
                    if name == 'Equal Weight': w = opt.equal_weight_portfolio()
                    elif name == 'Min Variance': w = opt.minimum_variance_portfolio()
                    elif name == 'Risk Parity': w = opt.risk_parity_portfolio()
                    elif name == 'HRP': w = opt.hierarchical_risk_parity()
                    elif name == 'Max Diversification': w = opt.maximum_diversification_portfolio()
                    
                    ret, std, sharpe, obj = opt.portfolio_performance(w)
                    results[name] = {
                        'metrics': {'Return': ret, 'Risk': std, 'Sharpe': sharpe},
                        'objective': obj,
                        'history': [obj]*n_gen,
                        'weights': w
                    }

            st.session_state.benchmark_results = results
            status_box.success("Benchmark Complete!")
            progress_bar.empty()
            
        except Exception as e:
            status_box.error(f"Error: {str(e)}")

    if st.session_state.benchmark_results:
        results = st.session_state.benchmark_results
        
        tab_sum, tab_conv, tab_front, tab_detail = st.tabs([
            "🏆 Overview", "📈 Convergence", "🎯 Frontier", "🔬 Deep Dive"
        ])
        
        with tab_sum:
            st.subheader("Performance Metrics")
            comp_data = []
            for name, data in results.items():
                m = data['metrics']
                comp_data.append({
                    'Method': name,
                    'Objective': data['objective'],
                    'Return': f"{m['Return']:.2%}",
                    'Risk (Std)': f"{m['Risk']:.2%}",
                    'Sharpe Ratio': f"{m['Sharpe']:.4f}"
                })
            st.dataframe(pd.DataFrame(comp_data).sort_values('Objective', ascending=False), use_container_width=True)
        
        with tab_conv:
            st.subheader("Optimization Trajectory")
            fig = go.Figure()
            for name, data in results.items():
                if name not in std_benchmarks: 
                    fig.add_trace(go.Scatter(y=data['history'], mode='lines', name=name))
            fig.update_layout(title="Best Objective Value per Generation", xaxis_title="Generation", yaxis_title="Objective Value")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab_front:
            st.subheader("Risk-Return Tradeoff")
            fig_front = go.Figure()
            for name, data in results.items():
                m = data['metrics']
                fig_front.add_trace(go.Scatter(
                    x=[m['Risk']], y=[m['Return']],
                    mode='markers+text',
                    marker=dict(size=15),
                    name=name,
                    text=[name],
                    textposition="top center"
                ))
            fig_front.update_layout(title="Solution Landscape", xaxis_title="Risk (Volatility)", yaxis_title="Return")
            st.plotly_chart(fig_front, use_container_width=True)
            
        with tab_detail:
            st.subheader("Single Strategy Inspection")
            selected_method = st.selectbox("Select Method to Inspect", list(results.keys()))
            
            if selected_method:
                opt = st.session_state.optimizer
                weights = results[selected_method]['weights']
                
                t1, t2, t3, t4 = st.tabs(["📊 Allocation", "📈 Rolling Metrics", "📉 Returns", "💾 Weights Table"])
                
                with t1:
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(viz.plot_allocation_pie(opt, weights), use_container_width=True)
                    with c2: st.plotly_chart(viz.plot_weights_bar(opt, weights), use_container_width=True)
                
                with t2:
                    st.plotly_chart(viz.plot_rolling_metrics(opt, weights), use_container_width=True)
                    
                with t3:
                    st.plotly_chart(viz.plot_return_distribution(opt, weights), use_container_width=True)
                    
                with t4:
                    w_df = pd.DataFrame({'Ticker': opt.tickers, 'Weight': weights}).sort_values('Weight', ascending=False)
                    st.dataframe(w_df[w_df['Weight'] > 0.001].style.format({'Weight': '{:.2%}'}), use_container_width=True)

if __name__ == "__main__":
    main()