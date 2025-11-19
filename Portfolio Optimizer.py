import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from financial_model import BiasedRandomizedPortfolioOptimizer
import visualization as viz

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
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    if 'best_weights' not in st.session_state:
        st.session_state.best_weights = None
    if 'all_solutions' not in st.session_state:
        st.session_state.all_solutions = None
    if 'benchmark_weights' not in st.session_state:
        st.session_state.benchmark_weights = {}
    if 'data_loaded_from_file' not in st.session_state:
        st.session_state.data_loaded_from_file = False

def main():
    setup_page()
    init_session_state()
    
    st.markdown('# 📊 Biased-Randomized Portfolio Optimizer', unsafe_allow_html=True)
    
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.subheader("💾 Data source")
    
    data_source = st.sidebar.radio(
        "Choose data source",
        ["New/Existing configuration", "Load from saved file"]
    )
    
    selected_tickers = []
    
    if data_source == "Load from saved file":
        temp_opt = BiasedRandomizedPortfolioOptimizer(['AAPL'], '2020-01-01', '2023-01-01')
        saved_files = temp_opt.list_saved_data_files()
        
        if saved_files:
            file_options = {f"{info['filename']} ({info['num_tickers']} stocks)": info for info in saved_files}
            selected_file_key = st.sidebar.selectbox("Select saved data file", options=list(file_options.keys()))
            selected_file_info = file_options[selected_file_key]
            
            if st.sidebar.button("Load this file", type="primary"):
                with st.spinner("Loading data..."):
                    optimizer = BiasedRandomizedPortfolioOptimizer(['AAPL'], '2020-01-01', '2023-01-01')
                    if optimizer.load_data_from_file(selected_file_info['filepath']):
                        st.session_state.optimizer = optimizer
                        st.session_state.data_loaded_from_file = True
                        st.session_state.best_weights = None
                        st.session_state.benchmark_weights = {}
                        st.success(f"✅ Loaded {len(optimizer.tickers)} stocks.")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load data.")
        else:
            st.sidebar.warning("No saved files found.")
            data_source = "New/Existing configuration"

    if data_source == "New/Existing configuration":
        st.sidebar.subheader("Stock selection")
        preset_portfolios = {
            "Tech giants": ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            "Diversified": ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'JPM', 'BAC', 'JNJ', 
                            'UNH', 'PG', 'KO', 'XOM', 'CVX', 'BA', 'NEE', 'AMT', 'DIS', 'NFLX'],
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

    st.sidebar.subheader("Objective Weights")
    sharpe_weight = st.sidebar.slider("Sharpe Weight (γ)", 0.0, 1.0, 1.0, 0.05)
    return_weight = st.sidebar.slider("Return Weight (ω)", 0.0, 1.0, 0.0, 0.05)
    
    total_weight = sharpe_weight + return_weight
    if total_weight > 0:
        sharpe_weight /= total_weight
        return_weight /= total_weight

    st.sidebar.subheader("🔧 Parameters")
    n_iterations = st.sidebar.slider("Iterations", 10, 2000, 500, 10)
    beta = st.sidebar.slider("Beta (β)", 0.01, 0.99, 0.25, 0.01)
    use_geometric = st.sidebar.radio("Distribution", ["Geometric", "Triangular"]) == "Geometric"
    allocation_range = st.sidebar.slider("Allocation range", 0.0, 1.0, (0.05, 0.60))
    apply_local_search = st.sidebar.checkbox("Local Search", value=True)
    local_search_prop = st.sidebar.slider("LS Proportion", 0.0, 1.0, 0.2, 0.05) if apply_local_search else 0.0

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
    
    if st.sidebar.button("Run optimization", type="primary", disabled=run_disabled):
        with st.spinner("Initializing..."):
            try:
                if data_source == "New/Existing configuration":
                    optimizer = BiasedRandomizedPortfolioOptimizer(
                        tickers=selected_tickers,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        risk_free_rate=risk_free_rate,
                        sharpe_weight=sharpe_weight,
                        return_weight=return_weight,
                        allocation_low=allocation_range[0],
                        allocation_high=allocation_range[1]
                    )
                    optimizer.get_data(use_saved=use_saved_data)
                else:
                    optimizer = st.session_state.optimizer
                    optimizer.sharpe_weight = sharpe_weight
                    optimizer.return_weight = return_weight
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, current_best):
                    progress_bar.progress(progress)
                    status_text.text(f"Best objective: {current_best:.4f}")

                best_weights, best_objective, all_solutions = optimizer.multi_start_biased_randomization(
                    n_iterations=n_iterations,
                    beta=beta,
                    use_geometric=use_geometric,
                    apply_local_search=apply_local_search,
                    local_search_prop=local_search_prop,
                    progress_callback=update_progress
                )
                
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.optimizer = optimizer
                st.session_state.best_weights = best_weights
                st.session_state.all_solutions = all_solutions
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
                
                st.success("Optimization completed!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state.optimizer and st.session_state.best_weights is not None:
        opt = st.session_state.optimizer
        bw = st.session_state.best_weights
        sols = st.session_state.all_solutions
        bench_w = st.session_state.benchmark_weights
        
        ret, std, sharpe, objective = opt.portfolio_performance(bw)

        with st.expander("📐 Mathematical formulation", expanded=False):
            st.markdown("### Portfolio optimization model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Portfolio return")
                st.latex(r"R_p = \sum_{i=1}^{n} w_i \cdot \mu_i \cdot 252")
                if st.session_state.optimizer is not None:
                    st.markdown(f"where $n = {len(st.session_state.optimizer.tickers)}$ assets")
                elif selected_tickers:
                    st.markdown(f"where $n = {len(selected_tickers)}$ assets")
                
                st.markdown("#### Portfolio variance")
                st.latex(r"\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} \cdot 252")
                
                st.markdown("#### Portfolio volatility")
                st.latex(r"\sigma_p = \sqrt{\sigma_p^2}")
                
            with col2:
                st.markdown("#### Sharpe Ratio")
                st.latex(r"SR = \frac{R_p - r_f}{\sigma_p}")
                if st.session_state.optimizer is not None:
                    st.markdown(f"where $r_f = {st.session_state.optimizer.risk_free_rate:.4f}$")
                elif data_source == "New/Existing Configuration":
                    st.markdown(f"where $r_f = {risk_free_rate:.4f}$")
                
                st.markdown("#### **Objective function** (Maximize)")
                st.latex(r"f(\mathbf{w}) = \gamma \cdot SR + \omega \cdot R_p")
                st.markdown(f"where $\\gamma = {sharpe_weight:.2f}$ (Sharpe weight)")
                st.markdown(f"and $\\omega = {return_weight:.2f}$ (Return weight)")
                
                if sharpe_weight == 1.0 and return_weight == 0.0:
                    st.info("📊 **Pure Sharpe Ratio optimization** (risk-adjusted)")
                elif sharpe_weight == 0.0 and return_weight == 1.0:
                    st.info("📈 **Pure Return optimization** (ignoring risk)")
                else:
                    st.info(f"⚖️ **Hybrid optimization** ({sharpe_weight*100:.0f}% risk-adjusted, {return_weight*100:.0f}% return-focused)")
            
            st.markdown("---")
            st.markdown("### Constraints")
            col1, col2 = st.columns(2)
            with col1:
                st.latex(r"\sum_{i=1}^{n} w_i = 1")
                st.markdown("*(weights sum to 1)*")
            with col2:
                st.latex(r"w_i \geq 0 \quad \forall i")
                st.markdown("*(no short selling)*")
            
            st.markdown("---")
            st.markdown("### Biased-Randomization")
            
            if use_geometric:
                st.markdown("#### Geometric distribution")
                st.latex(r"P(\text{select position } k) = \beta(1-\beta)^k")
                st.markdown(f"Current $\\beta = {beta:.3f}$")
            else:
                st.markdown("#### Triangular distribution")
                st.latex(r"k = \lfloor n(1 - \sqrt{U}) \rfloor")
                st.markdown("where $U \\sim \\text{Uniform}(0,1)$")
        
        st.markdown('<p class="sub-header">📊 Optimization results</p>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Exp. Return", f"{ret*100:.2f}%")
        c2.metric("Volatility", f"{std*100:.2f}%")
        c3.metric("Sharpe", f"{sharpe:.4f}")
        c4.metric("Objective", f"{objective:.4f}")
        c5.metric("Solutions", f"{len(sols)}/{n_iterations}")
        
        comparison_data = [{
            'Method': '🏆 Biased-Randomization',
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
    else:
        st.info("👈 Configure your portfolio parameters in the sidebar and click 'Run optimization' to begin!")
        
        st.markdown("""
        ### About this tool
        
        This application implements a **Biased-Randomized Portfolio Optimization** algorithm that combines:
        
        - **Multi-start heuristic approach** for exploring the solution space
        - **Biased randomization** using geometric or triangular distributions
        - **Local search optimization** for solution refinement
        - **Flexible objective function** combining Sharpe ratio and raw returns
        
        ### Getting started
        
        **Option 1: Load saved data**
        1. Select "Load from saved file" in the sidebar
        2. Choose a saved data file from the dropdown
        3. Click "Load this file"
        4. Configure optimization parameters
        5. Click "Run optimization"
        
        **Option 2: New configuration**
        1. Select your stocks or choose a preset portfolio
        2. Configure the date range and risk-free rate
        3. Adjust objective function weights (Sharpe vs. Return)
        4. Adjust optimization parameters (iterations, beta, etc.)
        5. Click "Run optimization" and wait for results
        6. Explore the interactive visualizations in the tabs
        """)

if __name__ == "__main__":
    main()