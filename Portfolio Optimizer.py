import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class BiasedRandomizedPortfolioOptimizer:
    
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02, data_dir='portfolio_data', 
                 sharpe_weight=1.0, return_weight=0.0):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.sharpe_weight = sharpe_weight
        self.return_weight = return_weight
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.prices = None
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def _get_data_filename(self):
        tickers_str = '_'.join(sorted(self.tickers))[:50]
        filename = f"data_{tickers_str}_{self.start_date}_{self.end_date}.pkl"
        return self.data_dir / filename
    
    def save_data(self):
        data_file = self._get_data_filename()
        data_dict = {
            'prices': self.prices,
            'returns': self.returns,
            'mean_returns': self.mean_returns,
            'cov_matrix': self.cov_matrix,
            'tickers': self.tickers,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'risk_free_rate': self.risk_free_rate,
            'fetch_timestamp': datetime.now()
        }
        
        with open(data_file, 'wb') as f:
            pickle.dump(data_dict, f)
    
    def load_data(self):
        data_file = self._get_data_filename()
        
        if not data_file.exists():
            return False
        
        try:
            with open(data_file, 'rb') as f:
                data_dict = pickle.load(f)
            
            self.prices = data_dict['prices']
            self.returns = data_dict['returns']
            self.mean_returns = data_dict['mean_returns']
            self.cov_matrix = data_dict['cov_matrix']
            self.tickers = data_dict['tickers']
            self.start_date = data_dict['start_date']
            self.end_date = data_dict['end_date']
            self.risk_free_rate = data_dict['risk_free_rate']
            
            return True
        except Exception as e:
            return False
    
    def load_data_from_file(self, filepath):
        """Load data from a specific file path"""
        try:
            with open(filepath, 'rb') as f:
                data_dict = pickle.load(f)
            
            self.prices = data_dict['prices']
            self.returns = data_dict['returns']
            self.mean_returns = data_dict['mean_returns']
            self.cov_matrix = data_dict['cov_matrix']
            self.tickers = data_dict['tickers']
            self.start_date = data_dict['start_date']
            self.end_date = data_dict['end_date']
            self.risk_free_rate = data_dict.get('risk_free_rate', 0.02)
            
            return True
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def check_saved_data_exists(self):
        data_file = self._get_data_filename()
        return data_file.exists()
    
    def list_saved_data_files(self):
        data_files = list(self.data_dir.glob('data_*.pkl'))
        saved_data_info = []
        
        for file in data_files:
            try:
                with open(file, 'rb') as f:
                    data_dict = pickle.load(f)
                saved_data_info.append({
                    'filename': file.name,
                    'filepath': file,
                    'tickers': data_dict.get('tickers', []),
                    'start_date': data_dict.get('start_date', 'Unknown'),
                    'end_date': data_dict.get('end_date', 'Unknown'),
                    'fetch_timestamp': data_dict.get('fetch_timestamp', 'Unknown'),
                    'num_tickers': len(data_dict.get('tickers', []))
                })
            except:
                pass
        
        return saved_data_info
        
    def fetch_data(self, save_after_fetch=True):
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False, threads=False)
        
        if data.empty:
            raise ValueError("No data downloaded. Check tickers and date range.")
        
        if len(self.tickers) == 1:
            if 'Adj Close' in data.columns:
                prices = data['Adj Close'].to_frame()
            else:
                prices = data[['Close']].copy()
            prices.columns = self.tickers
        else:
            if isinstance(data.columns, pd.MultiIndex):
                if 'Adj Close' in data.columns.get_level_values(0):
                    prices = data['Adj Close'].copy()
                else:
                    prices = data['Close'].copy()
            else:
                if 'Adj Close' in data.columns:
                    prices = data['Adj Close'].to_frame()
                else:
                    prices = data['Close'].to_frame()
                prices.columns = self.tickers
        
        prices = prices.dropna()
        
        if prices.empty:
            raise ValueError("No valid price data after cleaning.")
        
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        
        if self.returns.empty or len(self.returns) < 2:
            raise ValueError("Insufficient return data.")
        
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        if save_after_fetch:
            self.save_data()
        
        return self.returns
    
    def get_data(self, use_saved=True):
        if use_saved and self.check_saved_data_exists():
            success = self.load_data()
            if success:
                return self.returns
        
        self.fetch_data(save_after_fetch=True)
        return self.returns
    
    def portfolio_performance(self, weights):
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            return 0, np.inf, -np.inf, -np.inf
        
        portfolio_return = np.sum(self.mean_returns.values * weights) * 252
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix.values * 252, weights))
        
        if portfolio_variance < 0:
            portfolio_variance = 0
        
        portfolio_std = np.sqrt(portfolio_variance)
        
        if portfolio_std == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        combined_objective = self.sharpe_weight * sharpe_ratio + self.return_weight * portfolio_return
        
        return portfolio_return, portfolio_std, sharpe_ratio, combined_objective
    
    def greedy_heuristic_sharpe(self):
        n_assets = len(self.tickers)
        sharpe_ratios = []
        
        for i in range(n_assets):
            weights = np.zeros(n_assets)
            weights[i] = 1.0
            _, _, sharpe, _ = self.portfolio_performance(weights)
            sharpe_ratios.append(sharpe)
        
        sorted_indices = np.argsort(sharpe_ratios)[::-1]
        
        weights = np.zeros(n_assets)
        remaining_weight = 1.0
        
        for idx in sorted_indices[:-1]:
            allocation = remaining_weight * 0.3
            weights[idx] = allocation
            remaining_weight -= allocation
        
        weights[sorted_indices[-1]] = remaining_weight
        
        weights = np.clip(weights, 0, 1)
        weights = weights / np.sum(weights)
        
        return weights, sorted_indices
    
    def geometric_distribution_selection(self, sorted_list, beta):
        n = len(sorted_list)
        if n == 0:
            return None
        
        rand_val = np.random.random()
        if rand_val == 0:
            rand_val = 1e-10
        
        index = int(np.log(rand_val) / np.log(1 - beta)) % n
        return sorted_list[index]
    
    def triangular_distribution_selection(self, sorted_list):
        n = len(sorted_list)
        if n == 0:
            return None
        
        rand_val = np.random.random()
        index = int(n * (1 - np.sqrt(rand_val)))
        index = min(index, n - 1)
        return sorted_list[index]
    
    def biased_randomized_construction(self, beta=0.25, use_geometric=True):
        n_assets = len(self.tickers)
        objective_values = []
        
        for i in range(n_assets):
            weights = np.zeros(n_assets)
            weights[i] = 1.0
            _, _, _, obj = self.portfolio_performance(weights)
            objective_values.append((i, obj))
        
        sorted_assets = sorted(objective_values, key=lambda x: x[1], reverse=True)
        sorted_indices = [x[0] for x in sorted_assets]
        
        weights = np.zeros(n_assets)
        remaining_weight = 1.0
        selected_assets = []
        
        while remaining_weight > 0.01 and len(selected_assets) < n_assets:
            available_indices = [idx for idx in sorted_indices if idx not in selected_assets]
            if not available_indices:
                break
            
            if use_geometric:
                selected_idx = self.geometric_distribution_selection(available_indices, beta)
            else:
                selected_idx = self.triangular_distribution_selection(available_indices)
            
            if selected_idx is None:
                break
            
            selected_assets.append(selected_idx)
            allocation = remaining_weight * np.random.uniform(0.05, 0.60)
            weights[selected_idx] = allocation
            remaining_weight -= allocation
        
        if remaining_weight > 0:
            if len(selected_assets) > 0:
                weights[selected_assets[-1]] += remaining_weight
            else:
                weights[sorted_indices[0]] = 1.0
        
        weights = np.clip(weights, 0, 1)
        weights = weights / np.sum(weights)
        
        return weights
    
    def local_search_optimization(self, initial_weights):
        n_assets = len(self.tickers)
        
        def negative_objective(weights):
            _, _, _, obj = self.portfolio_performance(weights)
            if np.isnan(obj) or np.isinf(obj):
                return 1e10
            return -obj
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        try:
            result = minimize(negative_objective, initial_weights, method='SLSQP',
                             bounds=bounds, constraints=constraints, 
                             options={'maxiter': 100, 'ftol': 1e-6})
            
            if result.success and not np.any(np.isnan(result.x)):
                optimized_weights = np.clip(result.x, 0, 1)
                optimized_weights = optimized_weights / np.sum(optimized_weights)
                return optimized_weights
            else:
                return initial_weights
        except:
            return initial_weights
    
    def multi_start_biased_randomization(self, n_iterations=50, beta=0.25, 
                                        use_geometric=True, apply_local_search=True, local_search_prop=0.5):
        best_weights = None
        best_objective = -np.inf
        all_solutions = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(n_iterations):
            try:
                weights = self.biased_randomized_construction(beta, use_geometric)
                
                if apply_local_search and np.random.random() < local_search_prop:
                    weights = self.local_search_optimization(weights)                
                
                ret, std, sharpe, obj = self.portfolio_performance(weights)
                
                if not np.isnan(obj) and not np.isinf(obj):
                    all_solutions.append({
                        'iteration': i,
                        'weights': weights,
                        'return': ret,
                        'std': std,
                        'sharpe': sharpe,
                        'objective': obj
                    })
                    
                    if obj > best_objective:
                        best_objective = obj
                        best_weights = weights.copy()
                
                progress_bar.progress((i + 1) / n_iterations)
                status_text.text(f"Iteration {i+1}/{n_iterations} - Best objective: {best_objective:.4f}")
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return best_weights, best_objective, all_solutions
    
    def calculate_portfolio_value(self, weights):
        portfolio_returns = (self.returns * weights).sum(axis=1)
        portfolio_value = (1 + portfolio_returns).cumprod()
        return portfolio_value

if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'best_weights' not in st.session_state:
    st.session_state.best_weights = None
if 'all_solutions' not in st.session_state:
    st.session_state.all_solutions = None
if 'greedy_weights' not in st.session_state:
    st.session_state.greedy_weights = None
if 'data_loaded_from_file' not in st.session_state:
    st.session_state.data_loaded_from_file = False

st.markdown('<p class="main-header">📊 Biased-Randomized Portfolio Optimizer</p>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("💾 Data Source")
data_source = st.sidebar.radio(
    "Choose data source",
    ["New/Existing configuration", "Load from saved file"],
    help="Select whether to configure stocks manually or load from a saved data file"
)

selected_tickers = []

if data_source == "Load from saved file":
    temp_optimizer = BiasedRandomizedPortfolioOptimizer(
        tickers=['AAPL'],
        start_date='2020-01-01',
        end_date='2023-01-01',
        data_dir='portfolio_data'
    )
    
    saved_files = temp_optimizer.list_saved_data_files()
    
    if saved_files:
        file_options = {
            f"{info['filename']} ({info['num_tickers']} stocks, {info['start_date']} to {info['end_date']})": info
            for info in saved_files
        }
        
        selected_file_key = st.sidebar.selectbox(
            "Select saved data file",
            options=list(file_options.keys())
        )
        
        selected_file_info = file_options[selected_file_key]
        
        st.sidebar.info(f"""
        **Selected file details:**
        - **Stocks:** {', '.join(selected_file_info['tickers'][:5])}{'...' if len(selected_file_info['tickers']) > 5 else ''}
        - **Total:** {selected_file_info['num_tickers']} stocks
        - **Period:** {selected_file_info['start_date']} to {selected_file_info['end_date']}
        - **Fetched:** {selected_file_info['fetch_timestamp']}
        """)
        
        if st.sidebar.button("📥 Load this file", type="primary"):
            with st.spinner("Loading data from file..."):
                optimizer = BiasedRandomizedPortfolioOptimizer(
                    tickers=['AAPL'], # To be overwritten
                    start_date='2020-01-01',
                    end_date='2023-01-01',
                    data_dir='portfolio_data'
                )
                
                success = optimizer.load_data_from_file(selected_file_info['filepath'])
                
                if success:
                    st.session_state.optimizer = optimizer
                    st.session_state.data_loaded_from_file = True
                    st.session_state.best_weights = None
                    st.session_state.all_solutions = None
                    st.success(f"✅ Data loaded successfully! {len(optimizer.tickers)} stocks loaded.")
                    st.rerun()
                else:
                    st.error("❌ Failed to load data file")
    else:
        st.sidebar.warning("No saved data files found in portfolio_data directory")
        data_source = "New/Existing configuration"

if data_source == "New/Existing configuration":
    st.sidebar.subheader("📈 Stock Selection")
    
    preset_portfolios = {
        "Tech giants": ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        "Diversified portfolio": ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'JPM', 'BAC', 'JNJ', 
                                  'UNH', 'PG', 'KO', 'XOM', 'CVX', 'BA', 'NEE', 'AMT', 'DIS', 'NFLX'],
        "Custom": []
    }
    
    portfolio_choice = st.sidebar.selectbox("Select portfolio preset", list(preset_portfolios.keys()))
    
    if portfolio_choice == "Custom":
        custom_tickers = st.sidebar.text_area(
            "Enter stock tickers (comma-separated)",
            "AAPL, MSFT, GOOGL, TSLA, NVDA",
            help="Enter stock tickers separated by commas"
        )
        selected_tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
    else:
        selected_tickers = preset_portfolios[portfolio_choice]
        st.sidebar.info(f"Selected {len(selected_tickers)} stocks")
    
    st.sidebar.subheader("📅 Date range")
    years_back = st.sidebar.slider("Years of historical data", 1, 10, 3)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years_back)
    
    risk_free_rate = st.sidebar.number_input(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1
    ) / 100
    
    # Data management
    use_saved_data = st.sidebar.checkbox("Use saved data (if available)", value=True)

# Objective function weights
st.sidebar.subheader("🎯 Objective function weights")
st.sidebar.markdown("Define the optimization objective as a weighted combination:")

sharpe_weight = st.sidebar.slider(
    "Sharpe Ratio weight (γ)",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help="Weight for risk-adjusted returns (Sharpe Ratio)"
)

return_weight = st.sidebar.slider(
    "Raw Return weight (ω)",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Weight for absolute returns (ignoring risk)"
)

if sharpe_weight == 0 and return_weight == 0:
    st.sidebar.error("⚠️ At least one weight must be greater than 0")

# Optimization parameters
st.sidebar.subheader("🔧 Optimization Parameters")
n_iterations = st.sidebar.slider("Number of iterations", 10, 2000, 500, step=10)
beta = st.sidebar.slider("Beta parameter (β)", 0.01, 0.99, 0.25, step=0.01)
use_geometric = st.sidebar.radio("Distribution type", ["Geometric", "Triangular"]) == "Geometric"
apply_local_search = st.sidebar.checkbox("Apply local search", value=True)
local_search_prop = st.sidebar.slider("Local search proportion", 0.0, 1.0, 0.2, step=0.05) if apply_local_search else 0.0
compare_greedy = st.sidebar.checkbox("Compare with greedy heuristic", value=True)

# Display mathematical formulation
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

# Run optimization button
run_button_disabled = (sharpe_weight == 0 and return_weight == 0)

if data_source == "New/Existing configuration":
    if st.sidebar.button("🚀 Run optimization", type="primary", disabled=run_button_disabled):
        with st.spinner("Initializing optimizer..."):
            try:
                optimizer = BiasedRandomizedPortfolioOptimizer(
                    tickers=selected_tickers,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    risk_free_rate=risk_free_rate,
                    data_dir='portfolio_data',
                    sharpe_weight=sharpe_weight,
                    return_weight=return_weight
                )
                
                st.info("📥 Fetching market data...")
                optimizer.get_data(use_saved=use_saved_data)
                
                st.success(f"✅ Data loaded: {len(optimizer.returns)} trading days")
                
                st.info("🔄 Running optimization algorithm...")
                best_weights, best_objective, all_solutions = optimizer.multi_start_biased_randomization(
                    n_iterations=n_iterations,
                    beta=beta,
                    use_geometric=use_geometric,
                    apply_local_search=apply_local_search,
                    local_search_prop=local_search_prop
                )
                
                st.session_state.optimizer = optimizer
                st.session_state.best_weights = best_weights
                st.session_state.all_solutions = all_solutions
                st.session_state.data_loaded_from_file = False
                
                if compare_greedy:
                    greedy_weights, _ = optimizer.greedy_heuristic_sharpe()
                    st.session_state.greedy_weights = greedy_weights
                
                st.success("✅ Optimization completed!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
elif st.session_state.data_loaded_from_file and st.session_state.optimizer is not None:
    st.session_state.optimizer.sharpe_weight = sharpe_weight
    st.session_state.optimizer.return_weight = return_weight
    
    if st.sidebar.button("🚀 Run optimization", type="primary", disabled=run_button_disabled):
        with st.spinner("Running optimization on loaded data..."):
            try:
                optimizer = st.session_state.optimizer
                
                st.info("🔄 Running optimization algorithm...")
                best_weights, best_objective, all_solutions = optimizer.multi_start_biased_randomization(
                    n_iterations=n_iterations,
                    beta=beta,
                    use_geometric=use_geometric,
                    apply_local_search=apply_local_search,
                    local_search_prop=local_search_prop
                )
                
                st.session_state.best_weights = best_weights
                st.session_state.all_solutions = all_solutions
                
                if compare_greedy:
                    greedy_weights, _ = optimizer.greedy_heuristic_sharpe()
                    st.session_state.greedy_weights = greedy_weights
                
                st.success("✅ Optimization completed!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if st.session_state.optimizer is not None and st.session_state.best_weights is not None:
    optimizer = st.session_state.optimizer
    best_weights = st.session_state.best_weights
    all_solutions = st.session_state.all_solutions
    greedy_weights = st.session_state.greedy_weights
    
    ret, std, sharpe, objective = optimizer.portfolio_performance(best_weights)
    
    st.markdown('<p class="sub-header">📊 Optimization results</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Expected Annual Return", f"{ret*100:.2f}%", delta=None)
    with col2:
        st.metric("Annual Volatility", f"{std*100:.2f}%", delta=None)
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.4f}", delta=None)
    with col4:
        st.metric("Objective value", f"{objective:.4f}", delta=None)
    with col5:
        st.metric("Successful iterations", f"{len(all_solutions)}/{n_iterations}", delta=None)
    
    # Comparison with greedy
    if greedy_weights is not None:
        ret_greedy, std_greedy, sharpe_greedy, obj_greedy = optimizer.portfolio_performance(greedy_weights)
        improvement = ((objective - obj_greedy) / abs(obj_greedy)) * 100 if obj_greedy != 0 else 0
        
        st.info(f"📈 **Improvement over Greedy Heuristic:** {improvement:.2f}% | Greedy Objective: {obj_greedy:.4f} (Sharpe: {sharpe_greedy:.4f}, Return: {ret_greedy*100:.2f}%)")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio allocation", "📈 Performance", "🎯 Efficient frontier", "📉 Risk analysis", "💾 Data info"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            weights_to_plot = best_weights[best_weights > 0.001]
            tickers_to_plot = [optimizer.tickers[i] for i, w in enumerate(best_weights) if w > 0.001]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=tickers_to_plot,
                values=weights_to_plot,
                hole=0.3,
                textinfo='label+percent',
                textposition='auto'
            )])
            fig_pie.update_layout(
                title="Optimal portfolio allocation",
                height=500
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = go.Figure(data=[go.Bar(
                x=optimizer.tickers,
                y=best_weights,
                marker_color=best_weights,
                marker_colorscale='Viridis',
                text=[f"{w*100:.1f}%" for w in best_weights],
                textposition='auto'
            )])
            fig_bar.update_layout(
                title="Portfolio weights distribution",
                xaxis_title="Assets",
                yaxis_title="Weight",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Weights table
        st.markdown("### 📋 Detailed allocation")
        weights_df = pd.DataFrame({
            'Ticker': optimizer.tickers,
            'Weight': best_weights,
            'Percentage': [f"{w*100:.2f}%" for w in best_weights]
        }).sort_values('Weight', ascending=False)
        
        st.dataframe(weights_df[weights_df['Weight'] > 0.001], use_container_width=True, hide_index=True)
    
    with tab2:
        # Portfolio value over time
        portfolio_value = optimizer.calculate_portfolio_value(best_weights)
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            mode='lines',
            name='Optimal portfolio',
            line=dict(color='darkblue', width=3)
        ))
        
        if greedy_weights is not None:
            greedy_value = optimizer.calculate_portfolio_value(greedy_weights)
            fig_perf.add_trace(go.Scatter(
                x=greedy_value.index,
                y=greedy_value.values,
                mode='lines',
                name='Greedy portfolio',
                line=dict(color='orange', width=2, dash='dash')
            ))
        
        # Add top 5 individual assets
        for ticker in optimizer.tickers[:5]:
            if ticker in optimizer.returns.columns:
                asset_value = (1 + optimizer.returns[ticker]).cumprod()
                fig_perf.add_trace(go.Scatter(
                    x=asset_value.index,
                    y=asset_value.values,
                    mode='lines',
                    name=ticker,
                    line=dict(width=1),
                    opacity=0.5
                ))
        
        fig_perf.update_layout(
            title="Portfolio performance over time",
            xaxis_title="Date",
            yaxis_title="Cumulative value",
            height=600,
            hovermode='x unified'
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Rolling performance
        portfolio_returns = (optimizer.returns * best_weights).sum(axis=1)
        rolling_returns = portfolio_returns.rolling(window=30).mean()
        rolling_std = portfolio_returns.rolling(window=30).std()
        
        fig_rolling = go.Figure()
        fig_rolling.add_trace(go.Scatter(
            x=rolling_returns.index,
            y=rolling_returns.values,
            mode='lines',
            name='30-day rolling mean',
            line=dict(color='darkgreen', width=2)
        ))
        fig_rolling.add_trace(go.Scatter(
            x=rolling_returns.index,
            y=rolling_returns.values + rolling_std.values,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig_rolling.add_trace(go.Scatter(
            x=rolling_returns.index,
            y=rolling_returns.values - rolling_std.values,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0, 128, 0, 0.3)',
            fill='tonexty',
            name='± σ',
            hoverinfo='skip'
        ))
        
        fig_rolling.update_layout(
            title="Rolling performance analysis (30-day window)",
            xaxis_title="Date",
            yaxis_title="Rolling returns",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_rolling, use_container_width=True)
    
    with tab3:
        returns_data = [sol['return'] for sol in all_solutions]
        stds_data = [sol['std'] for sol in all_solutions]
        sharpes_data = [sol['sharpe'] for sol in all_solutions]
        objectives_data = [sol['objective'] for sol in all_solutions]
        
        fig_frontier = go.Figure()
        
        fig_frontier.add_trace(go.Scatter(
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
            name='Solutions'
        ))
        
        fig_frontier.add_trace(go.Scatter(
            x=[std],
            y=[ret],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='black')),
            name='Best solution',
            hovertemplate='<b>Best solution</b><br>Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>'
        ))
        
        if greedy_weights is not None:
            fig_frontier.add_trace(go.Scatter(
                x=[std_greedy],
                y=[ret_greedy],
                mode='markers',
                marker=dict(size=20, color='orange', symbol='square', line=dict(width=2, color='black')),
                name='Greedy solution',
                hovertemplate='<b>Greedy solution</b><br>Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>'
            ))
        
        fig_frontier.update_layout(
            title="Efficient frontier - all solutions",
            xaxis_title="Volatility (risk)",
            yaxis_title="Expected return",
            height=600,
            hovermode='closest'
        )
        st.plotly_chart(fig_frontier, use_container_width=True)
        
        # Convergence plot
        iterations = [sol['iteration'] for sol in all_solutions]
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=iterations,
            y=objectives_data,
            mode='lines+markers',
            name='Objective value',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ))
        fig_conv.add_hline(y=objective, line_dash="dash", line_color="red", 
                          annotation_text=f"Best: {objective:.4f}", annotation_position="right")
        
        if greedy_weights is not None:
            fig_conv.add_hline(y=obj_greedy, line_dash="dash", line_color="orange",
                              annotation_text=f"Greedy: {obj_greedy:.4f}", annotation_position="left")
        
        fig_conv.update_layout(
            title="Convergence: objective value over iterations",
            xaxis_title="Iteration",
            yaxis_title="Objective value",
            height=500
        )
        st.plotly_chart(fig_conv, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            portfolio_returns = (optimizer.returns * best_weights).sum(axis=1)
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=portfolio_returns,
                nbinsx=50,
                name='Returns',
                marker_color='steelblue',
                opacity=0.7
            ))
            fig_hist.add_vline(x=portfolio_returns.mean(), line_dash="dash", line_color="red",
                              annotation_text=f"Mean: {portfolio_returns.mean():.4f}")
            fig_hist.add_vline(x=portfolio_returns.median(), line_dash="dash", line_color="green",
                              annotation_text=f"Median: {portfolio_returns.median():.4f}")
            
            fig_hist.update_layout(
                title="Portfolio returns distribution",
                xaxis_title="Daily returns",
                yaxis_title="Frequency",
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            individual_sharpes = []
            for i in range(len(optimizer.tickers)):
                w = np.zeros(len(optimizer.tickers))
                w[i] = 1.0
                _, _, sharpe_ind, _ = optimizer.portfolio_performance(w)
                individual_sharpes.append(sharpe_ind)
            
            fig_sharpe = go.Figure()
            colors = ['green' if s > 0 else 'red' for s in individual_sharpes]
            fig_sharpe.add_trace(go.Bar(
                x=optimizer.tickers,
                y=individual_sharpes,
                marker_color=colors,
                text=[f"{s:.3f}" for s in individual_sharpes],
                textposition='auto'
            ))
            fig_sharpe.add_hline(y=sharpe, line_dash="dash", line_color="blue",
                                annotation_text=f"Portfolio: {sharpe:.4f}")
            
            fig_sharpe.update_layout(
                title="Individual asset Sharpe Ratios",
                xaxis_title="Assets",
                yaxis_title="Sharpe Ratio",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Covariance matrix
        fig_cov = go.Figure(data=go.Heatmap(
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
        
        fig_cov.update_layout(
            title="Asset covariance matrix",
            height=600,
            xaxis={'side': 'bottom'}
        )
        st.plotly_chart(fig_cov, use_container_width=True)
    
    with tab5:
        st.markdown("### 📊 Current portfolio data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Portfolio configuration:**")
            st.write(f"- **Number of stocks:** {len(optimizer.tickers)}")
            st.write(f"- **Date range:** {optimizer.start_date} to {optimizer.end_date}")
            st.write(f"- **Trading days:** {len(optimizer.returns)}")
            st.write(f"- **Risk-free rate:** {optimizer.risk_free_rate:.4f}")
        
        with col2:
            st.markdown("**Objective weights:**")
            st.write(f"- **Sharpe Ratio weight (γ):** {optimizer.sharpe_weight:.2f}")
            st.write(f"- **Return weight (ω):** {optimizer.return_weight:.2f}")
        
        st.markdown("---")
        st.markdown("### 💾 Available saved data files")
        
        saved_files = optimizer.list_saved_data_files()
        
        if saved_files:
            for i, file_info in enumerate(saved_files, 1):
                with st.expander(f"📁 {file_info['filename']}"):
                    st.write(f"**Tickers:** {', '.join(file_info['tickers'][:10])}" + 
                            (f" ... (+{len(file_info['tickers'])-10} more)" if len(file_info['tickers']) > 10 else ""))
                    st.write(f"**Total stocks:** {file_info['num_tickers']}")
                    st.write(f"**Period:** {file_info['start_date']} to {file_info['end_date']}")
                    st.write(f"**Fetched:** {file_info['fetch_timestamp']}")
        else:
            st.info("No saved data files found.")

elif st.session_state.data_loaded_from_file and st.session_state.optimizer is not None:
    # Show loaded data info before optimization
    optimizer = st.session_state.optimizer
    
    st.info(f"✅ **Data loaded from file!** {len(optimizer.tickers)} stocks available. Configure optimization parameters and click 'Run Optimization'.")
    
    st.markdown("### 📊 Loaded Portfolio Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of stocks", len(optimizer.tickers))
    with col2:
        st.metric("Trading days", len(optimizer.returns))
    with col3:
        st.metric("Date range", f"{optimizer.start_date} to {optimizer.end_date}")
    
    with st.expander("📋 View all stocks", expanded=False):
        st.write(", ".join(optimizer.tickers))

else:
    # Welcome screen
    st.info("👈 Configure your portfolio parameters in the sidebar and click '🚀 Run Optimization' to begin!")
    
    st.markdown("""
    ### 🎯 About this tool
    
    This application implements a **Biased-Randomized Portfolio Optimization** algorithm that combines:
    
    - **Multi-start heuristic approach** for exploring the solution space
    - **Biased randomization** using geometric or triangular distributions
    - **Local search optimization** for solution refinement
    - **Flexible objective function** combining Sharpe ratio and raw returns
    
    ### 📊 Features
    
    - ✅ Interactive stock selection with preset portfolios
    - ✅ **Load saved data files directly** without manual configuration
    - ✅ **Customizable objective function** (Sharpe ratio vs. raw returns)
    - ✅ Real-time mathematical equation display
    - ✅ Comprehensive visualization suite
    - ✅ Comparison with greedy heuristic baseline
    - ✅ Data caching for faster subsequent runs
    - ✅ Fully customizable optimization parameters
    
    ### 🚀 Getting started
    
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
