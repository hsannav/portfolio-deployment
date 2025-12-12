import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import pickle
from pathlib import Path
import copy

class BiasedRandomizedPortfolioOptimizer:
    
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02, data_dir='portfolio_data', 
                 sharpe_weight=1.0, return_weight=0.0, allocation_low=0.05, allocation_high=0.60):
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
        self.allocation_low = allocation_low
        self.allocation_high = allocation_high
        
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
        except Exception:
            return False
    
    def load_data_from_file(self, filepath):
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
        except Exception:
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
            raise ValueError("No data downloaded")
        
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
            raise ValueError("No valid price data")
        
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        
        if self.returns.empty or len(self.returns) < 2:
            raise ValueError("Insufficient return data")
        
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
    
    def equal_weight_portfolio(self):
        n_assets = len(self.tickers)
        weights = np.ones(n_assets) / n_assets
        return weights
    
    def minimum_variance_portfolio(self):
        n_assets = len(self.tickers)
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix.values * 252, weights))
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9})
            
            if result.success:
                weights = np.clip(result.x, 0, 1)
                weights = weights / np.sum(weights)
                return weights
            else:
                return initial_weights
        except:
            return initial_weights
    
    def risk_parity_portfolio(self):
        n_assets = len(self.tickers)
        cov_matrix = self.cov_matrix.values * 252
        
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib
        
        def risk_parity_objective(weights):
            risk_contrib = risk_contribution(weights)
            target = np.mean(risk_contrib)
            return np.sum((risk_contrib - target) ** 2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(risk_parity_objective, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9})
            
            if result.success:
                weights = np.clip(result.x, 0, 1)
                weights = weights / np.sum(weights)
                return weights
            else:
                return initial_weights
        except:
            return initial_weights
    
    def hierarchical_risk_parity(self):
        try:
            corr_matrix = self.returns.corr().values
            
            dist_matrix = np.sqrt((1 - corr_matrix) / 2)
            np.fill_diagonal(dist_matrix, 0)
            
            dist_condensed = squareform(dist_matrix, checks=False)
            linkage_matrix = linkage(dist_condensed, method='single')
            
            def get_quasi_diag(link):
                link = link.astype(int)
                sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
                num_items = link[-1, 3]
                
                while sort_ix.max() >= num_items:
                    sort_ix.index = list(range(0, sort_ix.shape[0] * 2, 2))
                    df0 = sort_ix[sort_ix >= num_items]
                    i = df0.index
                    j = df0.values - num_items
                    sort_ix[i] = link[j, 0]
                    df0 = pd.Series(link[j, 1], index=i + 1)
                    sort_ix = pd.concat([sort_ix, df0])
                    sort_ix = sort_ix.sort_index()
                    sort_ix.index = list(range(sort_ix.shape[0]))
                
                return sort_ix.tolist()
            
            sort_ix = get_quasi_diag(linkage_matrix)
            
            cov_matrix = self.cov_matrix.values * 252
            
            def get_cluster_var(cov, c_items):
                cov_slice = cov[np.ix_(c_items, c_items)]
                w = np.linalg.inv(cov_slice).dot(np.ones(len(c_items)))
                w /= w.sum()
                return np.dot(w, np.dot(cov_slice, w))
            
            def get_rec_bipart(cov, sort_ix):
                w = pd.Series(1.0, index=sort_ix)
                c_items = [sort_ix]
                
                while len(c_items) > 0:
                    c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
                    
                    for i in range(0, len(c_items), 2):
                        c_items0 = c_items[i]
                        c_items1 = c_items[i + 1]
                        
                        c_var0 = get_cluster_var(cov, c_items0)
                        c_var1 = get_cluster_var(cov, c_items1)
                        
                        alpha = 1 - c_var0 / (c_var0 + c_var1)
                        
                        w[c_items0] *= alpha
                        w[c_items1] *= 1 - alpha
                
                return w
            
            weights_series = get_rec_bipart(cov_matrix, sort_ix)
            weights = weights_series.values
            weights = weights / np.sum(weights)
            
            return weights
        except:
            return self.equal_weight_portfolio()
    
    def maximum_diversification_portfolio(self):
        n_assets = len(self.tickers)
        cov_matrix = self.cov_matrix.values * 252
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        def negative_diversification_ratio(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            weighted_vol = np.dot(weights, volatilities)
            if portfolio_vol == 0:
                return 1e10
            return -weighted_vol / portfolio_vol
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(negative_diversification_ratio, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9})
            
            if result.success:
                weights = np.clip(result.x, 0, 1)
                weights = weights / np.sum(weights)
                return weights
            else:
                return initial_weights
        except:
            return initial_weights
    
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
            allocation = remaining_weight * np.random.uniform(self.allocation_low, self.allocation_high)
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
                                     use_geometric=True, apply_local_search=True, local_search_prop=0.5,
                                     progress_callback=None):
        best_weights = None
        best_objective = -np.inf
        all_solutions = []
        
        for i in range(n_iterations):
            try:
                weights = self.biased_randomized_construction(beta, use_geometric)
                
                if apply_local_search and np.random.random() < local_search_prop:
                    weights = self.local_search_optimization(weights)                
                
                ret, std, sharpe, obj = self.portfolio_performance(weights)

                if apply_local_search and obj > 0.9 * best_objective:
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
                
                if progress_callback:
                    progress_callback((i + 1) / n_iterations, best_objective)

            except Exception:
                continue
        
        return best_weights, best_objective, all_solutions
    
class GeneticAlgorithmOptimizer(BiasedRandomizedPortfolioOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def selection_tournament(self, population, fitnesses, tournament_size=3):
        indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = indices[np.argmax(fitnesses[indices])]
        return population[best_idx]

    def selection_roulette_wheel(self, population, fitnesses):
        min_fit = np.min(fitnesses)
        adjusted_fitnesses = fitnesses - min_fit + 1e-6 
        probs = adjusted_fitnesses / np.sum(adjusted_fitnesses)
        idx = np.random.choice(len(population), p=probs)
        return population[idx]

    def crossover_arithmetic(self, parent1, parent2, alpha=0.5):
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def crossover_single_point(self, parent1, parent2):
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def crossover_uniform(self, parent1, parent2):
        mask = np.random.rand(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def mutate_gaussian(self, individual, mutation_rate=0.1, strength=0.1):
        if np.random.random() < mutation_rate:
            noise = np.random.normal(0, strength, size=len(individual))
            individual += noise
            individual = np.clip(individual, 0, 1)
            if np.sum(individual) > 0:
                 individual /= np.sum(individual)
        return individual

    def mutate_swap(self, individual, mutation_rate=0.1):
        if np.random.random() < mutation_rate:
            idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual
    
    def create_individual(self):
        weights = np.random.random(len(self.tickers))
        weights /= np.sum(weights)
        return weights

    def run_benchmark(self, configurations, n_generations=50, population_size=100, progress_callback=None):
        results = {}
        total_steps = len(configurations) * n_generations
        current_step = 0

        for config in configurations:
            population = [self.create_individual() for _ in range(population_size)]
            best_objective = -np.inf
            best_weights = None
            history = []

            for gen in range(n_generations):
                fitnesses = []
                for ind in population:
                    ind = np.clip(ind, 0, 1)
                    if np.sum(ind) > 0:
                        ind /= np.sum(ind)
                    else:
                        ind = self.equal_weight_portfolio()
                    
                    _, _, _, obj = self.portfolio_performance(ind)
                    fitnesses.append(obj)
                
                fitnesses = np.array(fitnesses)
                
                gen_best_idx = np.argmax(fitnesses)
                current_best_obj = fitnesses[gen_best_idx]
                
                if current_best_obj > best_objective:
                    best_objective = current_best_obj
                    best_weights = population[gen_best_idx].copy()
                
                history.append(best_objective)

                new_population = []
                elitism_count = config.get('elitism', 2)
                sorted_indices = np.argsort(fitnesses)[::-1]
                for i in range(elitism_count):
                    new_population.append(population[sorted_indices[i]].copy())

                while len(new_population) < population_size:
                    sel_method = config.get('selection', 'Tournament')
                    if sel_method == 'Roulette':
                        parent1 = self.selection_roulette_wheel(population, fitnesses)
                        parent2 = self.selection_roulette_wheel(population, fitnesses)
                    else:
                        parent1 = self.selection_tournament(population, fitnesses)
                        parent2 = self.selection_tournament(population, fitnesses)
                    
                    if np.random.random() < config.get('crossover_rate', 0.8):
                        cx_method = config.get('crossover', 'Arithmetic')
                        if cx_method == 'Single Point':
                            c1, c2 = self.crossover_single_point(parent1, parent2)
                        elif cx_method == 'Uniform':
                            c1, c2 = self.crossover_uniform(parent1, parent2)
                        else:
                            c1, c2 = self.crossover_arithmetic(parent1, parent2)
                    else:
                        c1, c2 = parent1.copy(), parent2.copy()

                    mut_rate = config.get('mutation_rate', 0.1)
                    mut_method = config.get('mutation', 'Gaussian')
                    
                    if mut_method == 'Swap':
                        c1 = self.mutate_swap(c1, mut_rate)
                        c2 = self.mutate_swap(c2, mut_rate)
                    else:
                        c1 = self.mutate_gaussian(c1, mut_rate)
                        c2 = self.mutate_gaussian(c2, mut_rate)

                    new_population.append(c1)
                    if len(new_population) < population_size:
                        new_population.append(c2)
                
                population = new_population
                
                current_step += 1
                if progress_callback:
                    progress_callback(current_step / total_steps, f"Running {config['name']} (Gen {gen+1})")
            
            ret, std, sharpe, _ = self.portfolio_performance(best_weights)
            results[config['name']] = {
                'weights': best_weights,
                'objective': best_objective,
                'history': history,
                'metrics': {'Return': ret, 'Risk': std, 'Sharpe': sharpe}
            }
            
        return results