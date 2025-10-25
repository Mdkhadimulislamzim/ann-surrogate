import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import random
from datetime import datetime
from typing import List, Tuple, Dict, Callable
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append('.')
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

from afn.afn_core import AFNCore
from afn.comparison_algorithms import GA, PSO, ACO
from data.sample import load_bbob_function

# Import Hansen and Bajer optimizers
try:
    import cma
    from skopt import gp_minimize
    from skopt.space import Real
    HANSEN_BAJER_AVAILABLE = True
except ImportError:
    HANSEN_BAJER_AVAILABLE = False
    print("Warning: cma or skopt not available. Hansen/Bajer optimizers will be skipped.")

def align_curve_to_budget(curve, budget):
    """Align curve to exact budget length"""
    if len(curve) >= budget:
        return curve[:budget]
    else:
        # Extend with last value
        last_val = curve[-1] if curve else 0.0
        return curve + [last_val] * (budget - len(curve))

def bajer_gp_minimize(func, lo_vec, hi_vec, dim, budget=100, seed=0):
    """
    Bajer-style GP surrogate baseline (GP + EI) ≈ DTS/S-CMA-ES spirit:
    we run a GP-EI loop (scikit-optimize) within the same evaluation budget.
    Reference: Bajer et al., 2019.  (GP uncertainty, EI-driven selection)
    """
    # scikit-optimize expects scalar bounds per dim
    lo = float(np.min(lo_vec))
    hi = float(np.max(hi_vec))
    space = [Real(lo, hi)] * dim

    # Run gp_minimize; it internally does initial points + EI acquisitions
    res = gp_minimize(
        func,
        space,
        n_calls=budget,
        n_initial_points=min(10, max(5, dim)),  # small warmup
        acq_func="EI",
        noise=0.0,
        random_state=seed,
    )

    # Build best-so-far curve from res.func_vals
    best = np.minimum.accumulate(np.array(res.func_vals, dtype=float))
    return align_curve_to_budget(best.tolist(), budget)

def hansen_cmaes(func, lo_vec, hi_vec, dim, budget=100, seed=0):
    """
    Hansen-style surrogate-assisted CMA-ES baseline (approx):
    use pycma's CMA-ES with bound constraints; pycma contains
    surrogate hooks (lq-CMA-ES family). Here we approximate with
    standard CMA-ES under identical budget—representative of the
    Hansen portfolio (CMA + surrogate gating).
    Reference: Hansen, 2019 (global linear/quad surrogate; rank-corr gating)
    """
    # Center start in the box; sigma ~ box size / 3
    lo = np.asarray(lo_vec, float)
    hi = np.asarray(hi_vec, float)
    x0 = (lo + hi) / 2.0
    sigma0 = float(np.mean((hi - lo) / 3.0))

    opts = {
        "bounds": [lo.tolist(), hi.tolist()],
        "popsize": 20,
        "maxfevals": budget,
        "seed": seed,
        "verb_disp": 0,
    }
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    best_curve = []
    fevals = 0
    while not es.stop():
        X = es.ask()
        y = [func(np.array(x, dtype=float)) for x in X]
        es.tell(X, y)
        fevals += len(y)
        # extend curve by the number of evaluations this gen consumed
        best = es.result.fbest
        best_curve.extend([best] * len(y))
        if fevals >= budget:
            break

    return align_curve_to_budget(best_curve, budget)

class AFNHansenBajerComparison:
    """Comparison between AFN, Hansen CMA-ES, and Bajer GP-EI"""
    
    def __init__(self, test_functions: List[int], dimensions: List[int], 
                 n_runs: int, max_evaluations: int, save_dir: str):
        self.test_functions = test_functions
        self.dimensions = dimensions
        self.n_runs = n_runs
        self.max_evaluations = max_evaluations
        self.save_dir = save_dir
        self.output_dir = os.path.join(save_dir, f"afn_hansen_bajer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Algorithms to compare
        self.algorithms = ["AFN", "Hansen", "Bajer"]
        if not HANSEN_BAJER_AVAILABLE:
            self.algorithms = ["AFN"]
            print("Warning: Only AFN will be tested due to missing dependencies")
        
        # Results storage
        self.results = {}
        self.metrics = {}
        
        # Save configuration
        config = {
            "test_functions": test_functions,
            "dimensions": dimensions,
            "n_runs": n_runs,
            "max_evaluations": max_evaluations,
            "algorithms": self.algorithms,
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def run_single_test(self, func_id: int, dimension: int, 
                       algorithm_name: str, run_idx: int, seed: int = None) -> Dict:
        """Run a single test case"""
        if seed is None:
            seed = run_idx * 1000 + func_id * 100 + dimension * 10
        
        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)
        
        # Load function
        # Get COCO/BBOB function and bounds
        problem, info = load_bbob_function(func_id, dimension, instance=1)
        objective_function = problem
        bounds = [(problem.lower_bounds[i], problem.upper_bounds[i]) for i in range(dimension)]
        optimum = 0.0  # Most BBOB functions have optimum at 0.0
        
        start_time = time.time()
        
        try:
            if algorithm_name == "AFN":
                algorithm = AFNCore(
                    input_dim=dimension,
                    bounds=bounds,
                    uncertainty_threshold=0.03,
                    batch_size=8,
                    max_evaluations=self.max_evaluations,
                    convergence_threshold=1e-6,
                    convergence_window=10
                )
                result = algorithm.optimize(objective_function, verbose=False)
                
                return {
                    'best_x': result['best_x'],
                    'best_y': result['best_y'],
                    'history': result['y_history'],
                    'evaluations': result['evaluation_count'],
                    'time': time.time() - start_time,
                    'converged': result['converged'],
                    'optimum': optimum
                }
                
            elif algorithm_name == "Hansen" and HANSEN_BAJER_AVAILABLE:
                lo_vec = [b[0] for b in bounds]
                hi_vec = [b[1] for b in bounds]
                history = hansen_cmaes(objective_function, lo_vec, hi_vec, dimension, 
                                     self.max_evaluations, seed)
                
                best_y = history[-1] if history else float('inf')
                
                return {
                    'best_x': None,  # CMA-ES doesn't return best x easily
                    'best_y': best_y,
                    'history': history,
                    'evaluations': len(history),
                    'time': time.time() - start_time,
                    'converged': False,  # CMA-ES convergence is complex
                    'optimum': optimum
                }
                
            elif algorithm_name == "Bajer" and HANSEN_BAJER_AVAILABLE:
                lo_vec = [b[0] for b in bounds]
                hi_vec = [b[1] for b in bounds]
                history = bajer_gp_minimize(objective_function, lo_vec, hi_vec, dimension,
                                          self.max_evaluations, seed)
                
                best_y = history[-1] if history else float('inf')
                
                return {
                    'best_x': None,  # GP minimize doesn't return best x easily
                    'best_y': best_y,
                    'history': history,
                    'evaluations': len(history),
                    'time': time.time() - start_time,
                    'converged': False,  # GP minimize convergence is complex
                    'optimum': optimum
                }
                
        except Exception as e:
            print(f"Error in {algorithm_name} (f{func_id}, d{dimension}, run{run_idx}): {e}")
            return {
                'best_x': None,
                'best_y': float('inf'),
                'history': [],
                'evaluations': 0,
                'time': time.time() - start_time,
                'converged': False,
                'optimum': optimum
            }

    def compute_metrics(self, results: List[Dict], algorithm_name: str = "AFN") -> Dict:
        """Compute performance metrics from results - using same logic as GA/PSO/ACO comparison"""
        if not results:
            return {}
        
        best_values = [r['best_y'] for r in results if r['best_y'] != float('inf')]
        exec_times = [r['time'] for r in results if r['time'] > 0]
        eval_counts = [r.get('evaluations', self.max_evaluations) for r in results]
        
        # Histories for baseline (avoid boolean ops on numpy arrays) - same logic as GA/PSO/ACO
        histories = []
        for r in results:
            h = []
            if isinstance(r, dict):
                if 'history' in r and r['history'] is not None:
                    h = r['history']
                elif 'y_history' in r and r['y_history'] is not None:
                    h = r['y_history']
            # Normalize to list
            if isinstance(h, np.ndarray):
                h = h.tolist()
            elif not isinstance(h, list):
                h = list(h) if h is not None else []
            histories.append(h)
        
        if not best_values:
            return {
                'mean_best': float('inf'),
                'std_best': float('inf'),
                'mean_time': 0.0,
                'convergence_speed': 0.0,
                'optimization_accuracy': 0.0,
                'resource_utilization': 0.0,
                'exploitation_balance': 0.0,
                'robustness': 0.0,
                'n_runs': len(results)
            }
        
        # Basic statistics
        mean_best = np.mean(best_values)
        std_best = np.std(best_values)
        mean_time = np.mean(exec_times) if exec_times else 0.0
        
        # Optimization Accuracy (same logic as GA/PSO/ACO)
        optimum = results[0]['optimum'] if results else 0.0
        opt_acc = []
        for i, bv in enumerate(best_values):
            start_val = float(histories[i][0]) if i < len(histories) and histories[i] else float(bv)
            gap = max(1e-12, start_val - optimum)
            acc = 100.0 * (1.0 - (float(bv) - optimum) / gap)
            opt_acc.append(float(np.clip(acc, 0.0, 100.0)))
        
        # Convergence speed as percentage of evaluations to reach 95% of improvement (same logic as GA/PSO/ACO)
        conv_speeds = []
        for h in histories:
            if h and len(h) > 1:
                initial_val = h[0]
                final_val = min(h)
                total_improvement = initial_val - final_val
                if total_improvement > 1e-12:
                    # Target: 95% of improvement achieved
                    target_val = initial_val - 0.95 * total_improvement
                    for i, v in enumerate(h):
                        if v <= target_val:
                            # Convert to percentage of max evaluations, cap at 100%
                            conv_percentage = min(100.0, ((i + 1) / max(1, self.max_evaluations)) * 100.0)
                            conv_speeds.append(conv_percentage)
                            break
                    else:
                        # If never reached target, use 100% (used all evaluations)
                        conv_speeds.append(100.0)
                else:
                    # No significant improvement, convergence at first evaluation
                    conv_speeds.append(100.0 / max(1, self.max_evaluations))
            else:
                # No history or single value, assume immediate convergence
                conv_speeds.append(100.0 / max(1, self.max_evaluations))
        
        # Resource utilization (same logic as GA/PSO/ACO)
        resource_utilization = float(np.mean([min(100.0, (c / max(1, self.max_evaluations)) * 100.0) for c in eval_counts]))
        
        # Exploitation balance using algorithm-specific E/(E+R) formula (0-100%)
        exploitation_balance = 50.0  # Default to 50% if no valid data
        if histories:
            balances = []
            for history in histories:
                if len(history) > 10:
                    E, R = self._calculate_algorithm_contributions(history, algorithm_name)
                    total = E + R
                    if total > 0:
                        balance = E / total
                        # Ensure balance is in reasonable range
                        balance = max(0.0, min(1.0, balance))
                        balances.append(balance)
            
            if balances:
                exploitation_balance = np.mean(balances) * 100
        
        # Robustness (same logic as GA/PSO/ACO)
        robustness = float(max(0.0, (1.0 - (np.std(best_values) / max(1e-12, np.mean(best_values)))) * 100.0)) if len(best_values) > 1 else 100.0
        
        # Success rate calculation removed - redundant with optimization accuracy
        
        return {
            'mean_best': mean_best,
            'std_best': std_best,
            'mean_time': mean_time,
            'convergence_speed': float(np.mean(conv_speeds)) if conv_speeds else 0.0,
            'optimization_accuracy': float(np.mean(opt_acc)) if opt_acc else 0.0,
            'resource_utilization': resource_utilization,
            'exploitation_balance': exploitation_balance,
            'robustness': robustness,
            'n_runs': len(results)
        }
    
    def _calculate_algorithm_contributions(self, history: List[float], algorithm_name: str) -> tuple:
        """Calculate algorithm-specific exploitation (E) and exploration (R) contributions"""
        
        if algorithm_name == "AFN":
            return self._calculate_afn_contributions(history)
        elif algorithm_name == "Hansen":
            return self._calculate_hansen_contributions(history)
        elif algorithm_name == "Bajer":
            return self._calculate_bajer_contributions(history)
        else:
            # Fallback to generic calculation
            return self._calculate_generic_contributions(history)
    
    def _calculate_afn_contributions(self, history: List[float]) -> tuple:
        """AFN-specific: E = surrogate model refinement, R = uncertainty-based sampling"""
        if len(history) < 5:
            return 0.0, 0.0
        
        E = 0.0  # Exploitation: local refinement efforts
        R = 0.0  # Exploration: uncertainty-driven sampling
        
        # Calculate variance and trend to determine exploitation vs exploration
        improvements = [history[i-1] - history[i] for i in range(1, len(history))]
        mean_improvement = np.mean(improvements) if improvements else 0.0
        improvement_variance = np.var(improvements) if len(improvements) > 1 else 0.0
        
        for i, improvement in enumerate(improvements):
            progress_factor = (i + 1) / len(improvements)
            
            if improvement > 0:  # Improving step
                # AFN exploitation: local refinement with adaptive weighting
                # More exploitation early, less as optimization progresses
                exploitation_weight = 0.6 + 0.4 * (1.0 - progress_factor)
                E += improvement * exploitation_weight
            else:
                # AFN exploration: uncertainty-driven sampling
                # More exploration needed when variance is high
                exploration_weight = 0.4 + 0.6 * min(1.0, improvement_variance / max(1.0, mean_improvement))
                R += abs(improvement) * exploration_weight
        
        # Simplified approach - return raw values
        
        return E, R
    
    def _calculate_hansen_contributions(self, history: List[float]) -> tuple:
        """Hansen CMA-ES-specific: E = mean vector updates, R = covariance matrix adaptation"""
        if len(history) < 5:
            return 0.0, 0.0
        
        E = 0.0  # Exploitation: mean vector updates (guidance towards better solutions)
        R = 0.0  # Exploration: covariance matrix adaptation (search distribution reshaping)
        
        improvements = [history[i-1] - history[i] for i in range(1, len(history))]
        mean_improvement = np.mean(improvements) if improvements else 0.0
        
        for i, improvement in enumerate(improvements):
            progress_factor = (i + 1) / len(improvements)
            
            if improvement > 0:  # Improving step
                # CMA-ES exploitation: mean vector guides search towards better regions
                # Strong exploitation, but adapts based on progress
                exploitation_weight = 0.7 + 0.3 * (1.0 - progress_factor)
                E += improvement * exploitation_weight
            else:
                # CMA-ES exploration: covariance matrix adaptation enables new region discovery
                # Moderate exploration, increases when adaptation is needed
                exploration_weight = 0.5 + 0.3 * progress_factor
                R += abs(improvement) * exploration_weight
        
        # Normalize to prevent extreme values
        total_activity = E + R
        if total_activity > 0:
            E = E / total_activity * len(improvements) * mean_improvement
            R = R / total_activity * len(improvements) * mean_improvement
        
        return E, R
    
    def _calculate_bajer_contributions(self, history: List[float]) -> tuple:
        """Bajer GP-EI-specific: E = expected improvement exploitation, R = GP uncertainty exploration"""
        if len(history) < 5:
            return 0.0, 0.0
        
        E = 0.0  # Exploitation: expected improvement-driven selection
        R = 0.0  # Exploration: GP uncertainty-based sampling
        
        improvements = [history[i-1] - history[i] for i in range(1, len(history))]
        mean_improvement = np.mean(improvements) if improvements else 0.0
        
        for i, improvement in enumerate(improvements):
            progress_factor = (i + 1) / len(improvements)
            
            if improvement > 0:  # Improving step
                # GP-EI exploitation: expected improvement balances exploitation
                # Moderate exploitation, adapts based on GP confidence
                exploitation_weight = 0.5 + 0.2 * progress_factor
                E += improvement * exploitation_weight
            else:
                # GP-EI exploration: GP uncertainty drives exploration
                # High exploration, especially when GP is uncertain
                exploration_weight = 0.6 + 0.3 * (1.0 - progress_factor)
                R += abs(improvement) * exploration_weight
        
        # Normalize to prevent extreme values
        total_activity = E + R
        if total_activity > 0:
            E = E / total_activity * len(improvements) * mean_improvement
            R = R / total_activity * len(improvements) * mean_improvement
        
        return E, R
    
    def _calculate_generic_contributions(self, history: List[float]) -> tuple:
        """Generic fallback calculation"""
        if len(history) < 5:
            return 0.0, 0.0
        
        E = 0.0
        R = 0.0
        
        for i in range(1, len(history)):
            improvement = history[i-1] - history[i]
            
            if improvement > 0:
                E += improvement * 0.6  # Generic exploitation weight
            else:
                R += abs(improvement) * 0.6  # Generic exploration weight
        
        return E, R

    def run_full_comparison(self, verbose: bool = True, save_plots: bool = True):
        """Run full comparison across all functions, dimensions, and algorithms"""
        total_tests = len(self.test_functions) * len(self.dimensions) * len(self.algorithms) * self.n_runs
        current_test = 0
        
        print(f"Starting comparison: {total_tests} total tests")
        print(f"Functions: {self.test_functions}")
        print(f"Dimensions: {self.dimensions}")
        print(f"Algorithms: {self.algorithms}")
        print(f"Runs per test: {self.n_runs}")
        print("-" * 60)
        
        for func_id in self.test_functions:
            for dimension in self.dimensions:
                test_key = f"f{func_id}_d{dimension}"
                self.results[test_key] = {}
                self.metrics[test_key] = {}
                
                for algorithm_name in self.algorithms:
                    if verbose:
                        print(f"\nTesting {algorithm_name} on Function {func_id}, Dimension {dimension}")
                    
                    algorithm_results = []
                    
                    for run_idx in range(self.n_runs):
                        current_test += 1
                        if verbose:
                            print(f"  Run {run_idx + 1}/{self.n_runs} ({current_test}/{total_tests})")
                        
                        result = self.run_single_test(func_id, dimension, algorithm_name, run_idx)
                        algorithm_results.append(result)
                        
                        if verbose and result['best_y'] != float('inf'):
                            print(f"    Best: {result['best_y']:.6f}, Time: {result['time']:.2f}s")
                    
                    self.results[test_key][algorithm_name] = algorithm_results
                    
                    # Compute metrics
                    metrics = self.compute_metrics(algorithm_results, algorithm_name)
                    self.metrics[test_key][algorithm_name] = metrics
                    
                    if verbose:
                        print(f"  {algorithm_name} Results:")
                        print(f"    Mean Best: {metrics['mean_best']:.6f} ± {metrics['std_best']:.6f}")
                        print(f"    Optimization Accuracy: {metrics['optimization_accuracy']:.1f}%")
                        print(f"    Robustness: {metrics['robustness']:.1f}%")
        
        # Save results
        self.save_results()
        
        # Create plots
        if save_plots:
            self.create_comparison_plots()
        
        return self.results, self.metrics

    def save_results(self):
        """Save results to JSON files"""
        # Save detailed results
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save metrics summary
        with open(os.path.join(self.output_dir, "metrics_summary.json"), "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

    def create_comparison_plots(self):
        """Create comparison plots"""
        if not self.metrics:
            return
        
        # Plot 1: Optimization Accuracy
        self._plot_metric_comparison('optimization_accuracy', 'Optimization Accuracy', 
                                   'Optimization Accuracy (%)', 'optimization_accuracy.png')
        
        # Plot 2: Convergence Speed
        self._plot_metric_comparison('convergence_speed', 'Convergence Speed', 
                                   'Convergence Speed (%)', 'convergence_speed.png')
        
        # Plot 3: Resource Utilization
        self._plot_metric_comparison('resource_utilization', 'Resource Utilization', 
                                   'Resource Utilization (%)', 'resource_utilization.png')
        
        # Plot 4: Exploitation Balance
        self._plot_metric_comparison('exploitation_balance', 'Exploitation Balance', 
                                   'Exploitation Balance (%)', 'exploitation_balance.png')
        
        # Plot 5: Robustness
        self._plot_metric_comparison('robustness', 'Robustness', 
                                   'Robustness (%)', 'robustness.png')
        
        # Success rate removed - redundant with optimization accuracy

    def _plot_metric_comparison(self, metric_name: str, title: str, ylabel: str, filename: str):
        """Create a comparison bar chart for a specific metric"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Collect data
        test_cases = []
        algorithm_data = {alg: [] for alg in self.algorithms}
        
        for test_key in sorted(self.metrics.keys()):
            # Extract function and dimension info
            if '_d' in test_key:
                func_dim = test_key.replace('f', 'F').replace('_d', 'D')
                test_cases.append(func_dim)
                
                for algorithm_name in self.algorithms:
                    if algorithm_name in self.metrics[test_key]:
                        value = self.metrics[test_key][algorithm_name].get(metric_name, 0)
                        algorithm_data[algorithm_name].append(value)
                    else:
                        algorithm_data[algorithm_name].append(0)
        
        # Create bar chart
        x = np.arange(len(test_cases))
        width = 0.25
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for AFN, Hansen, Bajer
        
        for i, (algorithm_name, values) in enumerate(algorithm_data.items()):
            if values:
                bars = ax.bar(x + i * width, values, width, 
                             label=algorithm_name, 
                             color=colors[i % len(colors)],
                             alpha=0.8,
                             edgecolor='black',
                             linewidth=0.5)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{height:.1f}' if height < 100 else f'{height:.0f}',
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Test Cases (Function + Dimension)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title} Comparison: AFN vs Hansen vs Bajer', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(test_cases, rotation=45, ha='right')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Set y-axis to start from 0 for better comparison
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AFN vs Hansen vs Bajer Comparison')
    parser.add_argument('--functions', type=str, default='1,8,3', 
                       help='Comma-separated list of BBOB function IDs (default: 1,8,3)')
    parser.add_argument('--dimensions', type=str, default='2,5,10', 
                       help='Comma-separated list of dimensions (default: 2,5,10)')
    parser.add_argument('--n_runs', type=int, default=5, 
                       help='Number of runs per test (default: 5)')
    parser.add_argument('--max_evals', type=int, default=100, 
                       help='Maximum evaluations per run (default: 100)')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Output directory for results (default: results)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test with minimal settings')
    
    return parser.parse_args()

def parse_list_arg(arg_str: str) -> List[int]:
    """Parse comma-separated list argument"""
    try:
        return [int(x.strip()) for x in arg_str.split(',') if x.strip()]
    except ValueError:
        return []

def print_banner():
    """Print comparison banner"""
    print("=" * 80)
    print("AFN vs Hansen vs Bajer Optimization Comparison")
    print("Using Ensemble-based AFN with RandomForestRegressor")
    print("=" * 80)

def main():
    """Main function"""
    args = parse_arguments()
    
    # Handle quick mode
    if args.quick:
        print("Quick mode enabled - using minimal settings")
        test_functions = [1]  # Only Sphere
        dimensions = [2]      # Only 2D
        args.n_runs = 2       # Only 2 runs
        args.max_evals = 50   # Only 50 evaluations
        args.verbose = True   # Always verbose in quick mode
    else:
        # Parse function and dimension lists
        test_functions = parse_list_arg(args.functions)
        dimensions = parse_list_arg(args.dimensions)
    
    # Print banner and configuration
    print_banner()
    print(f"\nConfiguration:")
    print(f"  Test functions: {test_functions}")
    print(f"  Dimensions: {dimensions}")
    print(f"  Runs per test: {args.n_runs}")
    print(f"  Max evaluations: {args.max_evals}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Verbose: {args.verbose}")
    
    # Calculate total runs
    algorithms = ["AFN", "Hansen", "Bajer"] if HANSEN_BAJER_AVAILABLE else ["AFN"]
    total_runs = len(test_functions) * len(dimensions) * len(algorithms) * args.n_runs
    print(f"  Total runs: {total_runs}")
    print()
    
    # Validate inputs
    if len(test_functions) == 0:
        print("Error: No test functions specified")
        return 1
    
    if len(dimensions) == 0:
        print("Error: No dimensions specified")
        return 1
    
    if args.n_runs < 1:
        print("Error: Number of runs must be at least 1")
        return 1
    
    if args.max_evals < 1:
        print("Error: Max evaluations must be at least 1")
        return 1
    
    # Check if output directory exists or can be created
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Cannot create output directory '{args.output_dir}': {e}")
        return 1
    
    try:
        # Create comparison instance
        comparison = AFNHansenBajerComparison(
            test_functions=test_functions,
            dimensions=dimensions,
            n_runs=args.n_runs,
            max_evaluations=args.max_evals,
            save_dir=args.output_dir
        )
        
        # Run the comparison
        print("Starting comparison...")
        results, metrics = comparison.run_full_comparison(verbose=args.verbose, save_plots=True)
        
        print("\nComparison completed successfully!")
        print(f"Results saved to: {comparison.output_dir}")
        
        # Print final summary
        print("\nFinal Summary:")
        print("-" * 60)
        
        for alg_name in algorithms:
            if any(alg_name in test_metrics for test_metrics in metrics.values()):
                # Calculate overall statistics
                all_accuracies = []
                all_robustness = []
                for test_data in metrics.values():
                    if alg_name in test_data:
                        all_accuracies.append(test_data[alg_name]['optimization_accuracy'])
                        all_robustness.append(test_data[alg_name]['robustness'])
                
                overall_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
                overall_robustness = sum(all_robustness) / len(all_robustness) if all_robustness else 0
                
                print(f"{alg_name}:")
                print(f"  Overall optimization accuracy: {overall_accuracy:.1f}%")
                print(f"  Overall robustness: {overall_robustness:.1f}%")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nComparison interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during comparison: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
