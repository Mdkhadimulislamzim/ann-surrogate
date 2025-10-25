# 🚀 AFN (Adaptive Fidelity Nexus) Framework

**Paper Implementation**: "Artificial neural networks as surrogate models in optimization"

This repository contains an **ensemble-based implementation** of the AFN algorithm for surrogate-based optimization using COCO/BBOB benchmark functions, with comparisons against GA, PSO, and ACO algorithms.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Algorithm Details](#algorithm-details)
- [Results & Metrics](#results--metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

The **Adaptive Fidelity Nexus (AFN)** is a novel methodology that adaptively improves an **ensemble regressor** surrogate model by intelligently querying expensive objective functions only where uncertainty is high or where potential optima are found. This leads to more efficient and robust optimization compared to static surrogate models.

### Key Features

✅ **Ensemble Regressor Surrogate** using Random Forest Regressors  
✅ **Real COCO/BBOB Benchmark Functions** (24 functions available)  
✅ **Paper-Accurate Implementation** (exact specifications)  
✅ **Uncertainty Quantification** via ensemble variance  
✅ **GA, PSO, ACO Comparison** algorithms  
✅ **Lightweight Dependencies** (minimal scikit-learn requirements)  
✅ **Command-Line Interface** for easy usage  
✅ **Comprehensive Visualization** and metrics  
✅ **No Heavy ML Dependencies** (PyTorch-free option)

## 🔧 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Virtual Environment Setup (Recommended)

**⚠️ Important**: It's highly recommended to use a virtual environment to avoid dependency conflicts with other Python projects.

#### For Windows (PowerShell/Command Prompt)

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# For PowerShell:
.\.venv\Scripts\Activate.ps1
# OR for Command Prompt:
.venv\Scripts\activate.bat

# Verify activation (you should see (.venv) in your prompt)
python --version
pip --version
```

#### For macOS/Linux (Terminal)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify activation (you should see (.venv) in your prompt)
python --version
pip --version
```

#### Deactivating Virtual Environment

When you're done working with the project:

```bash
# Deactivate virtual environment (works on all platforms)
deactivate
```

#### Troubleshooting Virtual Environment Issues

**Windows PowerShell Execution Policy Error:**
```powershell
# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activating again:
.\.venv\Scripts\Activate.ps1
```

**Alternative Windows Activation:**
```cmd
# Use Command Prompt instead of PowerShell
.venv\Scripts\activate.bat
```

**Python3 vs Python Command:**
- On some systems, use `python3` instead of `python`
- Check with: `python --version` or `python3 --version`

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```
scikit-learn>=1.1.0
numpy>=1.21.0
matplotlib>=3.5.0
coco-experiment
cocopp
```

## 🚀 Quick Start

### 1. Set Up Virtual Environment (if not already done)

```bash
# Create and activate virtual environment (see Virtual Environment Setup section above)
python -m venv .venv

# Windows:
.\.venv\Scripts\Activate.ps1  # PowerShell
# OR
.venv\Scripts\activate.bat    # Command Prompt

# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Everything Works

First, verify that all components are working correctly:

```bash
python -c "from afn.afn_core import AFNCore; print('✅ AFN ready!')"
```

### 3. Run the Complete Comparison

Execute the full comparison of AFN vs GA vs PSO vs ACO:

```bash
# Quick test (recommended first)
python run_afn_ga_pso_aco_comparison.py --functions 1 --dimensions 2 --n_runs 3 --max_evals 50

# Default comparison (Sphere, Rosenbrock, Rastrigin; 2D, 5D)
python run_afn_ga_pso_aco_comparison.py

# Full comparison with all functions and dimensions
python run_afn_ga_pso_aco_comparison.py --functions 1,2,3,4,5 --dimensions 2,5,10 --n_runs 20 --max_evals 200 --verbose
```

### 4. Run Hansen & Bajer Comparison

Compare AFN with state-of-the-art optimizers (Hansen CMA-ES and Bajer GP-EI):

```bash
# Install additional dependencies first
pip install -r requirements_hansen_bajer.txt

# Quick test
python run_afn_hansen_bajer_comparison.py --quick

# Default comparison (Sphere, Rosenbrock, Rastrigin; 2D, 5D, 10D)
python run_afn_hansen_bajer_comparison.py

# Full comparison
python run_afn_hansen_bajer_comparison.py --functions 1,8,3 --dimensions 2,5,10 --n_runs 10 --max_evals 200 --verbose
```

### 5. Test Individual Algorithms

Test individual algorithms on simple functions:

```bash
# Test GA, PSO, ACO algorithms
python -c "from afn.comparison_algorithms import test_algorithms; test_algorithms()"

# Test simple functions
python -c "from afn.simple_test_functions import test_all_functions; test_all_functions()"
```

## 💡 Usage Examples

### Example 1: Basic AFN Optimization

```python
from afn.afn_core import AFNCore
import numpy as np

# Define objective function
def sphere(x):
    return np.sum(x**2)

# Set up AFN
bounds = [(-5, 5), (-5, 5)]  # 2D problem
afn = AFNCore(
    input_dim=2,
    bounds=bounds,
    max_evaluations=100,
    n_models=5
)

# Optimize
result = afn.optimize(sphere, verbose=True)
print(f"Best solution: {result['best_x']}")
print(f"Best value: {result['best_y']}")
```

### Example 2: Algorithm Comparison

```python
from afn.afn_core import AFNCore
from afn.comparison_algorithms import GA, PSO, ACO
import numpy as np

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

bounds = [(-2.048, 2.048), (-2.048, 2.048)]

# Test all algorithms
algorithms = {
    'AFN': AFNCore(input_dim=2, bounds=bounds, max_evaluations=100),
    'GA': GA(bounds=bounds, max_generations=100),
    'PSO': PSO(bounds=bounds, max_iterations=100),
    'ACO': ACO(bounds=bounds, max_iterations=100)
}

for name, alg in algorithms.items():
    result = alg.optimize(rosenbrock, verbose=False)
    print(f"{name}: {result['best_y']:.6f}")
```

### Example 3: BBOB Function Testing

```python
from data.sample import load_bbob_function
from afn.afn_core import AFNCore

# Load BBOB function
problem, info = load_bbob_function(func_id=1, dimension=2, instance=1)
print(info)

# Set up AFN with BBOB bounds
bounds = [(problem.lower_bounds[i], problem.upper_bounds[i]) for i in range(2)]
afn = AFNCore(input_dim=2, bounds=bounds, max_evaluations=100)

# Optimize
result = afn.optimize(problem, verbose=True)
```

## 📁 Project Structure

```
final-repo/
├── afn/                                    # AFN framework core
│   ├── __init__.py                         # Package initialization
│   ├── afn_core.py                         # Main AFN implementation
│   ├── comparison_algorithms.py            # GA, PSO, ACO implementations
│   └── simple_test_functions.py            # Standard test functions (no COCO)
├── data/                                   # Data utilities
│   ├── __init__.py                         # Package initialization
│   └── sample.py                           # BBOB data sampling functions
├── models/                                 # Model implementations
│   └── __init__.py                         # Package initialization
├── run_afn_ga_pso_aco_comparison.py        # Main comparison runner
├── run_afn_hansen_bajer_comparison.py      # Hansen & Bajer comparison
├── test_afn.py                             # Test suite
├── example_usage.py                        # Usage examples
├── requirements.txt                        # Python dependencies
├── requirements_hansen_bajer.txt           # Additional dependencies for Hansen/Bajer
└── README.md                              # This file
```

## 🧠 Algorithm Details

### AFN Architecture

The AFN uses a **5-model Random Forest ensemble** with the following characteristics:

- **Surrogate Models**: 5 Random Forest Regressors (100 trees each)
- **Uncertainty Estimation**: Ensemble variance across predictions
- **Input Layer**: Variable dimension (based on problem)
- **Hyperparameters**: 
  - `n_estimators=100` per Random Forest
  - `n_jobs=-1` for parallel processing
  - `random_state` for reproducibility

### Algorithm Flow

1. **Initial Sampling**: Generate 30-200 random samples (adaptive based on dimension)
2. **Surrogate Training**: Train 5-model Random Forest ensemble on collected data
3. **Candidate Selection**: 
   - Generate 1000-2000 candidate points
   - Find potential optima (lowest predictions)
   - Find high-uncertainty regions (std > 0.03)
   - Select 8 best points combining both criteria
4. **Expensive Evaluation**: Evaluate selected points
5. **Update & Repeat**: Update best solution and repeat until convergence

### Paper Specifications

- **Uncertainty threshold**: 0.03
- **Batch size**: 8 new points per iteration
- **Max evaluations**: 100
- **Convergence**: improvement < 10^-6 over 10 consecutive evaluations
- **Ensemble**: 5 Random Forest models with 100 trees each

### Comparison Algorithms

#### Hansen CMA-ES
- **Implementation**: Covariance Matrix Adaptation Evolution Strategy
- **Features**: Bound constraints, population size 20, adaptive step size
- **Reference**: Hansen, 2019 (global linear/quad surrogate; rank-corr gating)

#### Bajer GP-EI
- **Implementation**: Gaussian Process with Expected Improvement acquisition
- **Features**: GP surrogate with EI-driven candidate selection
- **Reference**: Bajer et al., 2019 (GP uncertainty, EI-driven selection)

## 📊 Available Test Functions (COCO/BBOB Benchmark Suite)

| ID | Function Name | Description | Type |
|----|---------------|-------------|------|
| 1 | Sphere | Unimodal, smooth | Separable |
| 8 | Rosenbrock | Valley-shaped, deceptive | Non-separable |
| 23 | Katsuura | Highly multimodal, rugged | Non-separable |

**Note**: We use the standardized COCO/BBOB benchmark functions, which are the gold standard for optimization algorithm evaluation in academic research.

## 📈 Results & Metrics

### Performance Metrics

The comparison computes 5 key metrics:

1. **Convergence Speed**: Evaluations needed to reach 95% of best solution
2. **Optimization Accuracy**: How close to the true optimum
3. **Resource Utilization**: Efficiency of evaluation usage
4. **Exploitation Balance**: Balance between exploration and exploitation
5. **Robustness**: Consistency across multiple runs

### Generated Outputs

```
results/afn_ga_pso_aco_YYYYMMDD_HHMMSS/
├── runs.json                    # Raw results from all runs
├── metrics_summary.json         # Computed metrics and statistics
├── config.json                  # Configuration used
├── convergence_speed.png        # Convergence speed comparison
├── optimization_accuracy.png    # Optimization accuracy comparison
├── resource_utilization.png     # Resource utilization comparison
├── exploitation_balance.png     # Exploitation balance comparison
└── robustness.png              # Robustness comparison
```

### Command Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--functions` | Test function IDs | `1,2,3` | `1,2,3,4,5` |
| `--dimensions` | Problem dimensions | `2,5` | `2,5,10,20` |
| `--n_runs` | Runs per test case | `10` | `20`, `30`, `50` |
| `--max_evals` | Max evaluations | `100` | `200`, `500` |
| `--output_dir` | Results directory | `results` | `my_results` |
| `--verbose` | Detailed output | `False` | Flag |

### Expected Results

Based on the ensemble implementation, AFN should demonstrate:

- **Superior convergence speed** compared to GA, PSO, ACO
- **Higher optimization accuracy** across different functions
- **Better resource utilization** with fewer expensive evaluations
- **Improved robustness** across multiple runs
- **Faster execution** compared to neural network surrogates

## 🐛 Troubleshooting

### Common Issues

1. **Virtual Environment Not Activated**: Make sure you're in an activated virtual environment
   ```bash
   # Check if virtual environment is active (should see (.venv) in prompt)
   # If not, activate it:
   # Windows: .\.venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   ```

2. **Import Error**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

3. **COCO Installation Issues**: Use the simple version without COCO
   ```bash
   # Test simple functions instead
   python -c "from afn.simple_test_functions import test_all_functions; test_all_functions()"
   ```

4. **Memory Issues**: Reduce batch size or max evaluations for large problems

5. **Scikit-learn Version Issues**: Ensure scikit-learn >= 1.1.0
   ```bash
   pip install --upgrade scikit-learn
   ```

6. **Long Running Times**: Use smaller `--n_runs` and `--max_evals` parameters

### Getting Help

- Check ensemble installation: `python -c "import sklearn; print(sklearn.__version__)"`
- Test individual components: `python -c "from afn import AFNCore; print('✅ Ready!')"`
- Use verbose mode: `--verbose` flag in commands
- Start with quick test: `--functions 1 --dimensions 2 --n_runs 3`

## 🔬 Advanced Usage

### Custom Surrogate Models

You can extend the EnsembleRegressor class to use different regressor models:

```python
from sklearn.ensemble import GradientBoostingRegressor
from afn.afn_core import EnsembleRegressor

class CustomEnsemble(EnsembleRegressor):
    def __init__(self, input_dim: int, n_models: int = 5, random_state: int = 42):
        super().__init__(input_dim, n_models, random_state)
        # Replace with your preferred model
        for i in range(n_models):
            model = GradientBoostingRegressor(
                n_estimators=100,
                random_state=random_state + i
            )
            self.models.append(model)
```

### Performance Tuning

Adjust AFN parameters for better performance:

```python
afn = AFNCore(
    input_dim=dimension,
    bounds=bounds,
    uncertainty_threshold=0.05,  # Higher = more exploration
    batch_size=16,              # Larger batches for faster convergence
    max_evaluations=200,        # More evaluations for better results
    n_models=10,                # More models for better uncertainty estimation
)
```

## 📚 References

- **Paper**: "Artificial neural networks as surrogate models in optimization"
- **BBOB Suite**: [COCO/BBOB Benchmark](https://coco.gforge.inria.fr/)
- **Scikit-learn**: [Official Documentation](https://scikit-learn.org/)

## 🤝 Contributing

This implementation follows the paper specifications exactly. For modifications or improvements:

1. Maintain compatibility with existing interfaces
2. Update tests accordingly
3. Document any changes to the core algorithm
4. Ensure scikit-learn version compatibility

## 📄 License

This project is for research and educational purposes. Please cite the original paper if using this implementation in your research.

---

**🎉 Ready to run AFN optimization! Start with `python run_afn_ga_pso_aco_comparison.py --functions 1 --dimensions 2 --n_runs 3` for a quick test!**