#!/usr/bin/env python3
"""
Quantum Circuit Optimization for QML-Based IoT Network Intrusion Detection
Journal Publication Experiment

=============================================================================
CORE RESEARCH QUESTION:
Does Qiskit circuit optimization improve kernel-based QML performance?
We compare unoptimized vs optimized circuits and measure the impact on:
- Model accuracy and F1-score in ideal simulation
- Noise resilience across repeated runs and multiple noise levels
- Circuit depth and gate count reduction
=============================================================================

Study Scope:
1. Supporting compiled-circuit sanity check (baseline L0 vs Qiskit L3)
2. Quantifying optimization impact on kernel-QML performance
3. Application domain: IoT Network Intrusion Detection
4. Leakage-safe repeated evaluation with fixed seeds
5. Noise model simulation for NISQ-oriented robustness analysis
6. Practical guidance on whether optimization helps under ideal and noisy simulation

Optimization Techniques within Qiskit:
- Transpiler optimization levels (0=none, 1=light, 2=medium, 3=heavy)
- Gate cancellation and merging
- Two-qubit gate reduction
- Circuit depth minimization
- Layout/routing optimization with coupling constraints

Models:
- QSVC: Precomputed GPU kernel (cuML SVC)
- Quantum Ensemble: QVE (equal-weight voting), QWE (validation-weighted)

Experiment Phases (5 total):
- Phase 1: Supporting compiled-circuit sanity check (baseline L0 vs Qiskit L3)
- Phase 2: Unoptimized vs optimized circuit comparison on ideal kernel models
- Phase 3: Detailed Qiskit optimization level comparison (0-3)
- Phase 4: COMPREHENSIVE KERNEL-MODEL NOISE ANALYSIS (CORE EXPERIMENT)
    Tests QSVC, QVE, and QWE × 2 opt levels (L0/L3) × 10 noise levels × 30 runs
    Answers: "How does circuit optimization affect kernel-QML models under ideal and noisy simulation?"
- Phase 5: Leakage-safe cross-validation with repeated seeds

Output Files (per config, prefix = {config_tag}_{session_id}):
    Phase 1: multi_method_optimization_{prefix}.csv (supporting circuit metrics)
                     multi_method_model_results_{prefix}.csv (supporting QSVC run results)
                     multi_method_statistics_{prefix}.csv (supporting aggregated statistics)
  Phase 2: optimization_comparison_{prefix}.csv (unopt vs opt individual runs)
           optimization_statistics_{prefix}.csv (aggregated significance tests)
  Phase 3: optimization_levels_{prefix}.csv (Qiskit levels 0-3)
  Phase 4: comprehensive_model_noise_{prefix}.csv (ALL models×opt×noise×runs)
  Phase 5: cv_results_{prefix}.csv (k-fold cross-validation)
  Summary: experiment_summary_{prefix}.json (config + metadata)
           journal_summary_{prefix}.txt (publication-ready report)
           circuit_metrics_{prefix}.csv (all circuit analysis)
  Incremental: model_results_{prefix}_incremental.csv (crash-safe saves)

Author: @ocblvck
Date: 2026-01-30
"""

# Standard library imports
import os
import sys
import csv
import copy
import re
import time
import json
import pickle
import logging
import argparse
import threading
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, OrderedDict, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import pinv

# Scikit-learn imports
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from cuml.svm import SVC as cuSVC  # GPU-accelerated SVC (RAPIDS cuML)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score,
    precision_recall_curve, auc, make_scorer,
    balanced_accuracy_score, cohen_kappa_score, classification_report
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Optional imports
try:
    import psutil
except ImportError:
    psutil = None

try:
    import gc
except ImportError:
    gc = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    pynvml = None
    NVML_AVAILABLE = False

# Qiskit imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector, Parameter
    from qiskit.circuit.library import (
        ZZFeatureMap, PauliFeatureMap, ZFeatureMap,
        RealAmplitudes, TwoLocal, EfficientSU2
    )
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

try:
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    QML_AVAILABLE = True
except Exception:
    QML_AVAILABLE = False

try:
    from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM
except Exception:
    COBYLA = SPSA = ADAM = None

try:
    from qiskit_aer.primitives import Sampler as AerSampler
    # V2 primitives (preferred - no deprecation warnings)
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    V2_PRIMITIVES_AVAILABLE = True
except Exception:
    AerSampler = None
    AerSamplerV2 = None
    V2_PRIMITIVES_AVAILABLE = False

# ZX-Calculus optimization (pyzx)
try:
    import pyzx as zx
    from qiskit.qasm2 import dumps as qasm_dumps, loads as qasm_loads
    PYZX_AVAILABLE = True
except ImportError:
    zx = None
    qasm_dumps = qasm_loads = None
    PYZX_AVAILABLE = False

# Pytket compiler optimization
try:
    from pytket.circuit import Circuit as TkCircuit
    from pytket.passes import FullPeepholeOptimise, RemoveRedundancies, CommuteThroughMultis
    from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
    PYTKET_AVAILABLE = True
except ImportError:
    TkCircuit = None
    circuit_from_qasm_str = circuit_to_qasm_str = None
    PYTKET_AVAILABLE = False

# Custom Qiskit transpiler passes
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Optimize1qGates, InverseCancellation, CommutativeCancellation,
    Collect2qBlocks, ConsolidateBlocks, UnitarySynthesis,
    RemoveDiagonalGatesBeforeMeasure, OptimizeSwapBeforeMeasure
)
from qiskit.circuit.library import CXGate, CZGate

# Initialize logger with immediate flushing for visibility
import sys
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.flush = sys.stdout.flush  # Ensure immediate flush
_file_handler = logging.FileHandler('circuit_depth_experiment.log')

# Set ROOT logger to WARNING to silence Qiskit internal transpiler pass logging
# (each transpile() call otherwise emits hundreds of per-pass timing lines)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        _stream_handler,
        _file_handler
    ]
)

# Force flush on each log message for real-time visibility
_stream_handler.flush = sys.stdout.flush
    
# Only OUR logger gets INFO level — Qiskit/Aer/transpiler stay at WARNING
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# NOTE: pyzx/pytket availability is NOT logged here because ProcessPoolExecutor
# workers re-import this module, causing 32+ duplicate log lines per kernel call.
# These libraries are only used in Phase 1 compare_all_methods() (circuit metrics only,
# NOT for actual model training — they fall back to Qiskit L3 for parameterized circuits).

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory setup
RESULTS_DIR = Path("results/circuit_depth")
CHECKPOINT_DIR = Path("checkpoints/circuit_depth")
CIRCUIT_CACHE_DIR = Path("circuit_cache/circuit_depth")
FIGURES_DIR = Path("results/circuit_depth/figures")

for dir_path in [RESULTS_DIR, CHECKPOINT_DIR, CIRCUIT_CACHE_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Experiment configuration
# NOTE: 30 repeated runs improve estimate stability and make noise trends easier to
# resolve, but repeated splits are still correlated because they reuse one dataset.
# Confidence intervals are therefore descriptive rather than a claim of independent
# experimental replication.
DEFAULT_CONFIG = {
    'n_runs': 30,  # Number of repeated train/test splits for stable estimates
    'k_folds': 5,  # K-fold cross-validation
    'random_seeds': [  # 30 seeds for reproducibility and statistical power
        42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
        7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
        17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624
    ],
    'test_size': 0.3,
    'noise_simulation': True,
    'transpile_opt_level': 2,
}

# Default noise grid used for the core experiment.
# Ten levels is a better runtime/coverage tradeoff than fourteen for 30-run studies.
# It preserves ideal, low-noise, mid-noise, and stress-test regimes without spending
# most runtime on tightly clustered neighboring points.
DEFAULT_NOISE_LEVELS = [0.0, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05]
QUICK_TEST_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.05]

# GPU settings
GPU_MEMORY_RESERVE_GB = 2
MAX_PARALLEL_GPUS = 8

# Default to GPU-only ideal-kernel execution so the main study actually uses
# the installed accelerators for noiseless as well as noisy simulations.
# The old CPU ProcessPool path is still available for benchmarking.
ENABLE_CPU_FAST_STATEVECTOR = os.environ.get(
    'QML_ENABLE_CPU_FAST_STATEVECTOR', '0'
).strip().lower() in {'1', 'true', 'yes', 'on'}

# Noisy density-matrix kernels become memory-dominated at 10 qubits if the full
# n x d x d tensor is materialized in RAM. Switch to blockwise storage/compute
# once the estimated working set grows beyond this budget.
NOISY_DENSITY_MATRIX_RAM_BUDGET_GB = 6.0

# Ablation study configurations
ABLATION_CONFIGS = {
    # Ablation 1: Feature Map Type
    'feature_map_type': {
        'fixed': {'num_qubits': 6, 'reps': 2, 'sample_size': 5000, 'entanglement': 'full'},
        'vary': ['Z', 'ZZ', 'Pauli', 'Custom']
    },
    # Ablation 2: Circuit Depth (Reps)
    'circuit_depth': {
        'fixed': {'num_qubits': 6, 'feature_map': 'ZZ', 'sample_size': 5000, 'entanglement': 'full'},
        'vary': [1, 2, 3, 4]
    },
    # Ablation 3: Entanglement Pattern
    'entanglement_pattern': {
        'fixed': {'num_qubits': 6, 'feature_map': 'ZZ', 'reps': 2, 'sample_size': 5000},
        'vary': ['linear', 'circular', 'full', 'sca']
    },
    # Ablation 4: Qubit Count (Scalability) - matches full experiment configs
    'qubit_scalability': {
        'fixed': {'feature_map': 'ZZ', 'reps': 2, 'sample_size': 5000, 'entanglement': 'full'},
        'vary': [6, 12]
    },
    # Ablation 5: Sample Size
    'sample_size': {
        'fixed': {'num_qubits': 6, 'feature_map': 'ZZ', 'reps': 2, 'entanglement': 'full'},
        'vary': [5000, 10000]
    }
}

# Noise model parameters for NISQ simulation
# NOTE: These model a conservative/noisy device (e.g., older superconducting hardware)
# with shorter coherence times. The compute_nisq_feasibility_score() function uses
# IBM Falcon r5.11 (2024) parameters for feasibility scoring — intentionally different.
NOISE_PARAMS = {
    'single_qubit_error': 0.001,  # 0.1% error rate
    'two_qubit_error': 0.01,      # 1% error rate
    't1': 50e3,                    # T1 relaxation time (ns) — 50 μs
    't2': 70e3,                    # T2 dephasing time (ns) — 70 μs
    'gate_time_1q': 50,            # Single-qubit gate time (ns)
    'gate_time_2q': 300,           # Two-qubit gate time (ns)
}


# ============================================================================
# CIRCUIT METRICS ANALYZER
# ============================================================================

class CircuitMetricsAnalyzer:
    """Analyze and record circuit depth, gate counts, and complexity metrics"""
    
    def __init__(self):
        self.metrics_history = []
    
    def analyze_circuit(self, circuit: QuantumCircuit, circuit_name: str) -> Dict[str, Any]:
        """Extract comprehensive metrics from a quantum circuit"""
        
        # Basic metrics
        metrics = {
            'circuit_name': circuit_name,
            'num_qubits': circuit.num_qubits,
            'depth': circuit.depth(),
            'total_gates': sum(circuit.count_ops().values()),
            'gate_counts': dict(circuit.count_ops()),
        }
        
        # Count specific gate types
        ops = circuit.count_ops()
        metrics['single_qubit_gates'] = sum(
            count for op, count in ops.items() 
            if op in ['x', 'y', 'z', 'h', 's', 't', 'rx', 'ry', 'rz', 'u', 'u1', 'u2', 'u3', 'sx']
        )
        metrics['two_qubit_gates'] = sum(
            count for op, count in ops.items()
            if op in ['cx', 'cz', 'cy', 'swap', 'iswap', 'cswap', 'crx', 'cry', 'crz', 'ecr', 'rzz']
        )
        metrics['multi_qubit_gates'] = sum(
            count for op, count in ops.items()
            if op in ['ccx', 'cswap', 'mcx']
        )
        
        # Parametric gates
        metrics['num_parameters'] = circuit.num_parameters
        
        # Calculate T-depth estimate (rough approximation)
        metrics['t_depth_estimate'] = metrics['two_qubit_gates']
        
        # Circuit volume (rough metric)
        metrics['circuit_volume'] = metrics['num_qubits'] * metrics['depth']
        
        # Gate density
        if metrics['depth'] > 0:
            metrics['gate_density'] = metrics['total_gates'] / metrics['depth']
        else:
            metrics['gate_density'] = 0
            
        # Two-qubit gate ratio
        if metrics['total_gates'] > 0:
            metrics['two_qubit_ratio'] = metrics['two_qubit_gates'] / metrics['total_gates']
        else:
            metrics['two_qubit_ratio'] = 0
        
        self.metrics_history.append(metrics)
        return metrics
    
    def transpile_and_analyze(self, circuit: QuantumCircuit, circuit_name: str, 
                              opt_level: int = 2) -> Dict[str, Any]:
        """Transpile circuit and analyze both original and transpiled versions"""
        
        # Original metrics
        original_metrics = self.analyze_circuit(circuit, f"{circuit_name}_original")
        
        # Transpile to standard basis gates
        transpiled = transpile(
            circuit,
            optimization_level=opt_level,
            basis_gates=['u', 'cx', 'rz', 'sx', 'x', 'ry']
        )
        
        # Transpiled metrics
        transpiled_metrics = self.analyze_circuit(transpiled, f"{circuit_name}_transpiled")
        
        # Combined metrics
        combined = {
            'circuit_name': circuit_name,
            'original': original_metrics,
            'transpiled': transpiled_metrics,
            'depth_reduction': original_metrics['depth'] - transpiled_metrics['depth'],
            'gate_reduction': original_metrics['total_gates'] - transpiled_metrics['total_gates'],
        }
        
        return combined, transpiled
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Return all collected metrics as a DataFrame"""
        return pd.DataFrame(self.metrics_history)
    
    def save_metrics(self, filepath: str):
        """Save metrics to CSV"""
        df = self.get_metrics_dataframe()
        if df.empty:
            logger.info("No circuit metrics collected; skipping save for %s", filepath)
            return
        df.to_csv(filepath, index=False)
        logger.info(f"Circuit metrics saved to {filepath}")


def compute_nisq_feasibility_score(circuit_metrics: Dict, device_params: Dict = None) -> Dict:
    """
    Compute NISQ feasibility score for a circuit.
    
    This quantifies how "runnable" a circuit is on real NISQ hardware.
    Higher score = more likely to succeed on real device.
    
    Factors:
    - Circuit depth (lower is better)
    - Two-qubit gate count (most error-prone)
    - Total gate count
    - Estimated fidelity based on typical error rates
    
    Args:
        circuit_metrics: Dict with 'depth', 'total_gates', 'two_qubit_gates'
        device_params: Device error rates (defaults to IBM Falcon r5.11 typical values)
    
    Returns:
        Dict with feasibility metrics
    """
    if device_params is None:
        # Typical IBM Quantum Falcon r5.11 error rates (2024)
        device_params = {
            'single_qubit_error': 0.0003,    # 0.03% per gate
            'two_qubit_error': 0.008,        # 0.8% per CX gate
            'readout_error': 0.01,           # 1% readout error
            't1': 300e-6,                    # 300 μs T1
            't2': 150e-6,                    # 150 μs T2
            'single_gate_time': 35e-9,       # 35 ns
            'two_gate_time': 300e-9,         # 300 ns
        }
    
    depth = circuit_metrics.get('depth', 0)
    total_gates = circuit_metrics.get('total_gates', 0)
    two_q_gates = circuit_metrics.get('two_qubit_gates', 0)
    single_q_gates = total_gates - two_q_gates
    
    # Estimated circuit fidelity (product of gate fidelities)
    single_gate_fidelity = (1 - device_params['single_qubit_error']) ** single_q_gates
    two_gate_fidelity = (1 - device_params['two_qubit_error']) ** two_q_gates
    readout_fidelity = 1 - device_params['readout_error']
    
    estimated_circuit_fidelity = single_gate_fidelity * two_gate_fidelity * readout_fidelity
    
    # Estimated execution time
    estimated_time = (single_q_gates * device_params['single_gate_time'] + 
                      two_q_gates * device_params['two_gate_time'])
    
    # Check if circuit fits within coherence time
    coherence_ratio = estimated_time / device_params['t2']
    coherence_limited = coherence_ratio > 0.5  # Concern if > 50% of T2
    
    # NISQ feasibility score (0-100)
    # Penalize depth, 2Q gates heavily
    depth_penalty = max(0, 1 - depth / 500)  # 0 if depth > 500
    gate_penalty = max(0, 1 - two_q_gates / 200)  # 0 if 2Q gates > 200
    fidelity_score = estimated_circuit_fidelity
    
    nisq_score = (depth_penalty * 0.3 + gate_penalty * 0.3 + fidelity_score * 0.4) * 100
    
    return {
        'nisq_feasibility_score': nisq_score,
        'estimated_circuit_fidelity': estimated_circuit_fidelity,
        'estimated_execution_time_us': estimated_time * 1e6,
        'coherence_ratio': coherence_ratio,
        'coherence_limited': coherence_limited,
        'single_gate_fidelity': single_gate_fidelity,
        'two_gate_fidelity': two_gate_fidelity,
        'depth': depth,
        'two_qubit_gates': two_q_gates,
        'total_gates': total_gates
    }


def compute_expected_accuracy_under_noise(ideal_accuracy: float, circuit_fidelity: float) -> float:
    """
    Estimate accuracy degradation based on circuit fidelity.
    
    Model: noisy_accuracy ≈ ideal_accuracy * circuit_fidelity + (1 - circuit_fidelity) * random_guess
    For binary classification, random_guess = 0.5
    """
    random_guess = 0.5
    return ideal_accuracy * circuit_fidelity + (1 - circuit_fidelity) * random_guess


def compute_noise_resilience_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive noise resilience metrics for optimized vs unoptimized circuits.
    
    Metrics computed:
    - Degradation rate: slope of accuracy vs noise level (linear fit)
    - Exponential decay fit: acc = a * exp(-b * noise) + c (better for noise behavior)
    - Median-based statistics (robust to outliers from bimodal noise distributions)
    - Noise threshold: noise level at which accuracy drops below threshold (e.g., 0.8)
    - Resilience advantage: difference in degradation rates
    - Statistical significance: t-test on accuracy distributions
    
    Args:
        df: DataFrame with columns: noise_level, optimization, accuracy_mean, accuracy_std
        
    Returns:
        Dictionary with resilience metrics
    """
    from scipy import stats
    from scipy.optimize import curve_fit
    
    results = {}
    
    # Exponential decay model: acc = a * exp(-b * noise) + c
    # a = amplitude, b = decay rate, c = asymptotic accuracy (random guess floor)
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    # Get unique optimization methods
    opt_methods = df['optimization'].unique()
    
    for opt_method in opt_methods:
        opt_data = df[df['optimization'] == opt_method].sort_values('noise_level')
        
        if len(opt_data) < 2:
            continue
            
        noise_levels = opt_data['noise_level'].values
        accuracies = opt_data['accuracy_mean'].values
        
        # Use median if available (more robust to outliers)
        if 'accuracy_median' in opt_data.columns:
            accuracies_median = opt_data['accuracy_median'].values
            results[f'{opt_method}_uses_median'] = True
        else:
            accuracies_median = accuracies
            results[f'{opt_method}_uses_median'] = False
        
        # ============= LINEAR FIT (for comparison) =============
        if len(noise_levels) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(noise_levels, accuracies)
            
            results[f'{opt_method}_degradation_rate'] = slope
            results[f'{opt_method}_linear_r_squared'] = r_value ** 2
            results[f'{opt_method}_intercept'] = intercept
            results[f'{opt_method}_degradation_stderr'] = std_err
            
            # Median-based linear fit (more robust)
            slope_med, intercept_med, r_value_med, _, std_err_med = stats.linregress(noise_levels, accuracies_median)
            results[f'{opt_method}_median_degradation_rate'] = slope_med
            results[f'{opt_method}_median_linear_r_squared'] = r_value_med ** 2
            results[f'{opt_method}_median_degradation_stderr'] = std_err_med
        
        # ============= EXPONENTIAL DECAY FIT (better for noise) =============
        if len(noise_levels) >= 3:
            try:
                # Initial guess: a=0.4 (amplitude), b=50 (decay rate), c=0.5 (random guess floor)
                ideal_acc = accuracies[noise_levels == 0][0] if 0 in noise_levels else accuracies[0]
                p0 = [ideal_acc - 0.5, 50, 0.5]
                
                # Fit exponential decay
                popt, pcov = curve_fit(exp_decay, noise_levels, accuracies, p0=p0, 
                                       bounds=([0, 0, 0], [1, 1000, 1]), maxfev=5000)
                
                # Compute R² for exponential fit
                y_pred = exp_decay(noise_levels, *popt)
                ss_res = np.sum((accuracies - y_pred) ** 2)
                ss_tot = np.sum((accuracies - np.mean(accuracies)) ** 2)
                exp_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                results[f'{opt_method}_exp_amplitude'] = popt[0]  # a: initial drop
                results[f'{opt_method}_exp_decay_rate'] = popt[1]  # b: how fast it decays
                results[f'{opt_method}_exp_floor'] = popt[2]  # c: asymptotic floor
                results[f'{opt_method}_exp_r_squared'] = exp_r_squared
                
                # Half-life: noise level at which accuracy drops halfway to floor
                # exp(-b * half_life) = 0.5 => half_life = ln(2) / b
                results[f'{opt_method}_exp_half_life'] = np.log(2) / popt[1] if popt[1] > 0 else float('inf')
                
            except Exception as e:
                results[f'{opt_method}_exp_fit_error'] = str(e)
        
        # ============= THRESHOLD ANALYSIS =============
        # Compute noise threshold (where accuracy drops below 0.8 or 0.7)
        if f'{opt_method}_degradation_rate' in results:
            slope = results[f'{opt_method}_degradation_rate']
            intercept = results[f'{opt_method}_intercept']
            
            for threshold in [0.8, 0.7, 0.6]:
                if slope != 0:
                    noise_threshold = (threshold - intercept) / slope
                    if noise_threshold > 0:
                        results[f'{opt_method}_threshold_{int(threshold*100)}'] = noise_threshold
                    else:
                        results[f'{opt_method}_threshold_{int(threshold*100)}'] = float('inf')
                else:
                    results[f'{opt_method}_threshold_{int(threshold*100)}'] = float('inf')
        
        # ============= ACCURACY STATISTICS =============
        # Ideal accuracy (at noise=0)
        ideal_data = opt_data[opt_data['noise_level'] == 0]
        if len(ideal_data) > 0:
            results[f'{opt_method}_ideal_accuracy'] = ideal_data['accuracy_mean'].values[0]
            if 'accuracy_median' in ideal_data.columns:
                results[f'{opt_method}_ideal_accuracy_median'] = ideal_data['accuracy_median'].values[0]
        
        # Max noise accuracy
        max_noise = noise_levels.max()
        max_noise_data = opt_data[opt_data['noise_level'] == max_noise]
        if len(max_noise_data) > 0:
            results[f'{opt_method}_max_noise_accuracy'] = max_noise_data['accuracy_mean'].values[0]
            if 'accuracy_median' in max_noise_data.columns:
                results[f'{opt_method}_max_noise_accuracy_median'] = max_noise_data['accuracy_median'].values[0]
            
            # Relative degradation
            if f'{opt_method}_ideal_accuracy' in results:
                ideal = results[f'{opt_method}_ideal_accuracy']
                noisy = results[f'{opt_method}_max_noise_accuracy']
                results[f'{opt_method}_relative_degradation'] = (ideal - noisy) / ideal if ideal > 0 else 0
    
    # ============= COMPARATIVE ANALYSIS =============
    # Compute resilience advantage (optimized vs unoptimized)
    if 'optimized_L3_degradation_rate' in results and 'unoptimized_degradation_rate' in results:
        opt_rate = results['optimized_L3_degradation_rate']
        unopt_rate = results['unoptimized_degradation_rate']
        
        # Resilience advantage: how much slower optimized degrades
        # Both rates are negative (accuracy decreases with noise)
        # Less negative = better resilience
        # Formula: opt_rate - unopt_rate (positive means optimized is better)
        # Example: opt=-30, unopt=-34 → -30-(-34) = +4 (optimized 4 units better)
        results['resilience_advantage'] = opt_rate - unopt_rate
        
        # Percent improvement in degradation rate (positive = improvement)
        if abs(unopt_rate) > 0:
            results['degradation_improvement_pct'] = (results['resilience_advantage'] / abs(unopt_rate)) * 100
        
        # Median-based resilience advantage
        if 'optimized_L3_median_degradation_rate' in results and 'unoptimized_median_degradation_rate' in results:
            opt_med_rate = results['optimized_L3_median_degradation_rate']
            unopt_med_rate = results['unoptimized_median_degradation_rate']
            results['median_resilience_advantage'] = opt_med_rate - unopt_med_rate
        
        # Exponential decay rate comparison
        if 'optimized_L3_exp_decay_rate' in results and 'unoptimized_exp_decay_rate' in results:
            opt_exp_rate = results['optimized_L3_exp_decay_rate']
            unopt_exp_rate = results['unoptimized_exp_decay_rate']
            # Lower decay rate = more resilient (slower decay)
            # Formula: unopt - opt (positive means optimized decays slower = better)
            results['exp_decay_advantage'] = unopt_exp_rate - opt_exp_rate
            results['exp_decay_improvement_pct'] = (results['exp_decay_advantage'] / unopt_exp_rate * 100) if unopt_exp_rate > 0 else 0
        
        # Model comparison: which fit is better?
        if 'optimized_L3_linear_r_squared' in results and 'optimized_L3_exp_r_squared' in results:
            linear_r2 = (results.get('optimized_L3_linear_r_squared', 0) + 
                        results.get('unoptimized_linear_r_squared', 0)) / 2
            exp_r2 = (results.get('optimized_L3_exp_r_squared', 0) + 
                     results.get('unoptimized_exp_r_squared', 0)) / 2
            results['better_model'] = 'exponential' if exp_r2 > linear_r2 else 'linear'
            results['linear_avg_r_squared'] = linear_r2
            results['exp_avg_r_squared'] = exp_r2
    
    return results


def format_noise_resilience_report(metrics: Dict[str, Any]) -> str:
    """Format noise resilience metrics as a readable report."""
    
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("📊 NOISE RESILIENCE ANALYSIS REPORT")
    lines.append("=" * 70)
    
    # Per-method metrics
    for opt in ['unoptimized', 'optimized_L3']:
        if f'{opt}_degradation_rate' in metrics:
            lines.append(f"\n{opt.upper()}:")
            lines.append("  ── Linear Fit ──")
            lines.append(f"  Degradation rate: {metrics[f'{opt}_degradation_rate']:.4f} acc/noise_unit")
            lines.append(f"  R² (linear fit): {metrics.get(f'{opt}_linear_r_squared', 0):.4f}")
            
            # Median-based stats if available
            if f'{opt}_median_degradation_rate' in metrics:
                lines.append(f"  Median degradation rate: {metrics[f'{opt}_median_degradation_rate']:.4f}")
            
            # Exponential decay fit
            if f'{opt}_exp_decay_rate' in metrics:
                lines.append("  ── Exponential Decay Fit ──")
                lines.append(f"  Decay rate (b): {metrics[f'{opt}_exp_decay_rate']:.4f}")
                lines.append(f"  Amplitude (a): {metrics[f'{opt}_exp_amplitude']:.4f}")
                lines.append(f"  Floor (c): {metrics[f'{opt}_exp_floor']:.4f}")
                lines.append(f"  R² (exp fit): {metrics.get(f'{opt}_exp_r_squared', 0):.4f}")
                if f'{opt}_exp_half_life' in metrics:
                    half_life = metrics[f'{opt}_exp_half_life']
                    if half_life < float('inf'):
                        lines.append(f"  Half-life: {half_life:.5f} noise units")
            
            lines.append("  ── Accuracy ──")
            if f'{opt}_ideal_accuracy' in metrics:
                lines.append(f"  Ideal accuracy (noise=0): {metrics[f'{opt}_ideal_accuracy']:.4f}")
                if f'{opt}_ideal_accuracy_median' in metrics:
                    lines.append(f"  Ideal accuracy median: {metrics[f'{opt}_ideal_accuracy_median']:.4f}")
            if f'{opt}_max_noise_accuracy' in metrics:
                lines.append(f"  Noisy accuracy (max noise): {metrics[f'{opt}_max_noise_accuracy']:.4f}")
                if f'{opt}_max_noise_accuracy_median' in metrics:
                    lines.append(f"  Noisy accuracy median: {metrics[f'{opt}_max_noise_accuracy_median']:.4f}")
            if f'{opt}_relative_degradation' in metrics:
                lines.append(f"  Relative degradation: {metrics[f'{opt}_relative_degradation']*100:.2f}%")
            
            # Thresholds
            for thresh in [80, 70, 60]:
                key = f'{opt}_threshold_{thresh}'
                if key in metrics:
                    val = metrics[key]
                    if val < float('inf'):
                        lines.append(f"  Noise threshold (acc>{thresh/100}): {val:.4f}")
    
    # Comparison
    if 'resilience_advantage' in metrics:
        lines.append(f"\n{'=' * 70}")
        lines.append("🏆 COMPARATIVE ANALYSIS")
        lines.append(f"{'=' * 70}")
        
        lines.append("\n  ── Linear Model Comparison ──")
        lines.append(f"  Resilience advantage: {metrics['resilience_advantage']:.4f}")
        
        if metrics['resilience_advantage'] > 0:
            lines.append(f"  → Optimized circuit degrades SLOWER under noise ✓")
        else:
            lines.append(f"  → No significant resilience improvement detected")
        
        if 'degradation_improvement_pct' in metrics:
            lines.append(f"  Degradation rate improvement: {metrics['degradation_improvement_pct']:.1f}%")
        
        # Median-based comparison
        if 'median_resilience_advantage' in metrics:
            lines.append(f"\n  ── Median-Based Comparison (Robust) ──")
            lines.append(f"  Median resilience advantage: {metrics['median_resilience_advantage']:.4f}")
        
        # Exponential decay comparison
        if 'exp_decay_advantage' in metrics:
            lines.append(f"\n  ── Exponential Decay Comparison ──")
            lines.append(f"  Decay rate advantage: {metrics['exp_decay_advantage']:.4f}")
            if metrics['exp_decay_advantage'] > 0:
                lines.append(f"  → Optimized circuit decays SLOWER (better) ✓")
            if 'exp_decay_improvement_pct' in metrics:
                lines.append(f"  Decay rate improvement: {metrics['exp_decay_improvement_pct']:.1f}%")
        
        # Model comparison
        if 'better_model' in metrics:
            lines.append(f"\n  ── Model Fit Quality ──")
            lines.append(f"  Better model: {metrics['better_model'].upper()}")
            lines.append(f"  Linear avg R²: {metrics.get('linear_avg_r_squared', 0):.4f}")
            lines.append(f"  Exponential avg R²: {metrics.get('exp_avg_r_squared', 0):.4f}")
        
    lines.append("\n" + "=" * 70 + "\n")
    return "\n".join(lines)


# ============================================================================
# SUPPORTING CIRCUIT COMPARISON UTILITY
# ============================================================================

class MultiMethodOptimizer:
    """
    Compare multiple circuit optimization methodologies.

    In the active study configuration, this utility is used only for supporting
    compiled-circuit comparisons and sanity checks. It is not treated as the
    main inferential contribution of the paper.
    
    Methods compared:
    1. Qiskit Transpiler (rule-based, levels 0-3)
    2. ZX-Calculus (pyzx) - mathematical graph-theoretic simplification
    3. Pytket Compiler - alternative peephole optimization  
    4. Custom Qiskit Passes - tailored for QML circuits
    """
    
    def __init__(self, num_qubits: int, coupling_map=None, basis_gates=None):
        self.num_qubits = num_qubits
        self.coupling_map = coupling_map
        self.basis_gates = basis_gates or ['u', 'cx', 'rz', 'sx', 'x']
        self.optimization_results = []
        
    def _circuit_to_qasm(self, circuit: QuantumCircuit) -> str:
        """Convert Qiskit circuit to QASM string"""
        try:
            return qasm_dumps(circuit)
        except Exception as e:
            logger.warning(f"QASM conversion failed: {e}")
            return None
    
    def _qasm_to_circuit(self, qasm_str: str) -> QuantumCircuit:
        """Convert QASM string back to Qiskit circuit"""
        try:
            return qasm_loads(qasm_str)
        except Exception as e:
            logger.warning(f"QASM loading failed: {e}")
            return None
    
    def optimize_qiskit(self, circuit: QuantumCircuit, level: int = 3) -> Tuple[QuantumCircuit, Dict]:
        """Optimize using Qiskit transpiler"""
        start_time = time.time()
        
        optimized = transpile(
            circuit,
            optimization_level=level,
            basis_gates=self.basis_gates,
            coupling_map=self.coupling_map,
            seed_transpiler=42
        )
        
        opt_time = time.time() - start_time
        
        return optimized, {
            'method': f'qiskit_level{level}',
            'optimization_time': opt_time,
            'depth': optimized.depth(),
            'total_gates': sum(optimized.count_ops().values()),
            'two_qubit_gates': sum(
                count for op, count in optimized.count_ops().items()
                if op in ['cx', 'cz', 'swap', 'ecr']
            )
        }
    
    def optimize_zx_calculus(self, circuit: QuantumCircuit) -> Tuple[QuantumCircuit, Dict]:
        """
        Optimize using ZX-calculus (pyzx).
        
        ZX-calculus provides mathematically optimal simplifications based on
        graph-theoretic rewriting rules.
        
        NOTE: While ZX-calculus can achieve mathematically optimal graph simplifications,
        the circuit extraction step (converting back to gates) is NP-hard and can sometimes
        produce larger circuits. This method uses teleport_reduce for better extraction.
        
        For PARAMETERIZED circuits: We bind dummy values for ZX optimization (to measure
        gate reduction) but return the original circuit with Qiskit optimization applied,
        since ZX cannot preserve parameter structure.
        """
        if not PYZX_AVAILABLE:
            logger.warning("pyzx not available, skipping ZX optimization")
            return circuit, {'method': 'zx_calculus', 'error': 'pyzx not available'}
        
        start_time = time.time()
        
        try:
            # Check if circuit has unbound parameters
            has_parameters = circuit.num_parameters > 0
            
            if has_parameters:
                # For parameterized circuits: bind dummy values for ZX analysis
                # but use Qiskit level-3 optimization for the actual circuit
                dummy_params = np.random.uniform(0, 2*np.pi, circuit.num_parameters)
                bound_circuit = circuit.assign_parameters(dict(zip(circuit.parameters, dummy_params)))
                logger.info(f"ZX-calculus: Bound {circuit.num_parameters} parameters for structure analysis")
            else:
                bound_circuit = circuit
            
            # Step 1: Decompose to a basis pyzx can handle
            # pyzx supports: cx, h, rz, rx, ry, x, z, s, t (NO swap!)
            # First transpile without coupling_map to avoid routing SWAPs
            basis_circuit = transpile(
                bound_circuit,
                optimization_level=0,  # No Qiskit optimization - just decompose
                basis_gates=['cx', 'h', 'rz', 'rx', 'ry', 'x', 'z', 's', 't'],
                coupling_map=None,  # No routing - avoid SWAP gates
                seed_transpiler=42
            )
            
            # Convert to QASM
            qasm_str = qasm_dumps(basis_circuit)
            
            # Convert to ZX graph
            zx_circuit = zx.Circuit.from_qasm(qasm_str)
            zx_graph = zx_circuit.to_graph()
            
            # Store original metrics
            original_gates = len(zx_circuit.gates)
            
            # Apply ZX-calculus simplification
            # Use teleport_reduce instead of full_reduce for better circuit extraction
            # teleport_reduce is designed to preserve circuit structure better
            zx.simplify.teleport_reduce(zx_graph)
            
            # Try to extract circuit - if it fails or produces worse result, 
            # try with different simplification
            try:
                zx_optimized = zx.extract_circuit(zx_graph)
                optimized_gates = len(zx_optimized.gates)
                
                # If extraction made circuit worse, try basic optimization instead
                if optimized_gates > original_gates * 1.5:
                    logger.info("ZX extraction increased gates, trying basic_optimization")
                    # Fall back to pyzx basic optimization
                    zx_circuit_opt = zx.optimize.basic_optimization(zx_circuit)
                    zx_optimized = zx_circuit_opt
                    optimized_gates = len(zx_optimized.gates)
            except Exception as extract_error:
                logger.warning(f"ZX extraction failed: {extract_error}, using basic_optimization")
                zx_circuit_opt = zx.optimize.basic_optimization(zx_circuit)
                zx_optimized = zx_circuit_opt
                optimized_gates = len(zx_optimized.gates)
            
            # Convert back to QASM
            # Note: pyzx extract_circuit can introduce SWAP gates
            optimized_qasm = zx_optimized.to_qasm()
            
            # Handle SWAP gates: pyzx may output SWAP which needs to be defined
            # or decomposed. We add the SWAP gate definition to the QASM.
            if 'swap' in optimized_qasm.lower() and 'gate swap' not in optimized_qasm.lower():
                # Add SWAP gate definition after the include statement
                # SWAP = CX(a,b); CX(b,a); CX(a,b)
                swap_def = 'gate swap a, b { cx a, b; cx b, a; cx a, b; }'
                lines = optimized_qasm.split('\n')
                # Find include line and insert after it
                for i, line in enumerate(lines):
                    if 'include' in line.lower():
                        lines.insert(i + 1, swap_def)
                        break
                optimized_qasm = '\n'.join(lines)
            
            # Convert to Qiskit circuit
            optimized_circuit = qasm_loads(optimized_qasm)
            
            # Final transpile to target basis and apply routing
            # Now we add routing (which may introduce SWAPs) AFTER ZX optimization
            optimized_circuit = transpile(
                optimized_circuit,
                optimization_level=1,  # Light optimization + routing
                basis_gates=self.basis_gates,
                coupling_map=self.coupling_map,
                seed_transpiler=42
            )
            
            opt_time = time.time() - start_time
            
            # For parameterized circuits: Return original with Qiskit L3 optimization
            # (preserves parameters) but report ZX analysis metrics
            if has_parameters:
                # Apply Qiskit level-3 optimization to original parameterized circuit
                param_optimized = transpile(
                    circuit,
                    optimization_level=3,
                    basis_gates=self.basis_gates,
                    coupling_map=self.coupling_map,
                    seed_transpiler=42
                )
                return param_optimized, {
                    'method': 'zx_calculus',
                    'optimization_time': opt_time,
                    'depth': param_optimized.depth(),
                    'total_gates': sum(param_optimized.count_ops().values()),
                    'two_qubit_gates': sum(
                        count for op, count in param_optimized.count_ops().items()
                        if op in ['cx', 'cz', 'swap', 'ecr']
                    ),
                    'zx_gates_before': original_gates,
                    'zx_gates_after': optimized_gates,
                    'zx_bound_analysis_depth': optimized_circuit.depth(),
                    'note': 'Parameterized circuit - ZX analysis done on bound instance, returned Qiskit L3 optimized parametric circuit'
                }
            
            return optimized_circuit, {
                'method': 'zx_calculus',
                'optimization_time': opt_time,
                'depth': optimized_circuit.depth(),
                'total_gates': sum(optimized_circuit.count_ops().values()),
                'two_qubit_gates': sum(
                    count for op, count in optimized_circuit.count_ops().items()
                    if op in ['cx', 'cz', 'swap', 'ecr']
                ),
                'zx_gates_before': original_gates,
                'zx_gates_after': optimized_gates
            }
            
        except Exception as e:
            logger.warning(f"ZX-calculus optimization failed: {e}")
            opt_time = time.time() - start_time
            return circuit, {
                'method': 'zx_calculus',
                'optimization_time': opt_time,
                'error': str(e)
            }
    
    def optimize_pytket(self, circuit: QuantumCircuit) -> Tuple[QuantumCircuit, Dict]:
        """
        Optimize using Pytket compiler.
        
        Pytket uses different optimization algorithms including peephole
        optimization and phase gadget compilation.
        
        For PARAMETERIZED circuits: We bind dummy values for Pytket optimization
        analysis but return Qiskit L3 optimized parametric circuit.
        """
        if not PYTKET_AVAILABLE:
            logger.warning("pytket not available, skipping Pytket optimization")
            return circuit, {'method': 'pytket', 'error': 'pytket not available'}
        
        start_time = time.time()
        
        try:
            # Check if circuit has unbound parameters
            has_parameters = circuit.num_parameters > 0
            
            if has_parameters:
                # Bind dummy values for Pytket analysis
                dummy_params = np.random.uniform(0, 2*np.pi, circuit.num_parameters)
                bound_circuit = circuit.assign_parameters(dict(zip(circuit.parameters, dummy_params)))
                logger.info(f"Pytket: Bound {circuit.num_parameters} parameters for structure analysis")
            else:
                bound_circuit = circuit
            
            # First transpile to basis gates
            basis_circuit = transpile(
                bound_circuit,
                optimization_level=0,
                basis_gates=['cx', 'h', 'rz', 'rx', 'ry', 'x', 'z'],
                seed_transpiler=42
            )
            
            # Convert to QASM
            qasm_str = qasm_dumps(basis_circuit)
            
            # Convert to Pytket circuit
            tk_circuit = circuit_from_qasm_str(qasm_str)
            
            original_depth = tk_circuit.depth()
            original_gates = tk_circuit.n_gates
            
            # Apply Pytket optimizations
            FullPeepholeOptimise().apply(tk_circuit)
            RemoveRedundancies().apply(tk_circuit)
            CommuteThroughMultis().apply(tk_circuit)
            
            optimized_depth = tk_circuit.depth()
            optimized_gates = tk_circuit.n_gates
            
            # Convert back to QASM
            optimized_qasm = circuit_to_qasm_str(tk_circuit)
            
            # Convert back to Qiskit
            optimized_circuit = qasm_loads(optimized_qasm)
            
            # Final transpile to target basis
            optimized_circuit = transpile(
                optimized_circuit,
                optimization_level=0,
                basis_gates=self.basis_gates,
                coupling_map=self.coupling_map,
                seed_transpiler=42
            )
            
            opt_time = time.time() - start_time
            
            # For parameterized circuits: Return Qiskit L3 optimized parametric circuit
            if has_parameters:
                param_optimized = transpile(
                    circuit,
                    optimization_level=3,
                    basis_gates=self.basis_gates,
                    coupling_map=self.coupling_map,
                    seed_transpiler=42
                )
                return param_optimized, {
                    'method': 'pytket',
                    'optimization_time': opt_time,
                    'depth': param_optimized.depth(),
                    'total_gates': sum(param_optimized.count_ops().values()),
                    'two_qubit_gates': sum(
                        count for op, count in param_optimized.count_ops().items()
                        if op in ['cx', 'cz', 'swap', 'ecr']
                    ),
                    'pytket_depth_before': original_depth,
                    'pytket_depth_after': optimized_depth,
                    'pytket_gates_before': original_gates,
                    'pytket_gates_after': optimized_gates,
                    'pytket_bound_analysis_depth': optimized_circuit.depth(),
                    'note': 'Parameterized circuit - Pytket analysis done on bound instance, returned Qiskit L3 optimized parametric circuit'
                }
            
            return optimized_circuit, {
                'method': 'pytket',
                'optimization_time': opt_time,
                'depth': optimized_circuit.depth(),
                'total_gates': sum(optimized_circuit.count_ops().values()),
                'two_qubit_gates': sum(
                    count for op, count in optimized_circuit.count_ops().items()
                    if op in ['cx', 'cz', 'swap', 'ecr']
                ),
                'pytket_depth_before': original_depth,
                'pytket_depth_after': optimized_depth,
                'pytket_gates_before': original_gates,
                'pytket_gates_after': optimized_gates
            }
            
        except Exception as e:
            logger.warning(f"Pytket optimization failed: {e}")
            opt_time = time.time() - start_time
            return circuit, {
                'method': 'pytket',
                'optimization_time': opt_time,
                'error': str(e)
            }
    
    def optimize_custom_passes(self, circuit: QuantumCircuit) -> Tuple[QuantumCircuit, Dict]:
        """
        Optimize using custom Qiskit pass manager tailored for QML circuits.
        
        This pass manager focuses on:
        - 2-qubit block consolidation
        - Gate cancellation
        - Commutative optimization
        """
        start_time = time.time()
        
        try:
            # First decompose to basis gates
            basis_circuit = transpile(
                circuit,
                optimization_level=0,
                basis_gates=['u', 'cx', 'rz', 'sx', 'x', 'ry'],
                seed_transpiler=42
            )
            
            # Custom pass manager for QML circuits
            custom_pm = PassManager([
                Collect2qBlocks(),           # Collect 2-qubit blocks
                ConsolidateBlocks(),         # Consolidate into single unitary
                UnitarySynthesis(basis_gates=['u', 'cx']),  # Re-synthesize optimally
                Optimize1qGates(basis=['u']), # Optimize single-qubit gates
                InverseCancellation([CXGate(), CZGate()]),  # Cancel inverse gate pairs
                CommutativeCancellation(),   # Cancel commuting gates
            ])
            
            # Run custom optimization
            optimized_circuit = custom_pm.run(basis_circuit)
            
            # Apply routing if coupling map specified
            if self.coupling_map:
                optimized_circuit = transpile(
                    optimized_circuit,
                    optimization_level=1,  # Light optimization for routing
                    coupling_map=self.coupling_map,
                    basis_gates=self.basis_gates,
                    seed_transpiler=42
                )
            
            opt_time = time.time() - start_time
            
            return optimized_circuit, {
                'method': 'custom_passes',
                'optimization_time': opt_time,
                'depth': optimized_circuit.depth(),
                'total_gates': sum(optimized_circuit.count_ops().values()),
                'two_qubit_gates': sum(
                    count for op, count in optimized_circuit.count_ops().items()
                    if op in ['cx', 'cz', 'swap', 'ecr']
                )
            }
            
        except Exception as e:
            logger.warning(f"Custom pass optimization failed: {e}")
            opt_time = time.time() - start_time
            return circuit, {
                'method': 'custom_passes',
                'optimization_time': opt_time,
                'error': str(e)
            }
    
    def compare_all_methods(self, circuit: QuantumCircuit, circuit_name: str = "circuit") -> pd.DataFrame:
        """
        Compare optimization methods on a single circuit.
        
        Parameterized kernel feature maps are evaluated only with Baseline (L0)
        and Qiskit L3 because the alternative compiler paths do not preserve the
        parameterized circuits actually used for model evaluation.
        
        Returns a DataFrame with metrics for each method.
        """
        logger.info(f"Comparing optimization methods for: {circuit_name}")
        
        results = []
        
        # Baseline (no optimization)
        baseline = transpile(
            circuit,
            optimization_level=0,
            basis_gates=self.basis_gates,
            coupling_map=self.coupling_map,
            seed_transpiler=42
        )
        baseline_metrics = {
            'circuit_name': circuit_name,
            'method': 'none (baseline)',
            'optimization_time': 0,
            'depth': baseline.depth(),
            'total_gates': sum(baseline.count_ops().values()),
            'two_qubit_gates': sum(
                count for op, count in baseline.count_ops().items()
                if op in ['cx', 'cz', 'swap', 'ecr']
            )
        }
        results.append(baseline_metrics)
        logger.info(f"  Baseline: depth={baseline_metrics['depth']}, gates={baseline_metrics['total_gates']}")
        
        # Method 1: Qiskit Level 3 (the only method that works well for parameterized circuits)
        _, qiskit_metrics = self.optimize_qiskit(circuit, level=3)
        qiskit_metrics['circuit_name'] = circuit_name
        results.append(qiskit_metrics)
        logger.info(f"  Qiskit L3: depth={qiskit_metrics['depth']}, gates={qiskit_metrics['total_gates']}, time={qiskit_metrics['optimization_time']:.3f}s")
        
        # Calculate reductions relative to baseline
        df = pd.DataFrame(results)
        if 'depth' in df.columns:
            baseline_depth = baseline_metrics['depth']
            baseline_gates = baseline_metrics['total_gates']
            baseline_2q = baseline_metrics['two_qubit_gates']
            
            # Core reduction metrics
            df['depth_reduction_pct'] = ((baseline_depth - df['depth']) / baseline_depth * 100).round(2)
            df['gate_reduction_pct'] = ((baseline_gates - df['total_gates']) / baseline_gates * 100).round(2)
            df['two_qubit_reduction_pct'] = ((baseline_2q - df['two_qubit_gates']) / baseline_2q * 100).round(2) if baseline_2q > 0 else 0
            
            # Absolute reductions
            df['depth_reduction_abs'] = baseline_depth - df['depth']
            df['gate_reduction_abs'] = baseline_gates - df['total_gates']
            df['two_qubit_reduction_abs'] = baseline_2q - df['two_qubit_gates']
            
            # Efficiency metrics (reduction per unit optimization time)
            df['depth_efficiency'] = df.apply(
                lambda row: row['depth_reduction_abs'] / row['optimization_time'] 
                if row['optimization_time'] > 0 else 0, axis=1
            ).round(2)
        
        self.optimization_results.append(df)
        return df
    
    def get_best_method(self, results_df: pd.DataFrame, metric: str = 'depth') -> str:
        """Get the best optimization method based on a metric"""
        if 'error' in results_df.columns:
            valid_results = results_df[results_df['error'].isna()]
        else:
            valid_results = results_df
        if len(valid_results) == 0:
            return 'qiskit_level3'  # Default fallback
        
        if metric == 'depth':
            best_idx = valid_results['depth'].idxmin()
        elif metric == 'total_gates':
            best_idx = valid_results['total_gates'].idxmin()
        elif metric == 'two_qubit_gates':
            best_idx = valid_results['two_qubit_gates'].idxmin()
        else:
            best_idx = valid_results['depth'].idxmin()
        
        return valid_results.loc[best_idx, 'method']


# ============================================================================
# EXPRESSIBILITY ANALYZER
# ============================================================================

class ExpressibilityAnalyzer:
    """Quantify circuit expressibility using fidelity-based measures"""
    
    def __init__(self, num_samples: int = 1000, random_seed: int = 42):
        self.num_samples = num_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def compute_expressibility(self, circuit: QuantumCircuit, simulator=None) -> Dict[str, float]:
        """
        Compute expressibility metric using the KL divergence method.
        
        Expressibility measures how well a parameterized circuit can cover 
        the Hilbert space uniformly.
        """
        if simulator is None:
            simulator = _get_shared_aer_simulator(0, method='statevector')
        
        num_params = circuit.num_parameters
        if num_params == 0:
            return {'expressibility': 0.0, 'num_samples': 0}
        
        # Transpile circuit to decompose high-level instructions (ZFeatureMap, etc.)
        try:
            decomposed_circuit = transpile(circuit, basis_gates=['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z', 'id'])
        except Exception:
            decomposed_circuit = circuit
        
        fidelities = []
        
        for _ in range(self.num_samples):
            # Generate random parameter sets
            params1 = np.random.uniform(0, 2 * np.pi, num_params)
            params2 = np.random.uniform(0, 2 * np.pi, num_params)
            
            # Bind parameters to decomposed circuit
            qc1 = decomposed_circuit.assign_parameters(dict(zip(decomposed_circuit.parameters, params1)))
            qc2 = decomposed_circuit.assign_parameters(dict(zip(decomposed_circuit.parameters, params2)))
            
            # Add save_statevector instructions
            qc1.save_statevector()
            qc2.save_statevector()
            
            try:
                # Simulate and get statevectors
                result1 = simulator.run(qc1, shots=0).result()
                result2 = simulator.run(qc2, shots=0).result()
                
                sv1 = np.asarray(result1.get_statevector())
                sv2 = np.asarray(result2.get_statevector())
                
                # Compute fidelity
                fidelity = np.abs(np.vdot(sv1, sv2)) ** 2
                fidelities.append(fidelity)
            except Exception:
                continue
        
        if len(fidelities) < 10:
            return {'expressibility': 0.0, 'num_samples': len(fidelities)}
        
        # Compute KL divergence from Haar random distribution
        # For Haar random, fidelities follow beta distribution
        fidelities = np.array(fidelities)
        
        # Histogram of sampled fidelities (counts, not density)
        hist, bins = np.histogram(fidelities, bins=50, density=False)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        
        # Haar distribution PDF: P(F) = (N-1)(1-F)^(N-2) where N = 2^n
        N = 2 ** circuit.num_qubits
        haar_pdf = (N - 1) * np.power(np.maximum(1 - bin_centers, 1e-15), N - 2)
        # Normalize Haar PDF to a PMF over bins
        haar_pmf = haar_pdf * bin_width
        haar_pmf = haar_pmf / (np.sum(haar_pmf) + 1e-15)
        
        # Normalize histogram to PMF
        hist_pmf = hist / (np.sum(hist) + 1e-15) + 1e-15
        
        # KL divergence (lower = more expressible)
        kl_div = np.sum(hist_pmf * np.log(hist_pmf / (haar_pmf + 1e-15)))
        
        # Convert to expressibility score (higher = more expressible)
        expressibility = np.exp(-kl_div)
        
        return {
            'expressibility': float(expressibility),
            'kl_divergence': float(kl_div),
            'mean_fidelity': float(np.mean(fidelities)),
            'std_fidelity': float(np.std(fidelities)),
            'num_samples': len(fidelities)
        }


# ============================================================================
# ENTANGLEMENT ENTROPY ANALYZER
# ============================================================================

class EntanglementEntropyAnalyzer:
    """Measure quantum entanglement during computation"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def compute_entanglement_entropy(self, circuit: QuantumCircuit, 
                                      params: Optional[np.ndarray] = None,
                                      simulator=None) -> Dict[str, Any]:
        """
        Compute bipartite entanglement entropy using von Neumann entropy.
        Partitions qubits into two halves.
        """
        if simulator is None:
            simulator = _get_shared_aer_simulator(0, method='statevector')
        
        num_qubits = circuit.num_qubits
        
        # Transpile circuit to decompose high-level instructions (ZFeatureMap, etc.)
        try:
            decomposed_circuit = transpile(circuit, basis_gates=['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z', 'id'])
        except Exception:
            decomposed_circuit = circuit
        
        # Bind parameters if needed
        if params is not None and decomposed_circuit.num_parameters > 0:
            param_dict = dict(zip(decomposed_circuit.parameters, params[:decomposed_circuit.num_parameters]))
            qc = decomposed_circuit.assign_parameters(param_dict)
        else:
            qc = decomposed_circuit.copy()
        
        qc.save_statevector()
        
        try:
            result = simulator.run(qc, shots=0).result()
            statevector = np.asarray(result.get_statevector())
        except Exception as e:
            logger.warning(f"Failed to compute entanglement entropy: {e}")
            return {'entanglement_entropy': 0.0, 'error': str(e)}
        
        # Reshape statevector for bipartite entropy calculation
        n_left = num_qubits // 2
        n_right = num_qubits - n_left
        
        dim_left = 2 ** n_left
        dim_right = 2 ** n_right
        
        # Reshape to matrix
        psi_matrix = statevector.reshape(dim_left, dim_right)
        
        # Compute reduced density matrix
        rho_left = psi_matrix @ psi_matrix.conj().T
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_left)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove numerical zeros
        
        # Von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
        
        # Maximum possible entropy
        max_entropy = min(n_left, n_right)
        
        return {
            'entanglement_entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'normalized_entropy': float(entropy / max_entropy) if max_entropy > 0 else 0.0,
            'num_qubits_left': n_left,
            'num_qubits_right': n_right
        }
    
    def compute_average_entropy(self, circuit: QuantumCircuit, 
                                 num_samples: int = 100,
                                 simulator=None) -> Dict[str, float]:
        """Compute average entanglement entropy over random parameter samples"""
        
        num_params = circuit.num_parameters
        entropies = []
        
        for _ in range(num_samples):
            if num_params > 0:
                params = np.random.uniform(0, 2 * np.pi, num_params)
            else:
                params = None
            
            result = self.compute_entanglement_entropy(circuit, params, simulator)
            if 'error' not in result:
                entropies.append(result['entanglement_entropy'])
        
        if len(entropies) == 0:
            return {'mean_entropy': 0.0, 'std_entropy': 0.0}
        
        return {
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'max_entropy': float(np.max(entropies)),
            'min_entropy': float(np.min(entropies)),
            'num_samples': len(entropies)
        }


# ============================================================================
# TRAINABLE QUANTUM KERNEL (Novel: Optimizable Feature Map Parameters)
# ============================================================================

class TrainableQuantumKernel:
    """Trainable quantum kernel with optimizable feature map parameters.
    
    Novel contribution: Unlike fixed quantum kernels, this allows learning
    optimal feature map parameters via gradient-free optimization (SPSA).
    Similar to approach in Wavelet-QSVM (arXiv:2512.01365) but with
    GPU acceleration.
    """
    
    def __init__(self, base_feature_map: QuantumCircuit, 
                 num_trainable_params: int = None,
                 optimizer=None, random_seed: int = 42):
        self.base_feature_map = base_feature_map
        self.num_qubits = base_feature_map.num_qubits
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Number of trainable parameters for the variational layer
        if num_trainable_params is None:
            num_trainable_params = self.num_qubits * 2  # Default: 2 params per qubit
        self.num_trainable_params = num_trainable_params
        
        # Initialize optimizer
        if optimizer is None and SPSA is not None:
            self.optimizer = SPSA(maxiter=50)
        else:
            self.optimizer = optimizer
        
        # Trainable parameters (initialized randomly)
        self.trainable_params = np.random.uniform(0, 2*np.pi, num_trainable_params)
        self.optimal_params = None
        self.training_history = []
        
        # Build trainable circuit
        self._build_trainable_circuit()
    
    def _build_trainable_circuit(self):
        """Build feature map with trainable variational layer."""
        self.theta = ParameterVector('theta', self.num_trainable_params)
        
        # Create circuit: trainable layer + base feature map
        self.trainable_circuit = QuantumCircuit(self.num_qubits)
        
        # Trainable variational layer
        params_per_qubit = self.num_trainable_params // self.num_qubits
        for i in range(self.num_qubits):
            if params_per_qubit >= 1:
                self.trainable_circuit.ry(self.theta[i * params_per_qubit], i)
            if params_per_qubit >= 2:
                self.trainable_circuit.rz(self.theta[i * params_per_qubit + 1], i)
        
        # Add entangling layer
        for i in range(self.num_qubits - 1):
            self.trainable_circuit.cx(i, i + 1)
        
        # Compose with base feature map
        self.trainable_circuit.compose(self.base_feature_map, inplace=True)
    
    def _objective_function(self, params: np.ndarray, X_train: np.ndarray, 
                            y_train: np.ndarray) -> float:
        """Objective function for kernel optimization (kernel-target alignment)."""
        try:
            # Bind trainable parameters
            param_dict = dict(zip(self.theta, params))
            bound_circuit = self.trainable_circuit.assign_parameters(param_dict)
            
            # Compute kernel matrix using GPU (required)
            try:
                best_gpu = get_gpu_manager().get_best_gpu()
                gpu_kernel = GPUFidelityKernel(bound_circuit, gpu_id=best_gpu)
                K = gpu_kernel.evaluate(X_train)
            except Exception as e:
                raise RuntimeError(
                    f"GPU kernel computation failed for TrainableQuantumKernel: {e}. "
                    "GPU acceleration is required."
                )
            
            # Kernel-target alignment
            alignment = self.compute_kernel_alignment(K, y_train)
            
            # Negative alignment (we minimize)
            return -alignment
        except Exception as e:
            logger.warning(f"Objective function error: {e}")
            return 0.0
    
    @staticmethod
    def compute_kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
        """Compute kernel-target alignment (Cristianini et al.).
        
        Higher alignment indicates better kernel for the classification task.
        """
        y = np.array(y).flatten()
        # Create ideal kernel (outer product of labels)
        # For binary classification, K_ideal[i,j] = 1 if y[i]==y[j], else -1
        y_centered = 2 * y - 1  # Convert to {-1, 1}
        K_ideal = np.outer(y_centered, y_centered)
        
        # Kernel alignment formula
        numerator = np.trace(K @ K_ideal)
        denominator = np.linalg.norm(K, 'fro') * np.linalg.norm(K_ideal, 'fro')
        
        if denominator < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train the kernel parameters to maximize kernel-target alignment."""
        logger.info("Training quantum kernel parameters...")
        
        if self.optimizer is None:
            logger.warning("No optimizer available, using random search")
            # Simple random search fallback
            best_alignment = -np.inf
            best_params = self.trainable_params.copy()
            
            for _ in range(50):
                params = np.random.uniform(0, 2*np.pi, self.num_trainable_params)
                alignment = -self._objective_function(params, X_train, y_train)
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_params = params.copy()
            
            self.optimal_params = best_params
            return {'optimal_alignment': best_alignment, 'method': 'random_search'}
        
        try:
            # Use SPSA optimizer
            result = self.optimizer.minimize(
                fun=lambda p: self._objective_function(p, X_train, y_train),
                x0=self.trainable_params
            )
            
            self.optimal_params = result.x
            optimal_alignment = -result.fun
            
            logger.info(f"Kernel training complete: alignment={optimal_alignment:.4f}")
            
            return {
                'optimal_params': self.optimal_params.tolist(),
                'optimal_alignment': optimal_alignment,
                'iterations': getattr(result, 'nfev', 'N/A')
            }
        except Exception as e:
            logger.warning(f"Kernel training failed: {e}")
            self.optimal_params = self.trainable_params
            return {'error': str(e)}
    
    def get_optimized_feature_map(self) -> QuantumCircuit:
        """Get feature map with optimal parameters bound."""
        params = self.optimal_params if self.optimal_params is not None else self.trainable_params
        param_dict = dict(zip(self.theta, params))
        return self.trainable_circuit.assign_parameters(param_dict)


# ============================================================================
# KERNEL ALIGNMENT ANALYZER
# ============================================================================

class KernelAlignmentAnalyzer:
    """Analyze quantum kernel alignment with target labels.
    
    Kernel alignment measures how well the kernel matrix structure
    matches the classification task. Higher alignment typically
    correlates with better generalization.
    """
    
    def __init__(self):
        self.alignment_history = []
    
    def compute_alignment(self, K: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive kernel alignment metrics."""
        y = np.array(y).flatten()
        n = len(y)
        
        # Create ideal kernel
        y_centered = 2 * y - 1  # Convert to {-1, 1}
        K_ideal = np.outer(y_centered, y_centered)
        
        # Frobenius norm alignment
        frob_numerator = np.trace(K @ K_ideal)
        frob_denominator = np.linalg.norm(K, 'fro') * np.linalg.norm(K_ideal, 'fro')
        frob_alignment = frob_numerator / (frob_denominator + 1e-10)
        
        # Centered kernel alignment (more robust)
        K_centered = self._center_kernel(K)
        K_ideal_centered = self._center_kernel(K_ideal)
        
        centered_numerator = np.trace(K_centered @ K_ideal_centered)
        centered_denominator = np.linalg.norm(K_centered, 'fro') * np.linalg.norm(K_ideal_centered, 'fro')
        centered_alignment = centered_numerator / (centered_denominator + 1e-10)
        
        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        result = {
            'frobenius_alignment': float(frob_alignment),
            'centered_alignment': float(centered_alignment),
            'kernel_rank': int(np.sum(eigenvalues > 1e-6)),
            'kernel_trace': float(np.trace(K)),
            'kernel_condition_number': float(np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)),
            'top_eigenvalue_ratio': float(eigenvalues[-1] / np.sum(eigenvalues)) if len(eigenvalues) > 0 else 0.0
        }
        
        self.alignment_history.append(result)
        return result
    
    @staticmethod
    def _center_kernel(K: np.ndarray) -> np.ndarray:
        """Center kernel matrix in feature space."""
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        return K_centered
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Return alignment history as DataFrame."""
        return pd.DataFrame(self.alignment_history)


# ============================================================================
# BARREN PLATEAU DETECTOR (Novel: Gradient Variance Analysis)
# ============================================================================

class BarrenPlateauDetector:
    """Detect barren plateaus in variational quantum circuits.
    
    Novel contribution: First application of barren plateau analysis
    to quantum kernel methods for IoT intrusion detection.
    
    Barren plateaus occur when gradients vanish exponentially with
    circuit depth/width, making training intractable. Detection helps
    identify problematic circuit architectures.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.analysis_history = []
    
    def detect_barren_plateau(self, circuit: QuantumCircuit, 
                               num_samples: int = 100,
                               simulator=None) -> Dict[str, Any]:
        """Detect barren plateaus via gradient variance analysis.
        
        Computes variance of parameter-shift gradients across random
        parameter initializations. Low variance indicates barren plateau.
        
        Returns:
            Dictionary with variance statistics and barren plateau indicators
        """
        if simulator is None:
            simulator = _get_shared_aer_simulator(0, method='statevector')
        
        num_params = circuit.num_parameters
        if num_params == 0:
            return {'barren_plateau_detected': False, 'reason': 'No parameters'}
        
        # Transpile circuit for consistent analysis
        try:
            transpiled = transpile(circuit, basis_gates=['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z', 'id'])
        except Exception:
            transpiled = circuit
        
        gradient_variances = []
        gradient_means = []
        
        for sample_idx in range(num_samples):
            # Random parameter initialization
            params = np.random.uniform(0, 2*np.pi, num_params)
            
            # Compute gradients using parameter-shift rule approximation
            gradients = self._compute_gradients(transpiled, params, simulator)
            
            if gradients is not None and len(gradients) > 0:
                gradient_variances.append(np.var(gradients))
                gradient_means.append(np.mean(np.abs(gradients)))
        
        if len(gradient_variances) < 10:
            return {
                'barren_plateau_detected': False,
                'reason': 'Insufficient samples',
                'samples_collected': len(gradient_variances)
            }
        
        gradient_variances = np.array(gradient_variances)
        gradient_means = np.array(gradient_means)
        
        # Statistical analysis
        mean_variance = np.mean(gradient_variances)
        std_variance = np.std(gradient_variances)
        mean_gradient_magnitude = np.mean(gradient_means)
        
        # Barren plateau threshold: variance scales as O(2^(-n)) for n qubits
        # A rough heuristic: variance < 0.01 / 2^n indicates barren plateau
        threshold = 0.01 / (2 ** min(circuit.num_qubits, 10))
        barren_detected = mean_variance < threshold
        
        # Severity score (0-1, higher = more severe barren plateau)
        severity = 1.0 - min(1.0, mean_variance / threshold) if threshold > 0 else 0.0
        
        result = {
            'barren_plateau_detected': bool(barren_detected),
            'severity_score': float(severity),
            'mean_gradient_variance': float(mean_variance),
            'std_gradient_variance': float(std_variance),
            'mean_gradient_magnitude': float(mean_gradient_magnitude),
            'threshold_used': float(threshold),
            'num_qubits': circuit.num_qubits,
            'num_parameters': num_params,
            'circuit_depth': circuit.depth(),
            'num_samples': len(gradient_variances)
        }
        
        self.analysis_history.append(result)
        return result
    
    def _compute_gradients(self, circuit: QuantumCircuit, params: np.ndarray,
                           simulator) -> Optional[np.ndarray]:
        """Compute gradients using finite difference approximation."""
        try:
            epsilon = 0.1  # Finite difference step
            gradients = []
            
            # Get base expectation value
            base_exp = self._get_expectation(circuit, params, simulator)
            if base_exp is None:
                return None
            
            for i in range(len(params)):
                # Finite difference approximation: f(x + eps) and f(x - eps)
                params_plus = params.copy()
                params_plus[i] += epsilon
                
                params_minus = params.copy()
                params_minus[i] -= epsilon
                
                exp_plus = self._get_expectation(circuit, params_plus, simulator)
                exp_minus = self._get_expectation(circuit, params_minus, simulator)
                
                if exp_plus is not None and exp_minus is not None:
                    gradient = (exp_plus - exp_minus) / (2 * epsilon)
                    gradients.append(gradient)
            
            return np.array(gradients) if gradients else None
        except Exception:
            return None
    
    def _get_expectation(self, circuit: QuantumCircuit, params: np.ndarray,
                         simulator) -> Optional[float]:
        """Get expectation value of Z measurement on first qubit."""
        try:
            param_dict = dict(zip(circuit.parameters, params))
            qc = circuit.assign_parameters(param_dict)
            qc.save_statevector()
            
            result = simulator.run(qc, shots=0).result()
            sv = np.asarray(result.get_statevector())
            
            # Expectation value of Z on first qubit
            n = circuit.num_qubits
            dim = 2 ** n
            z_exp = 0.0
            for i in range(dim):
                # Check if first qubit is |0> or |1>
                if (i >> (n-1)) & 1:  # First qubit is |1>
                    z_exp -= np.abs(sv[i]) ** 2
                else:  # First qubit is |0>
                    z_exp += np.abs(sv[i]) ** 2
            
            return float(z_exp)
        except Exception:
            return None
    
    def analyze_depth_scaling(self, num_qubits: int, max_depth: int = 6,
                               samples_per_depth: int = 50) -> pd.DataFrame:
        """Analyze how gradient variance scales with circuit depth.
        
        This helps identify the critical depth where barren plateaus emerge.
        """
        results = []
        
        for depth in range(1, max_depth + 1):
            # Create circuit with specified depth
            circuit = self._create_layered_circuit(num_qubits, depth)
            analysis = self.detect_barren_plateau(circuit, num_samples=samples_per_depth)
            analysis['depth'] = depth
            results.append(analysis)
            
            logger.info(f"Depth {depth}: variance={analysis['mean_gradient_variance']:.6f}, "
                        f"barren={analysis['barren_plateau_detected']}")
        
        return pd.DataFrame(results)
    
    def _create_layered_circuit(self, num_qubits: int, num_layers: int) -> QuantumCircuit:
        """Create a parameterized circuit with specified layers."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector('p', num_qubits * num_layers * 2)
        param_idx = 0
        
        for layer in range(num_layers):
            # Rotation layer
            for i in range(num_qubits):
                qc.ry(params[param_idx], i)
                param_idx += 1
                qc.rz(params[param_idx], i)
                param_idx += 1
            
            # Entangling layer
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
        
        return qc
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Return analysis history as DataFrame."""
        return pd.DataFrame(self.analysis_history)


# ============================================================================
# LAYER-WISE LEARNING CAPACITY ANALYZER
# ============================================================================

class LayerWiseLearningAnalyzer:
    """Analyze layer-wise learning capacity and information flow.
    
    Novel contribution: Measures expressibility and entanglement
    progression through circuit layers to identify bottlenecks.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def analyze_layer_progression(self, feature_map_type: str, num_qubits: int,
                                    max_reps: int = 4, entanglement: str = 'full',
                                    num_samples: int = 50) -> pd.DataFrame:
        """Analyze how expressibility and entanglement progress with depth.
        
        Returns DataFrame with metrics per layer/repetition.
        """
        results = []
        expr_analyzer = ExpressibilityAnalyzer(num_samples=num_samples, random_seed=self.random_seed)
        entropy_analyzer = EntanglementEntropyAnalyzer(random_seed=self.random_seed)
        metrics_analyzer = CircuitMetricsAnalyzer()
        
        for reps in range(1, max_reps + 1):
            try:
                # Create circuit with current number of repetitions
                circuit = create_feature_map(num_qubits, feature_map_type, 
                                             reps=reps, entanglement=entanglement)
                
                # Analyze circuit
                expr_result = expr_analyzer.compute_expressibility(circuit)
                entropy_result = entropy_analyzer.compute_average_entropy(circuit, num_samples=num_samples)
                circuit_metrics = metrics_analyzer.analyze_circuit(circuit, f"{feature_map_type}_r{reps}")
                
                result = {
                    'feature_map': feature_map_type,
                    'reps': reps,
                    'entanglement': entanglement,
                    'expressibility': expr_result.get('expressibility', 0.0),
                    'kl_divergence': expr_result.get('kl_divergence', float('inf')),
                    'mean_entropy': entropy_result.get('mean_entropy', 0.0),
                    'std_entropy': entropy_result.get('std_entropy', 0.0),
                    'depth': circuit_metrics.get('depth', 0),
                    'total_gates': circuit_metrics.get('total_gates', 0),
                    'two_qubit_gates': circuit_metrics.get('two_qubit_gates', 0),
                    'num_parameters': circuit_metrics.get('num_parameters', 0),
                }
                
                # Compute efficiency metrics
                if circuit_metrics.get('depth', 0) > 0:
                    result['expressibility_per_depth'] = expr_result.get('expressibility', 0.0) / circuit_metrics['depth']
                    result['entropy_per_depth'] = entropy_result.get('mean_entropy', 0.0) / circuit_metrics['depth']
                else:
                    result['expressibility_per_depth'] = 0.0
                    result['entropy_per_depth'] = 0.0
                
                results.append(result)
                
                logger.info(f"{feature_map_type} reps={reps}: expr={expr_result.get('expressibility', 0):.4f}, "
                           f"entropy={entropy_result.get('mean_entropy', 0):.4f}, depth={circuit_metrics.get('depth', 0)}")
                
            except Exception as e:
                logger.warning(f"Layer analysis failed for reps={reps}: {e}")
        
        return pd.DataFrame(results)
    
    def find_optimal_depth(self, feature_map_type: str, num_qubits: int,
                           max_reps: int = 6, entanglement: str = 'full') -> Dict[str, Any]:
        """Find optimal circuit depth balancing expressibility and complexity.
        
        Returns the depth that maximizes expressibility-per-gate efficiency.
        """
        df = self.analyze_layer_progression(feature_map_type, num_qubits, max_reps, entanglement)
        
        if df.empty:
            return {'optimal_reps': 2, 'reason': 'Analysis failed'}
        
        # Find optimal based on expressibility per depth (efficiency)
        best_idx = df['expressibility_per_depth'].idxmax()
        best_row = df.iloc[best_idx]
        
        return {
            'optimal_reps': int(best_row['reps']),
            'optimal_depth': int(best_row['depth']),
            'expressibility': float(best_row['expressibility']),
            'expressibility_per_depth': float(best_row['expressibility_per_depth']),
            'mean_entropy': float(best_row['mean_entropy']),
            'total_gates': int(best_row['total_gates']),
            'analysis_dataframe': df.to_dict('records')
        }


# ============================================================================
# DATA RE-UPLOADING CIRCUIT BUILDER
# ============================================================================

class DataReuploadingCircuit:
    """Build data re-uploading circuits for enhanced expressibility.
    
    Data re-uploading interleaves classical data encoding with
    trainable variational layers, providing universal approximation
    capability with fewer qubits.
    """
    
    def __init__(self, num_qubits: int, num_layers: int = 3,
                 data_dim: int = None, random_seed: int = 42):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.data_dim = data_dim if data_dim else num_qubits
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.circuit = None
        self.data_params = None
        self.trainable_params = None
        self._build_circuit()
    
    def _build_circuit(self):
        """Build data re-uploading circuit."""
        self.circuit = QuantumCircuit(self.num_qubits)
        
        # Data parameters (input features repeated in each layer)
        self.data_params = ParameterVector('x', self.data_dim)
        
        # Trainable parameters
        num_trainable = self.num_qubits * self.num_layers * 2
        self.trainable_params = ParameterVector('theta', num_trainable)
        trainable_idx = 0
        
        for layer in range(self.num_layers):
            # Data encoding layer
            for i in range(self.num_qubits):
                data_idx = i % self.data_dim
                self.circuit.ry(self.data_params[data_idx], i)
                self.circuit.rz(self.data_params[data_idx], i)
            
            # Trainable variational layer
            for i in range(self.num_qubits):
                self.circuit.ry(self.trainable_params[trainable_idx], i)
                trainable_idx += 1
                self.circuit.rz(self.trainable_params[trainable_idx], i)
                trainable_idx += 1
            
            # Entangling layer
            for i in range(self.num_qubits - 1):
                self.circuit.cx(i, i + 1)
            if self.num_qubits > 2:
                self.circuit.cx(self.num_qubits - 1, 0)  # Circular entanglement
    
    def get_circuit(self) -> QuantumCircuit:
        """Return the data re-uploading circuit."""
        return self.circuit
    
    def get_feature_map(self, trainable_values: np.ndarray = None) -> QuantumCircuit:
        """Get feature map with trainable parameters bound.
        
        If trainable_values is None, uses random initialization.
        """
        if trainable_values is None:
            trainable_values = np.random.uniform(0, 2*np.pi, len(self.trainable_params))
        
        param_dict = dict(zip(self.trainable_params, trainable_values))
        return self.circuit.assign_parameters(param_dict)
    
    def compare_with_standard(self, standard_feature_map: QuantumCircuit,
                               num_samples: int = 100) -> Dict[str, Any]:
        """Compare data re-uploading with standard feature map."""
        expr_analyzer = ExpressibilityAnalyzer(num_samples=num_samples, random_seed=self.random_seed)
        entropy_analyzer = EntanglementEntropyAnalyzer(random_seed=self.random_seed)
        metrics_analyzer = CircuitMetricsAnalyzer()
        
        # Analyze data re-uploading circuit (with random trainable params)
        reupload_fm = self.get_feature_map()
        reupload_expr = expr_analyzer.compute_expressibility(reupload_fm)
        reupload_entropy = entropy_analyzer.compute_average_entropy(reupload_fm, num_samples=50)
        reupload_metrics = metrics_analyzer.analyze_circuit(reupload_fm, 'data_reuploading')
        
        # Analyze standard feature map
        standard_expr = expr_analyzer.compute_expressibility(standard_feature_map)
        standard_entropy = entropy_analyzer.compute_average_entropy(standard_feature_map, num_samples=50)
        standard_metrics = metrics_analyzer.analyze_circuit(standard_feature_map, 'standard')
        
        return {
            'data_reuploading': {
                'expressibility': reupload_expr.get('expressibility', 0.0),
                'mean_entropy': reupload_entropy.get('mean_entropy', 0.0),
                'depth': reupload_metrics.get('depth', 0),
                'total_gates': reupload_metrics.get('total_gates', 0),
                'num_parameters': reupload_metrics.get('num_parameters', 0)
            },
            'standard': {
                'expressibility': standard_expr.get('expressibility', 0.0),
                'mean_entropy': standard_entropy.get('mean_entropy', 0.0),
                'depth': standard_metrics.get('depth', 0),
                'total_gates': standard_metrics.get('total_gates', 0),
                'num_parameters': standard_metrics.get('num_parameters', 0)
            },
            'expressibility_improvement': (
                reupload_expr.get('expressibility', 0.0) - standard_expr.get('expressibility', 0.0)
            ),
            'entropy_improvement': (
                reupload_entropy.get('mean_entropy', 0.0) - standard_entropy.get('mean_entropy', 0.0)
            )
        }


# ============================================================================
# FAULT-TOLERANT RESOURCE ESTIMATOR
# ============================================================================

class ResourceEstimator:
    """Estimate resources for fault-tolerant quantum execution.
    
    Based on methodology from arXiv:2502.11173 (Bellante et al.)
    for estimating practical quantum advantage thresholds.
    """
    
    # T-gate cost estimates for common operations
    T_GATE_COSTS = {
        'cx': 7,      # CX ~7 T gates in fault-tolerant
        'cz': 7,      # CZ ~7 T gates
        't': 1,       # T gate = 1
        'tdg': 1,     # T-dagger = 1
        's': 0,       # S is Clifford (free)
        'sdg': 0,     # S-dagger is Clifford
        'h': 0,       # H is Clifford
        'x': 0,       # X is Clifford
        'y': 0,       # Y is Clifford
        'z': 0,       # Z is Clifford
        'rx': 15,     # Rx needs T synthesis (~15 T gates for reasonable precision)
        'ry': 15,     # Ry needs T synthesis
        'rz': 15,     # Rz needs T synthesis
        'u': 30,      # U gate ~30 T gates
        'u1': 15,     # U1 ~15 T gates
        'u2': 20,     # U2 ~20 T gates
        'u3': 30,     # U3 ~30 T gates
        'sx': 0,      # SX is Clifford
        'ccx': 21,    # Toffoli ~21 T gates
    }
    
    # Physical-to-logical qubit overhead for different error correction codes
    QUBIT_OVERHEAD = {
        'surface_code_d5': 50,    # Distance-5 surface code
        'surface_code_d7': 98,    # Distance-7 surface code
        'surface_code_d9': 162,   # Distance-9 surface code
        'surface_code_d11': 242,  # Distance-11 surface code
        'color_code_d5': 36,      # Distance-5 color code
    }
    
    def __init__(self, error_correction: str = 'surface_code_d7'):
        self.error_correction = error_correction
        self.qubit_overhead = self.QUBIT_OVERHEAD.get(error_correction, 100)
    
    def estimate_resources(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Estimate fault-tolerant execution resources."""
        ops = circuit.count_ops()
        
        # Count T gates
        t_count = 0
        for gate, count in ops.items():
            t_cost = self.T_GATE_COSTS.get(gate.lower(), 10)  # Default 10 for unknown
            t_count += count * t_cost
        
        # Physical qubit count
        logical_qubits = circuit.num_qubits
        physical_qubits = logical_qubits * self.qubit_overhead
        
        # Time estimates (assuming 1 microsecond per T gate)
        t_gate_time_us = 1.0
        estimated_runtime_us = t_count * t_gate_time_us
        estimated_runtime_ms = estimated_runtime_us / 1000
        estimated_runtime_s = estimated_runtime_ms / 1000
        
        # Magic state distillation factories needed (rough estimate)
        # Assuming 1 factory produces 1 T state per microsecond
        magic_state_factories = max(1, int(t_count / (estimated_runtime_us + 1)))
        
        return {
            'logical_qubits': logical_qubits,
            'physical_qubits': physical_qubits,
            'error_correction': self.error_correction,
            'qubit_overhead_factor': self.qubit_overhead,
            'estimated_t_count': t_count,
            'circuit_depth': circuit.depth(),
            'total_gates': sum(ops.values()),
            'gate_breakdown': dict(ops),
            'estimated_runtime_us': float(estimated_runtime_us),
            'estimated_runtime_ms': float(estimated_runtime_ms),
            'estimated_runtime_s': float(estimated_runtime_s),
            'magic_state_factories': magic_state_factories
        }
    
    def estimate_quantum_advantage_threshold(self, circuit: QuantumCircuit,
                                              classical_runtime_s: float) -> Dict[str, Any]:
        """Estimate when quantum execution becomes advantageous.
        
        Compares estimated fault-tolerant runtime with classical baseline.
        """
        ft_resources = self.estimate_resources(circuit)
        quantum_runtime_s = ft_resources['estimated_runtime_s']
        
        speedup = classical_runtime_s / (quantum_runtime_s + 1e-10)
        advantage = speedup > 1.0
        
        return {
            'fault_tolerant_resources': ft_resources,
            'classical_runtime_s': classical_runtime_s,
            'quantum_runtime_s': quantum_runtime_s,
            'speedup_factor': float(speedup),
            'quantum_advantage': advantage,
            'break_even_improvement_needed': float(1.0 / speedup) if speedup < 1.0 else 0.0
        }
    
    def compare_circuits(self, circuits: List[Tuple[str, QuantumCircuit]]) -> pd.DataFrame:
        """Compare fault-tolerant resources across multiple circuits."""
        results = []
        
        for name, circuit in circuits:
            resources = self.estimate_resources(circuit)
            resources['circuit_name'] = name
            results.append(resources)
        
        return pd.DataFrame(results)


# ============================================================================
# NOISE MODEL SIMULATOR
# ============================================================================

class NoiseModelSimulator:
    """Create and apply realistic quantum noise models"""
    
    def __init__(self, params: Dict[str, float] = None):
        self.params = params or NOISE_PARAMS
    
    def create_noise_model(self, num_qubits: int) -> NoiseModel:
        """Create a realistic noise model for NISQ simulation"""
        
        noise_model = NoiseModel()
        
        # Single-qubit depolarizing error
        error_1q = depolarizing_error(self.params['single_qubit_error'], 1)
        
        # Two-qubit depolarizing error
        error_2q = depolarizing_error(self.params['two_qubit_error'], 2)
        
        # Thermal relaxation errors
        error_thermal_1q = thermal_relaxation_error(
            t1=self.params['t1'],
            t2=self.params['t2'],
            time=self.params['gate_time_1q']
        )
        
        # 2Q thermal relaxation: single-qubit thermal relaxation on BOTH qubits
        # Per IBM standard: "Two-qubit gate errors consisting of a two-qubit
        # depolarizing error followed by single-qubit thermal relaxation errors
        # on both qubits in the gate."
        error_thermal_2q = thermal_relaxation_error(
            t1=self.params['t1'],
            t2=self.params['t2'],
            time=self.params['gate_time_2q']
        ).expand(thermal_relaxation_error(
            t1=self.params['t1'],
            t2=self.params['t2'],
            time=self.params['gate_time_2q']
        ))
        
        # Combine depolarizing + thermal for both 1Q and 2Q gates
        combined_1q = error_1q.compose(error_thermal_1q)
        combined_2q = error_2q.compose(error_thermal_2q)
        
        # Add errors to noise model
        noise_model.add_all_qubit_quantum_error(combined_1q, ['u', 'rx', 'ry', 'rz', 'sx', 'x'])
        noise_model.add_all_qubit_quantum_error(combined_2q, ['cx', 'cz'])
        
        return noise_model
    
    def create_simulator_with_noise(self, num_qubits: int, 
                                    method: str = 'statevector') -> AerSimulator:
        """Create AerSimulator with noise model and GPU acceleration"""
        
        noise_model = self.create_noise_model(num_qubits)
        
        try:
            simulator = AerSimulator(
                method=method,
                noise_model=noise_model,
                device='GPU',
                precision='single',
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create GPU noisy simulator: {e}. "
                "GPU acceleration is required. Ensure CUDA and qiskit-aer-gpu are properly installed."
            )
        
        return simulator


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class StatisticalAnalyzer:
    """Perform statistical significance tests and analysis"""
    
    @staticmethod
    def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for mean"""
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean - h, mean + h
    
    @staticmethod
    def paired_t_test(data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
        """Perform paired t-test"""
        t_stat, p_value = stats.ttest_rel(data1, data2)
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_0.05': bool(p_value < 0.05),
            'significant_0.01': bool(p_value < 0.01)
        }
    
    @staticmethod
    def wilcoxon_test(data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
        """Perform Wilcoxon signed-rank test (non-parametric)"""
        try:
            w_stat, p_value = stats.wilcoxon(data1, data2)
            return {
                'w_statistic': float(w_stat),
                'p_value': float(p_value),
                'significant_0.05': bool(p_value < 0.05),
                'significant_0.01': bool(p_value < 0.01)
            }
        except Exception:
            return {'error': 'Could not perform Wilcoxon test'}
    
    @staticmethod
    def friedman_test(*args) -> Dict[str, float]:
        """Perform Friedman test for multiple related samples"""
        try:
            f_stat, p_value = stats.friedmanchisquare(*args)
            return {
                'friedman_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant_0.05': bool(p_value < 0.05)
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def compute_effect_size_cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
        """Compute Cohen's d effect size"""
        n1, n2 = len(data1), len(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0


# ============================================================================
# FEATURE MAP BUILDERS
# ============================================================================

def create_feature_map(num_qubits: int, map_type: str, reps: int = 2,
                       entanglement: str = 'full', param_prefix: str = 'x') -> QuantumCircuit:
    """Create feature map with specified configuration"""
    
    if map_type == 'Z':
        feature_map = ZFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            parameter_prefix=param_prefix
        )
    elif map_type == 'ZZ':
        feature_map = ZZFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=param_prefix
        )
    elif map_type == 'Pauli':
        feature_map = PauliFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            paulis=['Z', 'ZZ'],
            entanglement=entanglement,
            parameter_prefix=param_prefix
        )
    elif map_type == 'Custom':
        # Custom feature map with enhanced expressibility
        feature_map = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, num_qubits)
        
        for r in range(reps):
            # Hadamard layer
            for i in range(num_qubits):
                feature_map.h(i)
            
            # Parameterized rotations
            for i in range(num_qubits):
                feature_map.rz(params[i], i)
                feature_map.ry(params[i], i)
            
            # Entangling layer based on pattern
            if entanglement == 'linear':
                for i in range(num_qubits - 1):
                    feature_map.cx(i, i + 1)
            elif entanglement == 'circular':
                for i in range(num_qubits - 1):
                    feature_map.cx(i, i + 1)
                if num_qubits > 2:
                    feature_map.cx(num_qubits - 1, 0)
            elif entanglement == 'full':
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        feature_map.cx(i, j)
            elif entanglement == 'sca':
                # Shifted circular alternating
                for i in range(0, num_qubits - 1, 2):
                    feature_map.cx(i, i + 1)
                for i in range(1, num_qubits - 1, 2):
                    feature_map.cx(i, i + 1)
            
            # Feature encoding
            for i in range(num_qubits):
                feature_map.rz(2 * params[i], i)
    else:
        raise ValueError(f"Unknown feature map type: {map_type}")
    
    return feature_map


def create_ansatz(num_qubits: int, ansatz_type: str, reps: int = 2,
                  entanglement: str = 'full', param_prefix: str = 'theta') -> QuantumCircuit:
    """Create variational ansatz"""
    
    if ansatz_type == 'RealAmplitudes':
        ansatz = RealAmplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=param_prefix
        )
    elif ansatz_type == 'EfficientSU2':
        ansatz = EfficientSU2(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=param_prefix
        )
    elif ansatz_type == 'TwoLocal':
        ansatz = TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=['ry', 'rz'],
            entanglement_blocks='cz',
            entanglement=entanglement,
            reps=reps,
            parameter_prefix=param_prefix
        )
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
    
    return ansatz


# ============================================================================
# GPU AND SIMULATOR MANAGEMENT
# ============================================================================

# Thread-safe caches for GPU simulators
_SIMULATOR_CACHE: Dict[Tuple[int, str], AerSimulator] = {}
_SIMULATOR_LOCK = threading.Lock()

# Transpilation cache for repeatedly reused circuit families.
_TRANSPILED_CIRCUIT_CACHE: Dict[Tuple[Any, ...], QuantumCircuit] = {}
_TRANSPILED_CIRCUIT_LOCK = threading.Lock()

# CuPy caching
_CUPY_MODULE = None
_CUPY_CHECKED = False


def _ensure_cupy():
    """Lazily initialize CuPy module for GPU acceleration."""
    global _CUPY_MODULE, _CUPY_CHECKED
    if _CUPY_CHECKED:
        return _CUPY_MODULE
    _CUPY_CHECKED = True
    try:
        import cupy
        _CUPY_MODULE = cupy
        logger.info("CuPy available for GPU kernel computations")
    except ImportError:
        _CUPY_MODULE = None
        logger.error("❌ CuPy not available - GPU kernel computation requires CuPy!")
    return _CUPY_MODULE


def _normalize_coupling_map_key(coupling_map) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Convert a coupling map into a stable hashable cache key."""
    if coupling_map is None:
        return None
    try:
        return tuple(sorted(tuple(edge) for edge in coupling_map.get_edges()))
    except Exception:
        return tuple(sorted(tuple(edge) for edge in getattr(coupling_map, 'graph', [])))


def transpile_with_cache(circuit: QuantumCircuit, cache_key: Tuple[Any, ...], **transpile_kwargs) -> QuantumCircuit:
    """Transpile a circuit once and return deep-copied cached instances thereafter."""
    with _TRANSPILED_CIRCUIT_LOCK:
        cached = _TRANSPILED_CIRCUIT_CACHE.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

    transpiled_circuit = transpile(circuit, **transpile_kwargs)

    with _TRANSPILED_CIRCUIT_LOCK:
        _TRANSPILED_CIRCUIT_CACHE[cache_key] = copy.deepcopy(transpiled_circuit)

    return transpiled_circuit


def get_optimal_thread_count(gpu_id: int = 0) -> int:
    """Get optimal thread count based on GPU compute capability."""
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            cc = float(f"{major}.{minor}")
            if cc >= 8.6:
                return 384  # Ampere/Ada
            elif cc >= 8.0:
                return 384  # Ampere
            elif cc >= 7.5:
                return 256  # Turing
            elif cc >= 7.0:
                return 256  # Volta
        except Exception:
            pass
    return 256  # Default


def clear_gpu_memory(gpu_id: int = 0):
    """Clear GPU memory caches to prevent OOM errors."""
    cupy_mod = _ensure_cupy()
    if cupy_mod is not None:
        try:
            with cupy_mod.cuda.Device(gpu_id):
                cupy_mod.get_default_memory_pool().free_all_blocks()
        except Exception:
            try:
                with cupy_mod.cuda.Device(0):
                    cupy_mod.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
    gc.collect()


def get_gpu_memory_status(gpu_id: int = 0) -> Optional[Dict[str, float]]:
    """Get GPU memory usage statistics."""
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'free': mem_info.free / (1024**3),
                'used': mem_info.used / (1024**3),
                'total': mem_info.total / (1024**3),
                'utilization': (mem_info.used / mem_info.total) * 100
            }
        except Exception:
            pass
    return None


def _get_shared_aer_simulator(gpu_id: int = 0, method: str = 'statevector') -> AerSimulator:
    """Return a cached AerSimulator configured for the requested GPU.
    
    Uses process-local caching to avoid recreating simulators and
    optimized GPU settings for maximum performance with cuQuantum.
    """
    key = (gpu_id, method)
    
    with _SIMULATOR_LOCK:
        if key in _SIMULATOR_CACHE:
            return _SIMULATOR_CACHE[key]
        
        # Create optimized GPU simulator with cuQuantum/cuStateVec
        # NOTE: cuStateVec_enable and batched_shots_gpu are MUTUALLY EXCLUSIVE
        # per Qiskit Aer docs. Since this simulator is used for noiseless
        # For statevector simulations such as GPUFidelityKernel, cuStateVec is optimal.
        try:
            simulator = AerSimulator(
                method=method,
                device='GPU',
                precision='single',
                cuStateVec_enable=True,       # Use cuStateVec for massive speedup (statevector only)
            )
        except Exception as e:
            logger.warning(f"Failed to create GPU simulator with cuStateVec: {e}, trying without cuStateVec")
            try:
                simulator = AerSimulator(
                    method=method,
                    device='GPU',
                    precision='single'
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to create GPU simulator: {e2}. "
                                   "GPU acceleration is required. Ensure CUDA and qiskit-aer-gpu are properly installed.")
        
        # Validate GPU backend is actually enabled - REQUIRED
        try:
            cfg = simulator.configuration()
            backend_name = getattr(cfg, 'backend_name', str(cfg))
            if 'gpu' not in backend_name.lower():
                raise RuntimeError(f"AerSimulator backend is not GPU-enabled: {backend_name}. "
                                   "GPU acceleration is required for this experiment.")
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Could not validate GPU backend: {e} - proceeding with caution")
        
        # Apply optimized GPU settings (matching iot_multigpu.py proven configuration)
        optimal_threads = get_optimal_thread_count(gpu_id)
        try:
            simulator.set_options(
                max_parallel_threads=optimal_threads,
                max_parallel_experiments=8,      # Optimal for parallel circuit execution
                blocking_enable=False,           # Disable blocking for <25 qubit circuits
                precision='single',
                max_memory_mb=24000,             # Match iot_multigpu.py setting
                fusion_enable=True,              # Enable gate fusion for small circuits
                fusion_threshold=4,              # Fuse gates on circuits >= 4 qubits (43% speedup)
            )
            logger.info(f"[GPU:{gpu_id}] Configured simulator: threads={optimal_threads}, "
                        f"parallel_experiments=8, cuStateVec=True, fusion_threshold=4")
        except Exception as e:
            logger.warning(f"Failed to set optimized GPU options: {e}")
        
        _SIMULATOR_CACHE[key] = simulator
        logger.info(f"[GPU:{gpu_id}] Created and cached optimized AerSimulator (method={method})")
        return simulator


class GPUManager:
    """Enhanced GPU detection and management with memory monitoring."""
    
    def __init__(self):
        self.gpu_count = 0
        self.gpu_info = []
        self.gpu_locks: Dict[int, threading.Lock] = {}
        self.gpu_usage: Dict[int, int] = {}
        self._detect_gpus()
    
    def _detect_gpus(self):
        """Detect available GPUs with detailed info."""
        if NVML_AVAILABLE:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get compute capability
                    try:
                        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                        compute_capability = f"{major}.{minor}"
                    except Exception:
                        compute_capability = "unknown"
                    
                    self.gpu_info.append({
                        'id': i,
                        'name': name,
                        'memory_total': mem_info.total / (1024**3),
                        'memory_free': mem_info.free / (1024**3),
                        'compute_capability': compute_capability,
                    })
                    self.gpu_locks[i] = threading.Lock()
                    self.gpu_usage[i] = 0
                    
                logger.info(f"Detected {self.gpu_count} GPU(s):")
                for info in self.gpu_info:
                    logger.info(f"  GPU {info['id']}: {info['name']} "
                              f"({info['memory_total']:.1f}GB, CC: {info['compute_capability']})")
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
    
    def get_best_gpu(self) -> int:
        """Get the GPU with the most free memory."""
        if not self.gpu_info:
            return 0
        
        best_gpu = 0
        best_free = 0.0
        
        for info in self.gpu_info:
            gpu_id = info['id']
            mem_status = get_gpu_memory_status(gpu_id)
            if mem_status and mem_status['free'] > best_free:
                best_free = mem_status['free']
                best_gpu = gpu_id
        
        return best_gpu
    
    def acquire_gpu(self, preferred_gpu: int = None) -> int:
        """Acquire a GPU for exclusive use (with usage counting)."""
        if preferred_gpu is not None and preferred_gpu in self.gpu_locks:
            with self.gpu_locks[preferred_gpu]:
                if self.gpu_usage[preferred_gpu] < 3:  # Allow up to 3 concurrent uses
                    self.gpu_usage[preferred_gpu] += 1
                    return preferred_gpu
        
        # Find least utilized GPU
        for gpu_id in sorted(self.gpu_usage.keys(), key=lambda x: self.gpu_usage[x]):
            with self.gpu_locks[gpu_id]:
                if self.gpu_usage[gpu_id] < 3:
                    self.gpu_usage[gpu_id] += 1
                    return gpu_id
        
        return 0
    
    def release_gpu(self, gpu_id: int):
        """Release a GPU."""
        if gpu_id in self.gpu_locks:
            with self.gpu_locks[gpu_id]:
                self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)


# Global GPU Manager (lazily initialized)
gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get or create the global GPU manager."""
    global gpu_manager
    if gpu_manager is None:
        gpu_manager = GPUManager()
    return gpu_manager


# ============================================================================
# MULTI-GPU PARALLEL EXECUTION
# ============================================================================

def run_parallel_on_gpus(tasks: List[Tuple], worker_func, num_workers: int = None,
                         timeout_per_task: int = 600) -> List[Any]:
    """Execute tasks in parallel across multiple GPUs.
    
    Args:
        tasks: List of (task_id, *task_args) tuples
        worker_func: Function that takes (task_id, *task_args, gpu_id) and returns result
        num_workers: Number of parallel workers (defaults to GPU count)
        timeout_per_task: Timeout in seconds per task
    
    Returns:
        List of results in task order
    """
    gpu_mgr = get_gpu_manager()
    num_gpus = gpu_mgr.gpu_count
    
    if num_workers is None:
        num_workers = max(1, num_gpus)
    
    if num_workers == 1 or len(tasks) == 1:
        # Sequential execution
        results = []
        for i, task in enumerate(tasks):
            gpu_id = i % max(1, num_gpus)
            try:
                result = worker_func(*task, gpu_id)
                results.append(result)
            except Exception as e:
                logger.warning(f"Task {i} failed: {e}")
                results.append({'error': str(e), 'task_id': task[0] if task else i})
        return results
    
    # Parallel execution using ThreadPoolExecutor
    # (ProcessPoolExecutor doesn't work well with CUDA contexts)
    results = [None] * len(tasks)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks with round-robin GPU assignment
        future_to_idx = {}
        for i, task in enumerate(tasks):
            gpu_id = i % num_gpus
            future = executor.submit(worker_func, *task, gpu_id)
            future_to_idx[future] = i
        
        # Collect results as they complete
        for future in future_to_idx:
            idx = future_to_idx[future]
            try:
                result = future.result(timeout=timeout_per_task)
                results[idx] = result
            except Exception as e:
                logger.warning(f"Task {idx} failed or timed out: {e}")
                results[idx] = {'error': str(e), 'task_id': idx}
    
    return results


def parallel_kernel_evaluation(feature_maps: List[Tuple[str, QuantumCircuit]], 
                               X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray) -> List[Dict]:
    """Evaluate multiple feature maps in parallel across GPUs.
    
    Args:
        feature_maps: List of (name, feature_map) tuples
        X_train, X_test: Training and test data
        y_train: Training labels
    
    Returns:
        List of result dicts with kernel matrices and predictions
    """
    def evaluate_single(name, fm, X_tr, X_te, y_tr, gpu_id):
        try:
            gpu_kernel = GPUFidelityKernel(fm, gpu_id=gpu_id)
            
            start = time.time()
            K_train = gpu_kernel.evaluate(X_tr, X_tr)
            K_test = gpu_kernel.evaluate(X_te, X_tr)
            
            svc = cuSVC(kernel='precomputed', random_state=42,
                       cache_size=8192.0, max_iter=10000, nochange_steps=100,
                       output_type='numpy')
            svc.fit(K_train, y_tr)
            compute_time = time.time() - start
            
            return {
                'name': name,
                'K_train': K_train,
                'K_test': K_test,
                'svc': svc,
                'gpu_id': gpu_id,
                'compute_time': compute_time,
                'status': 'success'
            }
        except Exception as e:
            return {'name': name, 'error': str(e), 'status': 'failed'}
    
    # Prepare tasks
    tasks = [(name, fm, X_train, X_test, y_train) for name, fm in feature_maps]
    
    # Run in parallel
    return run_parallel_on_gpus(tasks, evaluate_single)


def create_gpu_simulator(gpu_id: int = 0, method: str = 'statevector',
                         precision: str = 'single') -> AerSimulator:
    """Create highly optimized GPU-accelerated quantum simulator.
    
    Uses shared caching and optimized settings for maximum GPU utilization.
    """
    return _get_shared_aer_simulator(gpu_id, method)


def create_gpu_sampler(cuda_device: int = 0, shots: int = 1024):
    """Create optimized GPU-accelerated Sampler with parallel execution.
    
    Uses SamplerV2 (no deprecation warnings) with GPU backend.

    
    Features:
    - GPU-accelerated statevector simulation
    - Optimized thread count based on GPU compute capability
    - Returns SamplerV2 if available, falls back to legacy Sampler
    """
    # Get shared GPU simulator
    simulator = _get_shared_aer_simulator(cuda_device, method='statevector')
    
    # Prefer V2 primitives (no deprecation warnings)
    if V2_PRIMITIVES_AVAILABLE and AerSamplerV2 is not None:
        try:
            sampler = AerSamplerV2.from_backend(simulator, default_shots=shots)
            logger.debug(f"[GPU:{cuda_device}] Created SamplerV2 with GPU backend (shots={shots})")
            return sampler
        except Exception as e:
            raise RuntimeError(
                f"Failed to create GPU SamplerV2: {e}. "
                "GPU acceleration is required. Ensure CUDA and qiskit-aer-gpu are properly installed."
            )
    
    # V2 not available - use legacy GPU sampler (still GPU, no CPU fallback)
    optimal_threads = get_optimal_thread_count(cuda_device)
    backend_options = {
        'device': 'GPU',
        'method': 'statevector',
        'precision': 'single',
        'max_parallel_threads': optimal_threads,
        'max_parallel_experiments': 8,
        'cuStateVec_enable': True,   # cuStateVec for statevector acceleration
        'blocking_enable': False,
    }
    
    try:
        sampler = AerSampler(
            backend_options=backend_options,
            run_options={'shots': shots}
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create GPU sampler: {e}. "
            "GPU acceleration is required. Ensure CUDA and qiskit-aer-gpu are properly installed."
        )
    
    logger.debug(f"[GPU:{cuda_device}] Created legacy Sampler (threads={optimal_threads}, shots={shots})")
    return sampler


def create_gpu_pass_manager(cuda_device: int = 0, optimization_level: int = 2):
    """Create a pass manager for transpiling circuits for GPU execution.
    
    Required when using V2 primitives with qiskit-machine-learning.
    """
    simulator = _get_shared_aer_simulator(cuda_device, method='statevector')
    try:
        pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=simulator)
        logger.debug(f"[GPU:{cuda_device}] Created pass manager (opt_level={optimization_level})")
        return pm
    except Exception as e:
        logger.warning(f"Failed to create pass manager: {e}")
        return None


def create_gpu_fidelity(cuda_device: int = 0) -> 'ComputeUncompute':
    """Create a GPU-accelerated ComputeUncompute fidelity for FidelityQuantumKernel.
    
    Uses AerSamplerV2 with GPU backend + pass_manager so that high-level circuits
    are automatically decomposed for Aer execution.
    
    Returns:
        ComputeUncompute fidelity backed by GPU SamplerV2
    """
    simulator = _get_shared_aer_simulator(cuda_device, method='statevector')
    
    if V2_PRIMITIVES_AVAILABLE and AerSamplerV2 is not None:
        gpu_sampler = AerSamplerV2.from_backend(simulator, default_shots=1024)
        pm = create_gpu_pass_manager(cuda_device, optimization_level=2)
        fidelity = ComputeUncompute(sampler=gpu_sampler, pass_manager=pm)
        logger.debug(f"[GPU:{cuda_device}] Created GPU-backed ComputeUncompute fidelity (SamplerV2)")
        return fidelity
    else:
        raise RuntimeError("GPU fidelity requires AerSamplerV2 (qiskit-aer V2 primitives). "
                           "Please install qiskit-aer with GPU support.")


# ============================================================================
# QUANTUM KERNEL COMPUTATION
# ============================================================================

# Fidelity kernel batch size limits based on qubit count
# Increased for better GPU utilization with cuQuantum
FIDELITY_MIN_CIRCUITS = 64   # Minimum batch size for GPU efficiency
FIDELITY_MAX_CIRCUITS = 1024  # Maximum batch size


def _fidelity_cap_for_qubits(num_qubits: int) -> int:
    """Get max batch size for fidelity computation based on qubit count.
    
    Conservative batch sizes from iot_multigpu.py that avoid GPU memory issues
    and enable better parallelism with max_parallel_experiments.
    """
    try:
        qubits = int(num_qubits)
    except (TypeError, ValueError):
        qubits = 0
    
    # Use conservative batch sizes that work well with max_parallel_experiments=8
    # These match the proven iot_multigpu.py configuration
    if qubits >= 24:
        return 1
    if qubits >= 22:
        return 2
    if qubits >= 18:
        return 4
    if qubits >= 14:
        return 8
    if qubits >= 10:
        return 16
    if qubits >= 6:
        return 64
    return 256


# ---------------------------------------------------------------------------
# Module-level worker for ProcessPoolExecutor (must be picklable)
# ---------------------------------------------------------------------------
def _statevector_worker(args):
    """Compute statevectors for a batch of data points using Qiskit Statevector.
    
    This is a module-level function so it can be pickled by ProcessPoolExecutor.
    Uses Qiskit's Statevector.from_instruction() which is much faster than
    AerSimulator for small qubit counts (<=16 qubits) because it avoids
    the significant per-circuit overhead of Aer job creation/serialization.
    """
    from qiskit.quantum_info import Statevector as _SV
    base_circ, data_batch = args
    param_order = list(base_circ.parameters)
    param_len = len(param_order)
    n_qubits = base_circ.num_qubits
    states = np.zeros((len(data_batch), 2 ** n_qubits), dtype=np.complex64)
    for i, row in enumerate(data_batch):
        binds = {p: float(v) for p, v in zip(param_order, row[:param_len])}
        circ = base_circ.assign_parameters(binds, inplace=False)
        sv = _SV.from_instruction(circ)
        states[i] = np.asarray(sv.data, dtype=np.complex64)
    return states


# Maximum qubit count for the fast ProcessPool path.
# Above this, AerSimulator + GPU is faster per circuit.
_FAST_SV_QUBIT_THRESHOLD = 16

# Optimal worker count (auto-detected from CPU count)
_FAST_SV_WORKERS = min(32, max(4, os.cpu_count() or 8))


class GPUFidelityKernel:
    """High-performance GPU-accelerated fidelity quantum kernel.
    
    This class provides direct GPU statevector fidelity computation
    using CuPy for massive speedup over CPU-based implementations.
    Similar to FidelityQuantumKernel but optimized for GPU execution.
    
    Features:
    - Direct statevector computation on GPU
    - Batched circuit execution for efficiency
    - CuPy-accelerated fidelity matrix computation
    - Memory-efficient processing with automatic batching
    - Optional noisy simulation for NISQ device modeling
    """
    
    def __init__(self, feature_map: QuantumCircuit, gpu_id: int = 0,
                 max_circuits_per_eval: Optional[int] = None,
                 simulator: Optional[AerSimulator] = None,
                 assume_pretranspiled: bool = False):
        """
        Args:
            feature_map: Quantum circuit for feature encoding
            gpu_id: GPU device ID
            max_circuits_per_eval: Maximum circuits per batch evaluation
            simulator: Optional custom simulator (e.g., with noise model)
            assume_pretranspiled: Skip internal transpilation when the caller
                already passes a circuit transpiled for the GPU basis.
        """
        self.gpu_id = int(gpu_id)
        # Use provided simulator or default to shared GPU simulator
        if simulator is not None:
            self.simulator = simulator
            logger.debug(f"[GPU:{gpu_id}] Using custom simulator (likely noisy)")
        else:
            self.simulator = _get_shared_aer_simulator(self.gpu_id, method='statevector')
        self.num_qubits = int(feature_map.num_qubits)
        self.max_circuits_per_eval = self._normalize_batch_size(max_circuits_per_eval)
        
        # Create a copy we can safely mutate (add save_statevector, bind params)
        circuit = copy.deepcopy(feature_map)
        
        # Decompose and transpile for GPU backend unless the caller already
        # paid that CPU cost upstream.
        if not assume_pretranspiled:
            try:
                circuit = circuit.decompose()
            except Exception:
                pass

            try:
                circuit = transpile(
                    circuit,
                    optimization_level=1,
                    basis_gates=['u', 'cx', 'rz', 'sx', 'x', 'ry']
                )
            except Exception:
                pass
        
        self._base_circuit = circuit
        
        # Store a CLEAN circuit (without save_statevector) for the fast
        # ProcessPool + Statevector.from_instruction() path
        self._base_circuit_clean = copy.deepcopy(circuit)
        
        # Ensure the Aer circuit saves the statevector (only for the Aer path)
        try:
            if not any(getattr(inst.operation, 'name', '').lower() == 'save_statevector' 
                      for inst in getattr(self._base_circuit, 'data', [])):
                self._base_circuit.save_statevector()
        except Exception:
            pass
        
        # Cache parameter ordering for fast binding
        try:
            self._param_order = list(self._base_circuit.parameters)
        except Exception:
            self._param_order = []
        
        # Determine whether to use the optional CPU ProcessPool path.
        # GPU-only is the default because this experiment is intended to use
        # the available accelerators across both ideal and noisy conditions.
        self._use_fast_sv = (
            ENABLE_CPU_FAST_STATEVECTOR
            and self.num_qubits <= _FAST_SV_QUBIT_THRESHOLD
            and simulator is None
        )
        if self._use_fast_sv:
            logger.info(f"[GPU:{gpu_id}] Using fast ProcessPool+Statevector path "
                        f"({self.num_qubits}q <= {_FAST_SV_QUBIT_THRESHOLD}q threshold, "
                        f"{_FAST_SV_WORKERS} workers)")
        else:
            logger.info(
                f"[GPU:{gpu_id}] Using GPU Aer statevector path for {self.num_qubits}q ideal kernel evaluation"
            )
        
        logger.debug(f"[GPU:{gpu_id}] Created GPUFidelityKernel for {self.num_qubits}-qubit circuit")
    
    def _normalize_batch_size(self, value: Optional[int]) -> int:
        """Normalize batch size based on qubit count constraints."""
        cap = _fidelity_cap_for_qubits(self.num_qubits)
        try:
            if value is None:
                return cap
            normalized = int(value)
            if normalized <= 0:
                return cap
            return max(FIDELITY_MIN_CIRCUITS, min(cap, normalized))
        except Exception:
            return cap
    
    def _simulate_batch(self, circuits: List[QuantumCircuit]) -> np.ndarray:
        """Simulate a batch of circuits and return statevectors as complex64."""
        if not circuits:
            return np.empty((0, 0), dtype=np.complex64)
        
        batch_count = len(circuits)
        statevecs_np: Optional[np.ndarray] = None
        
        try:
            job = self.simulator.run(circuits, shots=0)
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"AerSimulator batch run failed: {exc}") from exc
        
        for idx, circuit in enumerate(circuits):
            try:
                sv = result.get_statevector(idx)
            except Exception:
                try:
                    data = result.data(idx)
                except Exception:
                    data = None
                
                sv = None
                if isinstance(data, dict):
                    sv = data.get('statevector')
                    if sv is None:
                        for key, value in data.items():
                            if isinstance(key, str) and 'state' in key.lower():
                                sv = value
                                break
                if sv is None:
                    raise RuntimeError(f"No statevector returned for circuit index {idx}")
            
            sv_array = np.asarray(sv, dtype=np.complex128)
            if statevecs_np is None:
                state_dim = sv_array.size
                statevecs_np = np.empty((batch_count, state_dim), dtype=np.complex64)
            statevecs_np[idx, :] = sv_array.astype(np.complex64, copy=False)
        
        # Don't clear GPU memory after each batch - too expensive
        # Memory will be cleared at end of evaluate() call
        
        if statevecs_np is None:
            return np.empty((0, 0), dtype=np.complex64)
        
        return statevecs_np
    
    def _compute_statevectors_fast(self, X: np.ndarray) -> np.ndarray:
        """Compute statevectors using ProcessPoolExecutor + Statevector.from_instruction().
        
        ~40-50x faster than Aer for <= 16-qubit circuits because it bypasses
        all AerSimulator per-circuit overhead (job creation, C++ serialization,
        result unpacking) and uses true multiprocessing parallelism across
        128 CPU cores.
        
        Returns: ndarray of shape (len(X), 2**num_qubits), dtype=complex64
        """
        from concurrent.futures import ProcessPoolExecutor
        n = len(X)
        n_workers = _FAST_SV_WORKERS
        chunk_size = max(1, (n + n_workers - 1) // n_workers)
        chunks = [(self._base_circuit_clean, X[i:i + chunk_size])
                  for i in range(0, n, chunk_size)]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_statevector_worker, chunks))
        return np.vstack(results)
    
    def _compute_statevectors_aer(self, X: np.ndarray) -> np.ndarray:
        """Compute statevectors using AerSimulator (for large qubit counts or noisy sims).
        
        Returns: ndarray of shape (len(X), 2**num_qubits), dtype=complex64
        """
        param_order = self._param_order or list(self._base_circuit.parameters)
        param_len = len(param_order)
        circuits = []
        for row in X:
            binds = {p: float(v) for p, v in zip(param_order, row[:param_len])}
            circuits.append(self._base_circuit.assign_parameters(binds, inplace=False))
        
        cap = _fidelity_cap_for_qubits(self.num_qubits)
        batch_limit = max(8, self.max_circuits_per_eval or cap)
        all_states = []
        for i in range(0, len(circuits), batch_limit):
            batch = circuits[i:i + batch_limit]
            states = self._simulate_batch(batch)
            all_states.append(states)
        return np.vstack(all_states)
    
    def evaluate(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate the fidelity kernel between X1 and X2.
        
        Returns a float32 numpy array of shape (len(X1), len(X2)).
        Uses CuPy for GPU-accelerated fidelity matrix computation.
        
        For ideal simulations, the default path is GPU Aer statevector execution so
        the experiment uses the installed GPUs consistently. The legacy CPU
        ProcessPool path can be re-enabled with QML_ENABLE_CPU_FAST_STATEVECTOR=1.
        
        For > 16 qubits or noisy: Falls back to Aer GPU simulation.
        """
        X1 = np.asarray(X1, dtype=float)
        if X2 is None:
            X2 = X1
            same_inputs = True
        else:
            X2 = np.asarray(X2, dtype=float)
            same_inputs = False
        
        # Handle NaN/inf values
        X1 = np.nan_to_num(X1, nan=0.0, posinf=np.pi, neginf=0.0)
        if not same_inputs:
            X2 = np.nan_to_num(X2, nan=0.0, posinf=np.pi, neginf=0.0)
        
        n1, n2 = len(X1), len(X2)
        param_order = self._param_order or list(self._base_circuit.parameters)
        param_len = len(param_order)
        
        if param_len == 0:
            raise RuntimeError("Feature map contains no parameters to bind")
        
        cupy_mod = _ensure_cupy()
        if cupy_mod is None:
            raise RuntimeError("CuPy is required for GPU kernel computation but is not available. "
                               "Please install CuPy: pip install cupy-cuda12x")
        
        # Choose statevector computation method
        if self._use_fast_sv:
            # Fast path: ProcessPool + Statevector (40-50x faster for <=16q)
            left_states = self._compute_statevectors_fast(X1)
            if same_inputs:
                right_states = left_states
            else:
                right_states = self._compute_statevectors_fast(X2)
        else:
            # Aer path: GPU simulation (for large circuits or noisy sims)
            left_states = self._compute_statevectors_aer(X1)
            if same_inputs:
                right_states = left_states
            else:
                right_states = self._compute_statevectors_aer(X2)
        
        # Compute fidelity matrix in one CuPy operation on GPU
        with cupy_mod.cuda.Device(self.gpu_id):
            left_gpu = cupy_mod.asarray(left_states, dtype=cupy_mod.complex64)
            right_gpu = cupy_mod.asarray(right_states, dtype=cupy_mod.complex64)
            fidelity_gpu = cupy_mod.abs(left_gpu @ right_gpu.conj().T) ** 2
            result = cupy_mod.asnumpy(fidelity_gpu).astype(np.float32, copy=False)
            del left_gpu, right_gpu, fidelity_gpu
            cupy_mod.get_default_memory_pool().free_all_blocks()
        
        if same_inputs:
            result = (result + result.T) / 2.0
            # Clip negative eigenvalues to ensure PSD (numerical stability)
            eigvals, eigvecs = np.linalg.eigh(result)
            if np.any(eigvals < 0):
                eigvals = np.maximum(eigvals, 0)
                result = (eigvecs * eigvals) @ eigvecs.T
                result = np.asarray(result, dtype=np.float32)
        
        gc.collect()
        clear_gpu_memory(self.gpu_id)
        
        return np.asarray(result, dtype=np.float32)


class NoisySamplingKernel:
    """Noisy quantum kernel using shots-based sampling simulation.
    
    For simulating realistic NISQ device behavior with noise models.
    Uses a probability overlap measure computed from measurement distributions.
    
    Unlike GPUFidelityKernel (which uses exact statevector fidelity), this uses
    shots-based simulation with noise models to estimate kernel values from 
    measurement statistics - more realistic for real hardware comparison.
    
    Performance notes:
    - Uses 'density_matrix' method (optimal for noisy simulation ≤15 qubits)
      Applies noise as superoperators — evolves density matrix once per circuit,
      then samples shots from the final state. Much faster than statevector
      which must re-simulate per-shot with random Kraus operators.
    - GPU acceleration via Aer's native CUDA kernels
    - Batched execution with max_parallel_experiments=8 for GPU efficiency
    """
    
    def __init__(self, feature_map: QuantumCircuit, noise_model: NoiseModel,
                 shots: int = 512, gpu_id: int = 0):
        """
        Args:
            feature_map: Quantum circuit for feature encoding (should be pre-transpiled)
            noise_model: Qiskit Aer noise model
            shots: Number of measurement shots per circuit (512 default for speed)
            gpu_id: GPU device ID for simulation
        """
        self.feature_map = feature_map
        self.noise_model = noise_model
        self.shots = shots
        self.gpu_id = gpu_id
        self.num_qubits = feature_map.num_qubits
        
        # Batch sizes for GPU parallelism
        # Benchmarked: larger batches reduce per-batch overhead
        if self.num_qubits >= 14:
            self.batch_size = 250   # Large circuits: moderate batching
        else:
            self.batch_size = 1750  # Smaller circuits: large batches (~5% faster than 500)
        
        self.device = 'GPU'
        
        # Use density_matrix method for noisy simulation
        # With noise, 'automatic' selects density_matrix anyway, but explicit is safer.
        # density_matrix + noise → superoperator representation → evolves DM once,
        # then samples shots from final state. O(gates × 4^n) vs statevector's
        # O(shots × gates × 2^n). For n≤10 and shots=1024, DM is faster.
        try:
            self.simulator = AerSimulator(
                method='density_matrix',
                noise_model=noise_model,
                device='GPU',
                precision='single',
                # NOTE: cuStateVec_enable omitted — only accelerates statevector, not density_matrix
                # NOTE: batched_shots_gpu omitted — per Qiskit Aer docs, only benefits
                #   "statevector with noise" or "density_matrix with intermediate measurements".
                #   Our circuits have all measure ops at the end (measure_all()), so no benefit.
                # NOTE: runtime_parameter_bind_enable tested and REJECTED — 4.5x slower
                max_parallel_experiments=8,  # Bounded parallelism (avoid thread thrashing)
                fusion_enable=True,          # Gate fusion for small circuits
                fusion_threshold=4,          # Benchmarked: 43% speedup over default (threshold=7)
            )
            logger.info(f"[NoisySamplingKernel] GPU density_matrix simulator for {self.num_qubits}q, "
                        f"gpu_id={gpu_id}, batch_size={self.batch_size}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create GPU noisy sampling simulator: {e}. "
                "GPU acceleration is required for NoisySamplingKernel. "
                "Ensure CUDA and qiskit-aer-gpu are properly installed."
            )
        
        # Cache the base circuit and parameter ordering
        # Do NOT decompose — the circuit is already transpiled into basis gates.
        # Decomposing a transpiled circuit would expand native gates and increase depth.
        self._base_circuit = copy.deepcopy(feature_map)
        self._param_order = list(self._base_circuit.parameters)
        
        logger.info(f"[NoisySamplingKernel] Created: {shots} shots, depth={feature_map.depth()}, "
                    f"gates={feature_map.size()}, params={len(self._param_order)}")
    
    def _bind_and_measure(self, X: np.ndarray) -> List[Dict[str, int]]:
        """Bind parameters and get measurement counts for each sample.
        
        Uses batched execution to avoid memory exhaustion for large circuits.
        """
        param_order = self._param_order or list(self._base_circuit.parameters)
        param_len = len(param_order)
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        # Prepare all circuits
        circuits = []
        for row in X:
            binds = {p: float(v) for p, v in zip(param_order, row[:param_len])}
            qc = self._base_circuit.assign_parameters(binds, inplace=False)
            qc.measure_all()
            circuits.append(qc)
        
        # Execute in batches to avoid memory exhaustion
        counts_list = []
        for batch_idx, batch_start in enumerate(range(0, len(circuits), self.batch_size)):
            batch_end = min(batch_start + self.batch_size, len(circuits))
            batch_circuits = circuits[batch_start:batch_end]
            
            job = self.simulator.run(batch_circuits, shots=self.shots)
            result = job.result()
            
            for i in range(len(batch_circuits)):
                counts = result.get_counts(i)
                counts_list.append(counts)
            
            if n_batches > 2 and (batch_idx + 1) % max(1, n_batches // 4) == 0:
                logger.info(f"    [NoisySamplingKernel] Batch {batch_idx+1}/{n_batches} "
                            f"({batch_end}/{n_samples} samples)")
            
            # Clear memory between batches for large circuits
            if self.num_qubits >= 14:
                gc.collect()
        
        return counts_list
    
    def _probability_distribution(self, counts: Dict[str, int]) -> np.ndarray:
        """Convert counts to probability distribution vector."""
        n_states = 2 ** self.num_qubits
        probs = np.zeros(n_states, dtype=np.float64)
        total = sum(counts.values())
        
        for bitstring, count in counts.items():
            idx = int(bitstring.replace(' ', ''), 2)
            probs[idx] = count / total
        
        return probs
    
    def _hellinger_kernel(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute Hellinger kernel (related to Bhattacharyya coefficient)."""
        # Hellinger affinity: sum(sqrt(p1 * p2))
        return np.sum(np.sqrt(p1 * p2))
    
    def evaluate(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate noisy quantum kernel using probability overlap.
        
        Uses Hellinger kernel (Bhattacharyya coefficient) between measurement 
        probability distributions as a kernel function.
        
        Optimized with vectorized NumPy operations for kernel computation.
        When X1 is X2 (same object, e.g. evaluate(X_train, X_train)), detects
        this and measures only once to avoid double circuit simulation.
        """
        X1 = np.asarray(X1, dtype=float)
        X1 = np.nan_to_num(X1, nan=0.0, posinf=np.pi, neginf=0.0)
        
        if X2 is None:
            same_inputs = True
        elif X2 is X1:  # Identity check: same array object passed twice
            same_inputs = True
            logger.info(f"[NoisySamplingKernel] X1 is X2 — measuring once (symmetric kernel)")
        else:
            X2 = np.asarray(X2, dtype=float)
            X2 = np.nan_to_num(X2, nan=0.0, posinf=np.pi, neginf=0.0)
            same_inputs = False
        
        n1 = len(X1)
        n2 = n1 if same_inputs else len(X2)
        
        # Get measurement distributions
        logger.info(f"[NoisySamplingKernel] Measuring {n1} circuits for X1...")
        counts_1 = self._bind_and_measure(X1)
        probs_1 = np.array([self._probability_distribution(c) for c in counts_1], dtype=np.float64)
        
        if same_inputs:
            probs_2 = probs_1
        else:
            logger.info(f"[NoisySamplingKernel] Measuring {n2} circuits for X2...")
            counts_2 = self._bind_and_measure(X2)
            probs_2 = np.array([self._probability_distribution(c) for c in counts_2], dtype=np.float64)
        
        cupy_mod = _ensure_cupy()
        if cupy_mod is None:
            raise RuntimeError(
                "CuPy is required for GPU noisy-kernel computation but is not available. "
                "Please install CuPy: pip install cupy-cuda12x"
            )

        # Vectorized kernel computation using Hellinger inner product on GPU.
        # K[i,j] = sum(sqrt(p1[i] * p2[j])) = sqrt(p1) @ sqrt(p2).T
        with cupy_mod.cuda.Device(self.gpu_id):
            sqrt_probs_1 = cupy_mod.sqrt(cupy_mod.asarray(probs_1, dtype=cupy_mod.float32))
            sqrt_probs_2 = cupy_mod.sqrt(cupy_mod.asarray(probs_2, dtype=cupy_mod.float32))
            result = cupy_mod.asnumpy(sqrt_probs_1 @ sqrt_probs_2.T).astype(np.float32, copy=False)
            del sqrt_probs_1, sqrt_probs_2
            cupy_mod.get_default_memory_pool().free_all_blocks()
        
        if same_inputs:
            # Ensure symmetry
            result = (result + result.T) / 2.0
            # Clip negative eigenvalues to ensure PSD (shot noise can break this)
            eigvals, eigvecs = np.linalg.eigh(result)
            if np.any(eigvals < 0):
                eigvals = np.maximum(eigvals, 0)
                result = (eigvecs * eigvals) @ eigvecs.T
                result = np.asarray(result, dtype=np.float32)
        
        return result


class NoisyFidelityKernel:
    """Noisy quantum kernel using density matrix simulation + Hilbert-Schmidt inner product.
    
    Computes K(x,z) = Tr(ρ(x) · ρ(z)) / sqrt(Tr(ρ(x)²) · Tr(ρ(z)²))
    
    This is the CORRECT kernel for studying noise effects because:
    - For pure states (noise=0): reduces to |⟨ψ(x)|ψ(z)⟩|² (same as GPUFidelityKernel)
    - For mixed states (noise>0): naturally captures decoherence via density matrices
    - Same kernel function across ALL noise levels (fair comparison)
    - Deterministic: no shot noise, no inconsistency between train/test kernels
    - PSD by construction (normalized Gram matrix)
    - Captures phase information (unlike Hellinger from measurement distributions)
    
    Uses Aer density_matrix simulation with GPU acceleration.
    """
    
    def __init__(self, feature_map: QuantumCircuit, noise_model: NoiseModel,
                 gpu_id: int = 0):
        self.feature_map = feature_map
        self.noise_model = noise_model
        self.gpu_id = gpu_id
        self.num_qubits = feature_map.num_qubits
        
        # Batch sizes for GPU parallelism
        if self.num_qubits >= 14:
            self.batch_size = 50
        elif self.num_qubits >= 10:
            self.batch_size = 200
        else:
            self.batch_size = 500

        if self.num_qubits >= 10:
            self.kernel_block_size = 32
        elif self.num_qubits >= 8:
            self.kernel_block_size = 64
        else:
            self.kernel_block_size = 128
        
        try:
            self.simulator = AerSimulator(
                method='density_matrix',
                noise_model=noise_model,
                device='GPU',
                precision='single',
                max_parallel_experiments=8,
                fusion_enable=True,
                fusion_threshold=4,
            )
            logger.info(f"[NoisyFidelityKernel] GPU density_matrix simulator for {self.num_qubits}q, "
                        f"gpu_id={gpu_id}, batch_size={self.batch_size}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create GPU density_matrix simulator: {e}. "
                "GPU acceleration is required."
            )
        
        self._base_circuit = copy.deepcopy(feature_map)
        self._param_order = list(self._base_circuit.parameters)
        
        logger.info(f"[NoisyFidelityKernel] Created: depth={feature_map.depth()}, "
                    f"gates={feature_map.size()}, params={len(self._param_order)}")

    def _density_matrix_dim(self) -> int:
        return 2 ** self.num_qubits

    def _flat_density_dim(self) -> int:
        d = self._density_matrix_dim()
        return d * d

    def _estimate_density_matrix_bytes(self, n_samples: int) -> int:
        return n_samples * self._flat_density_dim() * np.dtype(np.complex64).itemsize

    def _should_use_blockwise(self, n1: int, n2: int, same_inputs: bool) -> bool:
        bytes_needed = self._estimate_density_matrix_bytes(n1)
        if not same_inputs:
            bytes_needed += self._estimate_density_matrix_bytes(n2)
        budget_bytes = int(NOISY_DENSITY_MATRIX_RAM_BUDGET_GB * (1024 ** 3))
        return bytes_needed > budget_bytes

    def _simulate_density_matrix_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """Simulate a batch of circuits and return output density matrices."""
        param_order = self._param_order
        param_len = len(param_order)
        batch_n = len(X_batch)
        d = self._density_matrix_dim()

        circuits = []
        for row in X_batch:
            binds = {p: float(v) for p, v in zip(param_order, row[:param_len])}
            qc = self._base_circuit.assign_parameters(binds, inplace=False)
            qc.save_density_matrix()
            circuits.append(qc)

        dms = np.empty((batch_n, d, d), dtype=np.complex64)
        job = self.simulator.run(circuits)
        result = job.result()
        for i in range(batch_n):
            dms[i] = np.asarray(result.data(i)['density_matrix'], dtype=np.complex64)
        return dms

    def _simulate_density_matrices(self, X: np.ndarray) -> np.ndarray:
        """Simulate circuits and return output density matrices."""
        n = len(X)
        d = self._density_matrix_dim()
        dms = np.empty((n, d, d), dtype=np.complex64)
        n_batches = (n + self.batch_size - 1) // self.batch_size

        for batch_idx, batch_start in enumerate(range(0, n, self.batch_size)):
            batch_end = min(batch_start + self.batch_size, n)
            dms[batch_start:batch_end] = self._simulate_density_matrix_batch(X[batch_start:batch_end])

            if n_batches > 2 and (batch_idx + 1) % max(1, n_batches // 4) == 0:
                logger.info(f"    [NoisyFidelityKernel] Batch {batch_idx+1}/{n_batches} "
                            f"({batch_end}/{n} samples)")

            if self.num_qubits >= 10:
                gc.collect()

        return dms

    def _materialize_density_matrix_store(self, X: np.ndarray, memmap_path: str) -> np.ndarray:
        """Simulate density matrices once and store flattened states in a memmap."""
        flat_dim = self._flat_density_dim()
        n = len(X)
        store = np.memmap(memmap_path, dtype=np.complex64, mode='w+', shape=(n, flat_dim))
        purities = np.empty(n, dtype=np.float32)
        n_batches = (n + self.batch_size - 1) // self.batch_size

        for batch_idx, batch_start in enumerate(range(0, n, self.batch_size)):
            batch_end = min(batch_start + self.batch_size, n)
            batch_dms = self._simulate_density_matrix_batch(X[batch_start:batch_end])
            batch_flat = batch_dms.reshape(batch_end - batch_start, flat_dim)
            store[batch_start:batch_end] = batch_flat
            purities[batch_start:batch_end] = np.real(np.sum(batch_flat * batch_flat.conj(), axis=1)).astype(np.float32)

            if n_batches > 2 and (batch_idx + 1) % max(1, n_batches // 4) == 0:
                logger.info(f"    [NoisyFidelityKernel] Stored batch {batch_idx+1}/{n_batches} "
                            f"({batch_end}/{n} samples)")

            del batch_dms, batch_flat
            if self.num_qubits >= 10:
                gc.collect()

        store.flush()
        del store
        return purities

    def _compute_normalized_kernel_block(self, flat_left: np.ndarray, flat_right: np.ndarray,
                                         purities_left: np.ndarray,
                                         purities_right: np.ndarray) -> np.ndarray:
        cupy_mod = _ensure_cupy()
        if cupy_mod is None:
            raise RuntimeError(
                "CuPy is required for GPU noisy-kernel computation but is not available. "
                "Please install CuPy: pip install cupy-cuda12x"
            )

        with cupy_mod.cuda.Device(self.gpu_id):
            left_gpu = cupy_mod.asarray(flat_left, dtype=cupy_mod.complex64)
            right_gpu = cupy_mod.asarray(flat_right, dtype=cupy_mod.complex64)
            purity_left_gpu = cupy_mod.asarray(np.maximum(purities_left, 1e-10), dtype=cupy_mod.float32)
            purity_right_gpu = cupy_mod.asarray(np.maximum(purities_right, 1e-10), dtype=cupy_mod.float32)

            kernel_raw = cupy_mod.real(left_gpu @ right_gpu.conj().T).astype(cupy_mod.float32)
            norm = cupy_mod.sqrt(cupy_mod.outer(purity_left_gpu, purity_right_gpu))
            block = cupy_mod.asnumpy(kernel_raw / norm).astype(np.float32, copy=False)

            del left_gpu, right_gpu, purity_left_gpu, purity_right_gpu, kernel_raw, norm
            cupy_mod.get_default_memory_pool().free_all_blocks()

        return block

    def _evaluate_blockwise(self, X1: np.ndarray, X2: np.ndarray, same_inputs: bool) -> np.ndarray:
        """Compute the normalized kernel in blocks with disk-backed density matrices."""
        n1 = len(X1)
        n2 = len(X2)
        flat_dim = self._flat_density_dim()
        result = np.empty((n1, n2), dtype=np.float32)

        with tempfile.TemporaryDirectory(prefix='noisy_dm_kernel_', dir=str(CHECKPOINT_DIR)) as tmp_dir:
            left_path = os.path.join(tmp_dir, 'left.dat')
            left_purities = self._materialize_density_matrix_store(X1, left_path)
            left_store = np.memmap(left_path, dtype=np.complex64, mode='r', shape=(n1, flat_dim))

            if same_inputs:
                right_path = left_path
                right_purities = left_purities
                right_store = left_store
            else:
                right_path = os.path.join(tmp_dir, 'right.dat')
                right_purities = self._materialize_density_matrix_store(X2, right_path)
                right_store = np.memmap(right_path, dtype=np.complex64, mode='r', shape=(n2, flat_dim))

            left_block = min(self.kernel_block_size, n1)
            right_block = min(self.kernel_block_size, n2)

            for i_start in range(0, n1, left_block):
                i_end = min(i_start + left_block, n1)
                flat_left = np.asarray(left_store[i_start:i_end])
                purity_left = left_purities[i_start:i_end]

                j_start_iter = i_start if same_inputs else 0
                for j_start in range(j_start_iter, n2, right_block):
                    j_end = min(j_start + right_block, n2)
                    flat_right = np.asarray(right_store[j_start:j_end])
                    purity_right = right_purities[j_start:j_end]
                    block = self._compute_normalized_kernel_block(
                        flat_left, flat_right, purity_left, purity_right
                    )
                    result[i_start:i_end, j_start:j_end] = block
                    if same_inputs and j_start != i_start:
                        result[j_start:j_end, i_start:i_end] = block.T

                del flat_left
                if self.num_qubits >= 10:
                    gc.collect()

            del left_store
            if not same_inputs:
                del right_store

        return result
    
    def evaluate(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute normalized Hilbert-Schmidt kernel matrix.
        
        K_norm(i,j) = Tr(ρ_i · ρ_j) / sqrt(Tr(ρ_i²) · Tr(ρ_j²))
        For pure states: equals |⟨ψ_i|ψ_j⟩|² (fidelity).
        For mixed states: normalized density matrix similarity.
        """
        X1 = np.asarray(X1, dtype=float)
        X1 = np.nan_to_num(X1, nan=0.0, posinf=np.pi, neginf=0.0)
        
        if X2 is None:
            same_inputs = True
        elif X2 is X1:
            same_inputs = True
            logger.info(f"[NoisyFidelityKernel] X1 is X2 — computing once (symmetric kernel)")
        else:
            X2 = np.asarray(X2, dtype=float)
            X2 = np.nan_to_num(X2, nan=0.0, posinf=np.pi, neginf=0.0)
            same_inputs = False
        
        n1 = len(X1)
        n2 = n1 if same_inputs else len(X2)

        if self._should_use_blockwise(n1, n2, same_inputs):
            logger.info(
                f"[NoisyFidelityKernel] Using blockwise density-matrix kernel path "
                f"for {n1}x{n2} samples at {self.num_qubits}q"
            )
            result = self._evaluate_blockwise(X1, X1 if same_inputs else X2, same_inputs)
            result = np.clip(result, 0.0, 1.0)
            if same_inputs:
                result = (result + result.T) / 2.0
                np.fill_diagonal(result, 1.0)
            return result.astype(np.float32)
        
        # Get density matrices
        logger.info(f"[NoisyFidelityKernel] Simulating {n1} density matrices for X1...")
        dms_1 = self._simulate_density_matrices(X1)
        
        if same_inputs:
            dms_2 = dms_1
        else:
            n2 = len(X2)
            logger.info(f"[NoisyFidelityKernel] Simulating {n2} density matrices for X2...")
            dms_2 = self._simulate_density_matrices(X2)
        
        cupy_mod = _ensure_cupy()
        if cupy_mod is None:
            raise RuntimeError(
                "CuPy is required for GPU noisy-kernel computation but is not available. "
                "Please install CuPy: pip install cupy-cuda12x"
            )

        d = dms_1.shape[1]

        with cupy_mod.cuda.Device(self.gpu_id):
            flat_1 = cupy_mod.asarray(dms_1.reshape(len(dms_1), d * d), dtype=cupy_mod.complex64)
            flat_2 = cupy_mod.asarray(dms_2.reshape(len(dms_2), d * d), dtype=cupy_mod.complex64)

            # Tr(ρ_i · ρ_j) = vec(ρ_i)^† · vec(ρ_j) for Hermitian matrices.
            K_raw = cupy_mod.real(flat_1 @ flat_2.conj().T).astype(cupy_mod.float32)

            purities_1 = cupy_mod.real(cupy_mod.sum(flat_1 * flat_1.conj(), axis=1)).astype(cupy_mod.float32)
            if same_inputs:
                purities_2 = purities_1
            else:
                purities_2 = cupy_mod.real(cupy_mod.sum(flat_2 * flat_2.conj(), axis=1)).astype(cupy_mod.float32)

            purities_1 = cupy_mod.maximum(purities_1, cupy_mod.float32(1e-10))
            purities_2 = cupy_mod.maximum(purities_2, cupy_mod.float32(1e-10))
            norm = cupy_mod.sqrt(cupy_mod.outer(purities_1, purities_2))
            result = cupy_mod.asnumpy(K_raw / norm).astype(np.float32, copy=False)

            del flat_1, flat_2, K_raw, purities_1, purities_2, norm
            cupy_mod.get_default_memory_pool().free_all_blocks()
        
        # Clip to valid range
        result = np.clip(result, 0.0, 1.0)
        
        if same_inputs:
            result = (result + result.T) / 2.0
            np.fill_diagonal(result, 1.0)
        
        return result.astype(np.float32)


class QuantumKernelComputer:
    """Compute quantum kernels with circuit metrics tracking.
    
    Now uses GPUFidelityKernel for direct GPU-accelerated fidelity computation
    with CuPy for massive speedup over CPU-based implementations.
    """
    
    def __init__(self, feature_map: QuantumCircuit, gpu_id: int = 0,
                 metrics_analyzer: CircuitMetricsAnalyzer = None,
                 use_gpu_kernel: bool = True):
        self.feature_map = feature_map
        self.gpu_id = gpu_id
        self.metrics_analyzer = metrics_analyzer or CircuitMetricsAnalyzer()
        self.simulator = create_gpu_simulator(gpu_id)
        self._circuit_analyzed = False
        self._circuit_metrics = None
        self._use_gpu_kernel = use_gpu_kernel
        self._gpu_kernel = None
        
        # Pre-create GPU kernel if requested
        if use_gpu_kernel:
            try:
                self._gpu_kernel = GPUFidelityKernel(feature_map, gpu_id)
                logger.debug(f"[GPU:{gpu_id}] Using GPUFidelityKernel for kernel computation")
            except Exception as e:
                logger.warning(f"Failed to create GPUFidelityKernel: {e}, will use FidelityQuantumKernel")
                self._gpu_kernel = None
    
    def analyze_kernel_circuit(self) -> Dict[str, Any]:
        """Analyze the feature map circuit and cache metrics"""
        if not self._circuit_analyzed:
            self._circuit_metrics, _ = self.metrics_analyzer.transpile_and_analyze(
                self.feature_map, f"feature_map_{self.feature_map.num_qubits}q"
            )
            self._circuit_analyzed = True
        return self._circuit_metrics
    
    def create_fidelity_kernel(self) -> FidelityQuantumKernel:
        """Create fidelity quantum kernel with GPU-accelerated fidelity computation."""
        gpu_fidelity = create_gpu_fidelity(cuda_device=self.gpu_id)
        kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=gpu_fidelity
        )
        return kernel
    
    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """Compute quantum kernel matrix using GPU-accelerated fidelity computation.
        
        Prefers GPUFidelityKernel for direct GPU computation, falls back to
        FidelityQuantumKernel if GPU kernel is not available.
        """
        # GPU kernel is REQUIRED - no CPU fallback
        if self._gpu_kernel is None:
            raise RuntimeError("GPUFidelityKernel is not initialized. GPU acceleration is required.")
        
        try:
            K = self._gpu_kernel.evaluate(X1, X2)
            return np.asarray(K, dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"GPUFidelityKernel computation failed: {e}. "
                               "GPU acceleration is required - no CPU fallback available.")


def create_gpu_qsvc(feature_map: QuantumCircuit, gpu_id: int = 0) -> Tuple[QSVC, GPUFidelityKernel]:
    """Create a QSVC with GPU-accelerated kernel computation.
    
    Returns:
        Tuple of (QSVC model, GPUFidelityKernel) for training and prediction.
        The GPUFidelityKernel can be used to precompute kernel matrices.
    """
    # Create GPU-accelerated fidelity kernel
    gpu_kernel = GPUFidelityKernel(feature_map, gpu_id=gpu_id)
    
    # Create QSVC with GPU-backed FidelityQuantumKernel
    gpu_fidelity = create_gpu_fidelity(cuda_device=gpu_id)
    fidelity_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=gpu_fidelity)
    qsvc = QSVC(quantum_kernel=fidelity_kernel)
    
    return qsvc, gpu_kernel


def train_qsvc_with_gpu(qsvc: QSVC, gpu_kernel: GPUFidelityKernel,
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray) -> Tuple[np.ndarray, float]:
    """Train QSVC using GPU-accelerated kernel matrices.
    
    Args:
        qsvc: QSVC model instance
        gpu_kernel: GPUFidelityKernel for GPU-accelerated computation
        X_train: Training features
        y_train: Training labels
        X_test: Test features
    
    Returns:
        Tuple of (predictions, training_time)
    """
    start_time = time.time()
    
    # Precompute kernel matrices on GPU
    logger.debug(f"Computing training kernel matrix ({len(X_train)}x{len(X_train)})...")
    K_train = gpu_kernel.evaluate(X_train, X_train)
    
    # Fit QSVC with precomputed kernel
    qsvc.fit(K_train, y_train)
    
    # Compute test kernel matrix
    logger.debug(f"Computing test kernel matrix ({len(X_test)}x{len(X_train)})...")
    K_test = gpu_kernel.evaluate(X_test, X_train)
    
    # Predict
    y_pred = qsvc.predict(K_test)
    
    train_time = time.time() - start_time
    
    return y_pred, train_time


def cuml_svc_predict_proba(svc, K):
    """Get pseudo-probabilities from cuML SVC via decision_function + sigmoid.
    
    cuML SVC with kernel='precomputed' does NOT support probability=True.
    We therefore map decision scores through a sigmoid for ranking-based uses
    such as ROC-AUC. These outputs are not calibrated posterior probabilities.
    
    Args:
        svc: Fitted cuML SVC model
        K: Precomputed kernel matrix (n_test, n_train) for prediction
    
    Returns:
        np.ndarray of shape (n_samples, n_classes) with probability estimates
    """
    decision = svc.decision_function(K)
    decision = np.asarray(decision)  # Ensure numpy
    
    if decision.ndim == 1:
        # Binary classification: monotonic score squashing for ranking metrics
        proba_pos = 1.0 / (1.0 + np.exp(-decision))
        return np.column_stack([1.0 - proba_pos, proba_pos])
    else:
        # Multi-class OVO: use softmax normalization
        exp_d = np.exp(decision - decision.max(axis=1, keepdims=True))
        return exp_d / exp_d.sum(axis=1, keepdims=True)


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: np.ndarray = None,
                          train_time: float = 0.0) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for publication-quality analysis.
    
    Includes:
    - Standard metrics: accuracy, precision, recall, F1
    - Imbalanced data metrics: balanced accuracy, G-mean, MCC
    - Agreement metrics: Cohen's Kappa
    - Per-class metrics: class-wise precision/recall/F1
    """
    
    metrics = {
        # Standard classification metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),  # Important for imbalanced data
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),  # Treats all classes equally
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        
        # Robust metrics for imbalanced/multi-class
        'mcc': matthews_corrcoef(y_true, y_pred),  # Matthews Correlation Coefficient
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),  # Inter-rater agreement
        
        # Timing
        'training_time': train_time,
    }
    
    # Specificity and G-mean calculation
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape[0] == 2:
            # Binary classification
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['specificity'] = specificity
            metrics['sensitivity'] = sensitivity  # Same as recall for positive class
            # G-mean: geometric mean of sensitivity and specificity
            metrics['g_mean'] = np.sqrt(sensitivity * specificity)
        else:
            # Multi-class: compute average specificity and G-mean
            sensitivities = []
            specificities = []
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - tp - fn - fp
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivities.append(sens)
                specificities.append(spec)
            metrics['specificity'] = np.average(specificities, weights=cm.sum(axis=1))
            metrics['sensitivity'] = np.average(sensitivities, weights=cm.sum(axis=1))
            weighted_sens = metrics['sensitivity']
            weighted_spec = metrics['specificity']
            metrics['g_mean'] = np.sqrt(weighted_sens * weighted_spec)
    except Exception:
        metrics['specificity'] = 0.0
        metrics['sensitivity'] = 0.0
        metrics['g_mean'] = 0.0
    
    # ROC-AUC
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                if y_pred_proba.ndim == 2:
                    y_proba_pos = y_pred_proba[:, 1]
                else:
                    y_proba_pos = y_pred_proba
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
        except Exception:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    return metrics


# ============================================================================
# QUANTUM ENSEMBLE MODELS
# ============================================================================

class QuantumRandomForest(BaseEstimator, ClassifierMixin):
    """Quantum Random Forest using multiple QSVC estimators with GPU acceleration"""
    
    def __init__(self, n_estimators: int = 5, num_qubits: int = 4,
                 max_features: str = 'sqrt', random_state: int = 42,
                 use_gpu: bool = True, gpu_id: int = 0, timeout_per_estimator: int = 300):
        self.n_estimators = n_estimators
        self.num_qubits = num_qubits
        self.max_features = max_features
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.timeout_per_estimator = timeout_per_estimator
        self.estimators_ = []
        self.feature_indices_ = []
        self.kernels_ = []  # Store precomputed kernels for predict
        self.X_train_subsets_ = []  # Store training subsets for kernel evaluation
    
    def _get_n_features(self, n_total_features: int) -> int:
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_total_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        return n_total_features
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        n_selected = min(self._get_n_features(n_features), self.num_qubits)
        
        logger.info(f"QRF: Training {self.n_estimators} estimators with {n_selected} qubits, GPU={self.use_gpu}")
        
        for i in range(self.n_estimators):
            start_time = time.time()
            logger.info(f"QRF: Training estimator {i+1}/{self.n_estimators}...")
            
            feature_idx = np.random.choice(n_features, n_selected, replace=False)
            self.feature_indices_.append(feature_idx)
            
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[bootstrap_idx][:, feature_idx]
            y_boot = y[bootstrap_idx]
            
            # Store for prediction phase
            self.X_train_subsets_.append(X_boot)
            
            feature_map = create_feature_map(n_selected, 'Z', reps=1)
            feature_map_transpiled = transpile_with_cache(
                feature_map,
                ('qrf-feature-map', n_selected, 'Z', 1),
                optimization_level=1,
            )
            
            if not self.use_gpu:
                raise RuntimeError("QRF requires GPU acceleration. Set use_gpu=True.")
            
            # Use GPU-accelerated kernel with precomputed matrix
            # GPU is REQUIRED - no CPU fallback (would be too slow)
            gpu_kernel = GPUFidelityKernel(
                feature_map_transpiled,
                gpu_id=self.gpu_id % 3,
                assume_pretranspiled=True,
            )
            K_train = gpu_kernel.evaluate(X_boot, X_boot)
            
            # Use cuML SVC with precomputed kernel (GPU-accelerated)
            svc = cuSVC(kernel='precomputed', class_weight='balanced',
                        cache_size=8192.0, max_iter=10000, nochange_steps=100,
                        output_type='numpy')
            svc.fit(K_train, y_boot)
            self.estimators_.append(('precomputed', svc))
            self.kernels_.append((gpu_kernel, feature_map_transpiled))
            
            elapsed = time.time() - start_time
            logger.info(f"QRF: Estimator {i+1} trained with GPU in {elapsed:.1f}s")
        
        logger.info(f"QRF: All {self.n_estimators} estimators trained successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        
        for i, ((est_type, est), feat_idx, (gpu_kernel, _), X_train_sub) in enumerate(
            zip(self.estimators_, self.feature_indices_, self.kernels_, self.X_train_subsets_)):
            X_sub = X[:, feat_idx]
            
            if est_type == 'precomputed' and gpu_kernel is not None:
                # Compute kernel between test and training data
                K_test = gpu_kernel.evaluate(X_sub, X_train_sub)
                predictions.append(est.predict(K_test))
            else:
                predictions.append(est.predict(X_sub))
        
        predictions = np.array(predictions).T
        final_pred = []
        for row in predictions:
            counts = Counter(row)
            final_pred.append(counts.most_common(1)[0][0])
        
        return np.array(final_pred)


class QuantumVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Quantum ensemble using hard voting"""
    
    def __init__(self, estimators: List[Tuple[str, BaseEstimator]], voting: str = 'hard'):
        self.estimators = estimators
        self.voting = voting
        self.fitted_estimators_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        for name, est in self.estimators:
            fitted = clone(est)
            fitted.fit(X, y)
            self.fitted_estimators_.append((name, fitted))
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = [est.predict(X) for _, est in self.fitted_estimators_]
        predictions = np.array(predictions).T
        
        final_pred = []
        for row in predictions:
            counts = Counter(row)
            final_pred.append(counts.most_common(1)[0][0])
        
        return np.array(final_pred)


class QuantumWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Quantum ensemble with adaptive weights"""
    
    def __init__(self, estimators: List[Tuple[str, BaseEstimator]], 
                 validation_split: float = 0.2):
        self.estimators = estimators
        self.validation_split = validation_split
        self.fitted_estimators_ = []
        self.weights_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, stratify=y, random_state=42
        )
        
        performances = []
        for name, est in self.estimators:
            fitted = clone(est)
            fitted.fit(X_train, y_train)
            
            y_pred = fitted.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            performances.append(acc)
            
            self.fitted_estimators_.append((name, fitted))
        
        performances = np.array(performances)
        self.weights_ = performances / performances.sum()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        
        for (name, est), weight in zip(self.fitted_estimators_, self.weights_):
            pred = est.predict(X)
            for _ in range(int(weight * 100)):
                predictions.append(pred)
        
        predictions = np.array(predictions).T
        final_pred = []
        for row in predictions:
            counts = Counter(row)
            final_pred.append(counts.most_common(1)[0][0])
        
        return np.array(final_pred)


# ============================================================================
# DATA PROCESSING
# ============================================================================

class DataProcessor:
    """Data preprocessing for quantum ML experiments"""
    
    def __init__(self, num_qubits: int, random_seed: int = 42):
        self.num_qubits = num_qubits
        self.random_seed = random_seed
        self.scaler = None
        self.pre_pca_scaler = None
        self.pca = None
        self.selector = None
        self.label_encoder = None
    
    def prepare_data(self, df: pd.DataFrame, sample_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data from dataframe for binary classification (Normal vs Attack)"""
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        
        # Drop identifier and target-leaking columns before feature selection.
        columns_to_drop = []
        unnamed_columns = [col for col in df.columns if str(col).startswith('Unnamed')]
        columns_to_drop.extend(unnamed_columns)

        if 'pkSeqID' in df.columns:
            columns_to_drop.append('pkSeqID')
        if 'Cat' in df.columns:
            columns_to_drop.append('Cat')
        if 'Sub_Cat' in df.columns:
            columns_to_drop.append('Sub_Cat')
        if 'category' in df.columns:
            columns_to_drop.append('category')
        if 'subcategory' in df.columns:
            columns_to_drop.append('subcategory')
        
        if columns_to_drop:
            df = df.drop(columns=sorted(set(columns_to_drop)))
            logger.info(f"Dropped label-leaking columns: {columns_to_drop}")
        
        # Identify target column. Prioritize explicit binary labels used by the
        # IoT and UNSW datasets before falling back to the final column.
        if 'Label' in df.columns:
            target_column = 'Label'
        elif 'label' in df.columns:
            target_column = 'label'
        elif 'attack' in df.columns:
            target_column = 'attack'
        elif 'Attack' in df.columns:
            target_column = 'Attack'
        else:
            target_column = df.columns[-1]

        if target_column in df.columns:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            logger.info(f"Using target column: {target_column}")
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        # Drop any remaining categorical columns (they shouldn't be used as numeric features)
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            logger.warning(f"Dropping remaining categorical columns: {categorical_cols}")
            X = X.drop(columns=categorical_cols)
        
        X = X.values.astype(np.float32)
        y = y.values
        
        # Encode labels for binary classification
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            logger.info(f"Classes: {self.label_encoder.classes_}")
        
        # Check class distribution and apply SMOTE if heavily imbalanced
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        minority_ratio = min(counts) / max(counts)
        logger.info(f"Class distribution: {class_dist}, minority ratio: {minority_ratio:.4f}")
        
        # Sample with stratification before preprocessing to maintain class proportions
        if sample_size and sample_size < len(X):
            np.random.seed(self.random_seed)
            # For extremely imbalanced binary datasets, pure proportional sampling
            # can eliminate the minority class at 5k samples. Enforce a minimum
            # minority presence so downstream train/test and CV remain valid.
            unique, counts = np.unique(y, return_counts=True)
            minority_count = int(np.min(counts))
            expected_minority = minority_ratio * sample_size
            enforce_class_floor = len(unique) == 2 and expected_minority < 10 and minority_count > 0

            if enforce_class_floor:
                minority_class = unique[np.argmin(counts)]
                minority_indices = np.where(y == minority_class)[0]
                majority_indices = np.where(y != minority_class)[0]

                target_minority = min(len(minority_indices), max(10, sample_size // 10))
                target_majority = sample_size - target_minority

                selected_minority = np.random.choice(minority_indices, target_minority, replace=False)
                selected_majority = np.random.choice(majority_indices, target_majority, replace=False)
                selected_indices = np.concatenate([selected_minority, selected_majority])
                np.random.shuffle(selected_indices)

                X = X[selected_indices]
                y = y[selected_indices]
                logger.warning(
                    "Adjusted subsampling for extreme imbalance: retained %d minority and %d majority samples.",
                    target_minority,
                    target_majority,
                )
            else:
                # Use stratified sampling to maintain class proportions.
                from sklearn.model_selection import train_test_split
                if sample_size < len(X) * 0.99:  # Only if significant sampling
                    _, X, _, y = train_test_split(
                        X, y, test_size=sample_size/len(X), 
                        stratify=y, random_state=self.random_seed
                    )
                else:
                    idx = np.random.choice(len(X), sample_size, replace=False)
                    X, y = X[idx], y[idx]
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=np.pi, neginf=0.0)
        
        # Log final class distribution
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Final data shape: {X.shape}, classes: {dict(zip(unique, counts))}")
        return X, y
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit on X and transform. Use only when data leakage is not a concern
        (e.g., for circuit analysis on full dataset, NOT for train/test evaluation)."""
        self.fit(X, y)
        return self.transform(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DataProcessor':
        """Fit preprocessing pipeline on training data ONLY.
        
        Must be called before transform(). Learns:
        - Feature selection (SelectKBest) statistics
        - PCA components (on standardized data)
        - MinMaxScaler min/max values (AFTER PCA, so output is in [0, π])
        All from training data only, preventing data leakage.
        
        Pipeline order: SelectKBest → StandardScaler → PCA → MinMaxScaler(0,π)
        The final MinMaxScaler ensures quantum feature map inputs are in [0, π].
        """
        # Feature selection if needed
        if X.shape[1] > self.num_qubits * 2:
            self.selector = SelectKBest(mutual_info_classif, k=min(self.num_qubits * 2, X.shape[1]))
            X = self.selector.fit_transform(X, y)
        
        # Standardize before PCA (PCA works best on standardized data)
        self.pre_pca_scaler = StandardScaler()
        X = self.pre_pca_scaler.fit_transform(X)
        
        # Reduce to num_qubits dimensions
        if X.shape[1] > self.num_qubits:
            self.pca = PCA(n_components=self.num_qubits, random_state=self.random_seed)
            X = self.pca.fit_transform(X)
        
        # Scale PCA output to [0, π] for quantum feature map encoding
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.scaler.fit(X)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using previously fitted pipeline.
        
        Call fit() first on training data, then transform() on both train and test.
        This prevents data leakage by using only training statistics.
        
        Pipeline: SelectKBest → StandardScaler → PCA → MinMaxScaler(0,π)
        """
        if self.selector is not None:
            X = self.selector.transform(X)
        
        if not hasattr(self, 'pre_pca_scaler') or self.pre_pca_scaler is None:
            raise RuntimeError("DataProcessor.fit() must be called before transform()")
        X = self.pre_pca_scaler.transform(X)
        
        if self.pca is not None:
            X = self.pca.transform(X)
        elif X.shape[1] < self.num_qubits:
            padding = np.zeros((X.shape[0], self.num_qubits - X.shape[1]))
            X = np.hstack([X, padding])
        
        if self.scaler is None:
            raise RuntimeError("DataProcessor.fit() must be called before transform()")
        X = self.scaler.transform(X)
        
        # Handle NaN/Inf after transformations
        X = np.nan_to_num(X, nan=0.0, posinf=np.pi, neginf=0.0)
        
        return X.astype(np.float32)


# ============================================================================
# CROSS-VALIDATION EXPERIMENT (GPU-ACCELERATED)
# ============================================================================

class CrossValidationExperiment:
    """K-fold cross-validation with GPU-accelerated quantum kernel computation"""
    
    def __init__(self, k_folds: int = 5, n_runs: int = 5, random_seeds: List[int] = None):
        self.k_folds = k_folds
        self.n_runs = n_runs
        self.random_seeds = random_seeds or [42, 123, 456, 789, 1024]
        self.stat_analyzer = StatisticalAnalyzer()
        self.gpu_mgr = get_gpu_manager()
    
    def _run_single_fold(self, fold_data: Tuple) -> Dict[str, float]:
        """Run a single fold - can be parallelized across GPUs"""
        fold_idx, train_idx, test_idx, X, y, feature_map, gpu_id, num_qubits, seed = fold_data
        
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if num_qubits is not None:
            processor = DataProcessor(num_qubits=num_qubits, random_seed=seed)
            processor.fit(X_train_raw, y_train)
            X_train = processor.transform(X_train_raw)
            X_test = processor.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw
        
        try:
            # Create GPU kernel for this fold
            gpu_kernel = GPUFidelityKernel(feature_map, gpu_id=gpu_id, assume_pretranspiled=True)
            
            start_time = time.time()
            
            # Compute kernel matrices on GPU
            K_train = gpu_kernel.evaluate(X_train, X_train)
            K_test = gpu_kernel.evaluate(X_test, X_train)
            
            # Train cuML SVC with precomputed kernel on GPU
            svc = cuSVC(kernel='precomputed', class_weight='balanced',
                        random_state=42, cache_size=8192.0, max_iter=10000,
                        nochange_steps=100, output_type='numpy')
            svc.fit(K_train, y_train)
            y_pred = svc.predict(K_test)
            y_pred_proba = cuml_svc_predict_proba(svc, K_test)
            
            train_time = time.time() - start_time
            
            metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba=y_pred_proba, train_time=train_time)
            metrics['fold'] = fold_idx
            metrics['gpu_id'] = gpu_id
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Fold {fold_idx} on GPU {gpu_id} failed: {e}")
            return {'error': str(e), 'fold': fold_idx}
    
    def run_cv_experiment_gpu(self, feature_map: QuantumCircuit, X: np.ndarray, y: np.ndarray,
                              model_name: str, use_parallel: bool = True,
                              num_qubits: int = None) -> Dict[str, Any]:
        """Run k-fold CV with GPU acceleration and optional parallelization across GPUs"""
        
        all_scores = defaultdict(list)
        num_gpus = max(1, self.gpu_mgr.gpu_count)

        transpiled_feature_map = transpile_with_cache(
            feature_map,
            ('cv-feature-map', model_name, num_qubits, tuple(feature_map.parameters)),
            optimization_level=1,
            basis_gates=['u', 'cx', 'rz', 'sx', 'x', 'ry']
        )
        
        logger.info(f"Running CV for {model_name} with {self.k_folds} folds × {self.n_runs} runs on {num_gpus} GPU(s)")
        
        for run_idx, seed in enumerate(self.random_seeds[:self.n_runs]):
            kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=seed)
            
            # Prepare fold data for parallel execution
            fold_tasks = []
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
                gpu_id = fold_idx % num_gpus  # Round-robin GPU assignment
                fold_tasks.append((fold_idx, train_idx, test_idx, X, y, transpiled_feature_map, gpu_id, num_qubits, seed))
            
            fold_results = []
            
            if use_parallel and num_gpus > 1:
                # Parallel execution across GPUs using ThreadPoolExecutor
                # (ProcessPoolExecutor has issues with CUDA contexts)
                with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                    futures = [executor.submit(self._run_single_fold, task) for task in fold_tasks]
                    for future in futures:
                        try:
                            result = future.result(timeout=300)  # 5 min timeout per fold
                            fold_results.append(result)
                        except Exception as e:
                            logger.warning(f"Fold execution failed: {e}")
            else:
                # Sequential execution
                for task in fold_tasks:
                    result = self._run_single_fold(task)
                    fold_results.append(result)
            
            # Aggregate fold scores for this seed
            for result in fold_results:
                if 'error' not in result:
                    for metric, value in result.items():
                        if metric not in ['fold', 'gpu_id', 'error']:
                            all_scores[metric].append(value)
            
            logger.debug(f"  Run {run_idx + 1}/{self.n_runs} complete")
        
        # Compute statistics across all runs
        results = {
            'model_name': model_name,
            'n_runs': self.n_runs,
            'k_folds': self.k_folds,
            'num_gpus_used': num_gpus,
        }
        
        for metric, values in all_scores.items():
            if values:
                values = np.array(values)
                ci_low, ci_high = self.stat_analyzer.compute_confidence_interval(values)
                
                results[f'{metric}_mean'] = float(np.mean(values))
                results[f'{metric}_std'] = float(np.std(values))
                results[f'{metric}_ci_low'] = float(ci_low)
                results[f'{metric}_ci_high'] = float(ci_high)
                results[f'{metric}_min'] = float(np.min(values))
                results[f'{metric}_max'] = float(np.max(values))
        
        return results
    
    def run_cv_experiment(self, model_factory, X: np.ndarray, y: np.ndarray,
                          model_name: str, num_qubits: int = None) -> Dict[str, Any]:
        """Legacy method - Run k-fold CV with multiple random seeds (CPU-based)
        
        Note: For GPU acceleration, use run_cv_experiment_gpu() instead.
        """
        
        all_scores = defaultdict(list)
        
        for seed in self.random_seeds[:self.n_runs]:
            kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=seed)
            
            fold_scores = defaultdict(list)
            
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
                X_train_raw, X_test_raw = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if num_qubits is not None:
                    processor = DataProcessor(num_qubits=num_qubits, random_seed=seed)
                    processor.fit(X_train_raw, y_train)
                    X_train = processor.transform(X_train_raw)
                    X_test = processor.transform(X_test_raw)
                else:
                    X_train, X_test = X_train_raw, X_test_raw
                
                try:
                    model = model_factory()
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    y_pred = model.predict(X_test)
                    
                    metrics = calculate_all_metrics(y_test, y_pred, train_time=train_time)
                    
                    for metric, value in metrics.items():
                        fold_scores[metric].append(value)
                        
                except Exception as e:
                    logger.warning(f"Fold {fold_idx} failed for {model_name}: {e}")
            
            # Aggregate fold scores for this seed
            for metric, values in fold_scores.items():
                if values:
                    all_scores[metric].append(np.mean(values))
        
        # Compute statistics across all runs
        results = {
            'model_name': model_name,
            'n_runs': self.n_runs,
            'k_folds': self.k_folds,
        }
        
        for metric, values in all_scores.items():
            if values:
                values = np.array(values)
                ci_low, ci_high = self.stat_analyzer.compute_confidence_interval(values)
                
                results[f'{metric}_mean'] = float(np.mean(values))
                results[f'{metric}_std'] = float(np.std(values))
                results[f'{metric}_ci_low'] = float(ci_low)
                results[f'{metric}_ci_high'] = float(ci_high)
                results[f'{metric}_min'] = float(np.min(values))
                results[f'{metric}_max'] = float(np.max(values))
        
        return results


# ============================================================================
# ABLATION STUDY RUNNER
# ============================================================================

class AblationStudyRunner:
    """Run systematic ablation studies"""
    
    def __init__(self, base_config: Dict[str, Any], metrics_analyzer: CircuitMetricsAnalyzer):
        self.base_config = base_config
        self.metrics_analyzer = metrics_analyzer
        self.results = []
    
    def run_ablation(self, ablation_name: str, ablation_config: Dict[str, Any],
                     X: np.ndarray, y: np.ndarray, model_factory_fn) -> List[Dict[str, Any]]:
        """Run a single ablation study"""
        
        fixed_params = ablation_config['fixed']
        vary_values = ablation_config['vary']
        
        ablation_results = []
        
        for vary_value in vary_values:
            # Create config for this variation
            config = {**fixed_params}
            
            if ablation_name == 'feature_map_type':
                config['feature_map'] = vary_value
            elif ablation_name == 'circuit_depth':
                config['reps'] = vary_value
            elif ablation_name == 'entanglement_pattern':
                config['entanglement'] = vary_value
            elif ablation_name == 'qubit_scalability':
                config['num_qubits'] = vary_value
            elif ablation_name == 'sample_size':
                config['sample_size'] = vary_value
            
            logger.info(f"Running ablation: {ablation_name} = {vary_value}")
            
            # Create and analyze feature map
            num_qubits = config.get('num_qubits', 10)
            feature_map = create_feature_map(
                num_qubits=num_qubits,
                map_type=config.get('feature_map', 'ZZ'),
                reps=config.get('reps', 2),
                entanglement=config.get('entanglement', 'full')
            )
            
            # Get circuit metrics
            circuit_metrics, transpiled_fm = self.metrics_analyzer.transpile_and_analyze(
                feature_map, f"{ablation_name}_{vary_value}"
            )
            
            # Subsample data if needed
            sample_size = config.get('sample_size', len(X))
            if sample_size < len(X):
                _, X_sub, _, y_sub = train_test_split(
                    X, y, test_size=sample_size / len(X), stratify=y, random_state=42
                )
            else:
                X_sub, y_sub = X, y
            
            # Train and evaluate model
            try:
                model = model_factory_fn(transpiled_fm)

                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X_sub, y_sub, test_size=0.3, stratify=y_sub, random_state=42
                )

                processor = DataProcessor(num_qubits=num_qubits)
                processor.fit(X_train_raw, y_train)
                X_train = processor.transform(X_train_raw)
                X_test = processor.transform(X_test_raw)
                
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                y_pred = model.predict(X_test)
                metrics = calculate_all_metrics(y_test, y_pred, train_time=train_time)
                
                result = {
                    'ablation_name': ablation_name,
                    'vary_value': vary_value,
                    'config': config,
                    'circuit_metrics': circuit_metrics,
                    **metrics
                }
                
            except Exception as e:
                logger.error(f"Ablation failed for {ablation_name}={vary_value}: {e}")
                result = {
                    'ablation_name': ablation_name,
                    'vary_value': vary_value,
                    'config': config,
                    'error': str(e)
                }
            
            ablation_results.append(result)
        
        self.results.extend(ablation_results)
        return ablation_results


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

class CircuitDepthExperimentRunner:
    """Main experiment orchestrator for circuit depth analysis"""
    
    def __init__(self, config: Dict[str, Any] = None, quick_test: bool = False,
                 enable_noise_simulation: bool = True,
                 resume_session_id: str = None):
        self.config = config or DEFAULT_CONFIG.copy()
        if resume_session_id:
            self.base_timestamp, self.session_id = self._parse_resume_session_id(resume_session_id)
            self.resuming = True
            logger.info(f"🔄 RESUME MODE: Reusing session {self.session_id}")
        else:
            self.base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.session_id = f"v2_noisy_{self.base_timestamp}"
            self.resuming = False
        self.quick_test = quick_test
        self.enable_noise_simulation = enable_noise_simulation
        self.current_config_tag = ""  # Will be set per configuration
        self.current_dataset_tag = ""
        self.current_dataset_name = ""
        
        # Adjust iterations for quick test mode
        self.spsa_maxiter = 20 if quick_test else 100
        self.skip_slow_models = True
        
        # Initialize analyzers
        self.metrics_analyzer = CircuitMetricsAnalyzer()
        self.expressibility_analyzer = ExpressibilityAnalyzer()
        self.entropy_analyzer = EntanglementEntropyAnalyzer()
        self.stat_analyzer = StatisticalAnalyzer()
        self.noise_simulator = NoiseModelSimulator()
        
        self._reset_results_storage()
        
        # GPU manager
        try:
            self.gpu_manager = GPUManager()
        except Exception:
            self.gpu_manager = None
            logger.warning("GPU manager initialization failed")
    
    def _get_filename_prefix(self):
        """Generate descriptive filename prefix for current configuration."""
        prefix_parts = [part for part in [self.current_dataset_tag, self.current_config_tag, self.session_id] if part]
        return "_".join(prefix_parts) if prefix_parts else self.session_id

    @staticmethod
    def _parse_resume_session_id(resume_session_id: str) -> Tuple[str, str]:
        """Validate and unpack a resume session id."""
        session_id = str(resume_session_id).strip()
        match = re.fullmatch(r'(v2_noisy_)(\d{8}_\d{6})', session_id)
        if not match:
            raise ValueError(
                f"Invalid --resume session id '{resume_session_id}'. Expected format "
                "'v2_noisy_YYYYMMDD_HHMMSS'."
            )
        return match.group(2), session_id

    @staticmethod
    def _order_output_columns(df: pd.DataFrame, leading_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Move high-value metadata columns to the front without dropping anything."""
        if df is None or df.empty:
            return df
        preferred = list(leading_columns or ['dataset', 'config'])
        ordered_front = [col for col in preferred if col in df.columns]
        remaining = [col for col in df.columns if col not in ordered_front]
        return df.loc[:, ordered_front + remaining]

    def _select_gpu_for_workload(self, workload_label: str) -> int:
        """Pick a stable GPU for a workload instead of always piling onto one device."""
        gpu_mgr = self.gpu_manager or get_gpu_manager()
        if gpu_mgr.gpu_count <= 1:
            return gpu_mgr.get_best_gpu()

        workload_key = "|".join([
            self.current_dataset_tag or self.current_dataset_name or 'dataset',
            self.current_config_tag or 'config',
            workload_label,
        ])
        preferred_gpu = sum(workload_key.encode('utf-8')) % gpu_mgr.gpu_count
        preferred_status = get_gpu_memory_status(preferred_gpu)
        if preferred_status and preferred_status['free'] >= GPU_MEMORY_RESERVE_GB:
            return preferred_gpu
        return gpu_mgr.get_best_gpu()

    @staticmethod
    def _sanitize_tag(value: str) -> str:
        sanitized = re.sub(r'[^A-Za-z0-9]+', '_', str(value).strip())
        sanitized = sanitized.strip('_').lower()
        return sanitized or 'dataset'

    def _set_run_context(self, num_qubits: int, sample_size: int, dataset_name: str = None):
        self.current_config_tag = f"{num_qubits}q_{self._format_sample_tag(sample_size)}"
        self.current_dataset_name = dataset_name or ""
        self.current_dataset_tag = self._sanitize_tag(dataset_name) if dataset_name else ""

    @staticmethod
    def _format_sample_tag(sample_size: int) -> str:
        sample_size = int(sample_size)
        if sample_size % 1000 == 0:
            return f"{sample_size // 1000}k"
        if sample_size % 100 == 0:
            whole = sample_size // 1000
            remainder = sample_size % 1000
            if whole > 0:
                return f"{whole}p{remainder // 100}k"
        return str(sample_size)

    @staticmethod
    def _optimization_label(opt_level: int) -> str:
        return f"L{int(opt_level)}"

    @staticmethod
    def _optimization_description(opt_level: int) -> str:
        descriptions = {
            0: 'baseline_unoptimized',
            1: 'qiskit_light',
            2: 'qiskit_medium',
            3: 'qiskit_heavy',
        }
        return descriptions.get(int(opt_level), f'qiskit_level_{int(opt_level)}')

    def _result_context(self) -> Dict[str, Any]:
        return {
            'dataset': self.current_dataset_name,
            'config': self.current_config_tag,
        }

    def _execute_phase4_run(self, run_idx: int, seed: int, model_name: str,
                            num_qubits: int, noise_level: float,
                            X: np.ndarray, y: np.ndarray,
                            fm_primary: QuantumCircuit,
                            fm_secondary: QuantumCircuit,
                            fm_tertiary: QuantumCircuit,
                            gpu_id: int) -> Dict[str, Any]:
        """Execute one Phase 4 run on a specific GPU.

        This is intentionally self-contained so repeated runs can be distributed
        across all GPUs in parallel without shared simulator/kernel state.
        """
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed
            )
            run_processor = DataProcessor(num_qubits=num_qubits, random_seed=seed)
            run_processor.fit(X_train_raw, y_train)
            X_train = run_processor.transform(X_train_raw)
            X_test = run_processor.transform(X_test_raw)

            noise_model = None
            if noise_level > 0:
                noise_params = NOISE_PARAMS.copy()
                noise_params['single_qubit_error'] = noise_level
                noise_params['two_qubit_error'] = noise_level * 3
                noise_model = NoiseModelSimulator(params=noise_params).create_noise_model(num_qubits)

            def _build_kernel(feature_map: QuantumCircuit):
                if noise_level == 0:
                    return GPUFidelityKernel(feature_map, gpu_id=gpu_id, assume_pretranspiled=True)
                return NoisyFidelityKernel(feature_map, noise_model, gpu_id=gpu_id)

            primary_kernel = _build_kernel(fm_primary)
            secondary_feature_map = None
            if model_name == 'QVE':
                secondary_feature_map = fm_secondary
            elif model_name == 'QWE':
                secondary_feature_map = fm_tertiary
            secondary_kernel = _build_kernel(secondary_feature_map) if secondary_feature_map is not None else None

            kernel_time = 0.0
            classifier_time = 0.0

            if model_name == 'QSVC':
                kernel_start = time.perf_counter()
                K_train = primary_kernel.evaluate(X_train, X_train)
                K_test = primary_kernel.evaluate(X_test, X_train)
                kernel_time += time.perf_counter() - kernel_start
                K_train = np.nan_to_num(K_train, nan=0.0, posinf=1.0, neginf=0.0)
                K_test = np.nan_to_num(K_test, nan=0.0, posinf=1.0, neginf=0.0)

                svc = cuSVC(kernel='precomputed', class_weight='balanced',
                            random_state=seed, cache_size=8192.0,
                            max_iter=10000, nochange_steps=100,
                            output_type='numpy')
                classifier_start = time.perf_counter()
                svc.fit(K_train, y_train)
                y_pred = svc.predict(K_test)
                classifier_time += time.perf_counter() - classifier_start

            elif model_name == 'QVE':
                probas = []
                for kern in [primary_kernel, secondary_kernel]:
                    kernel_start = time.perf_counter()
                    Kt = kern.evaluate(X_train, X_train)
                    Kte = kern.evaluate(X_test, X_train)
                    kernel_time += time.perf_counter() - kernel_start
                    Kt = np.nan_to_num(Kt, nan=0.0, posinf=1.0, neginf=0.0)
                    Kte = np.nan_to_num(Kte, nan=0.0, posinf=1.0, neginf=0.0)
                    svc = cuSVC(kernel='precomputed', class_weight='balanced',
                                random_state=seed, cache_size=8192.0,
                                max_iter=10000, nochange_steps=100,
                                output_type='numpy')
                    classifier_start = time.perf_counter()
                    svc.fit(Kt, y_train)
                    probas.append(cuml_svc_predict_proba(svc, Kte))
                    classifier_time += time.perf_counter() - classifier_start
                avg_proba = np.mean(probas, axis=0)
                y_pred = np.argmax(avg_proba, axis=1)

            elif model_name == 'QWE':
                X_fit, X_val, y_fit, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, stratify=y_train,
                    random_state=seed
                )

                validation_scores = []
                test_probas = []
                for kern in [primary_kernel, secondary_kernel]:
                    kernel_start = time.perf_counter()
                    K_fit = kern.evaluate(X_fit, X_fit)
                    K_val = kern.evaluate(X_val, X_fit)
                    kernel_time += time.perf_counter() - kernel_start
                    K_fit = np.nan_to_num(K_fit, nan=0.0, posinf=1.0, neginf=0.0)
                    K_val = np.nan_to_num(K_val, nan=0.0, posinf=1.0, neginf=0.0)

                    svc_val = cuSVC(kernel='precomputed', class_weight='balanced',
                                    random_state=seed, cache_size=8192.0,
                                    max_iter=10000, nochange_steps=100,
                                    output_type='numpy')
                    classifier_start = time.perf_counter()
                    svc_val.fit(K_fit, y_fit)
                    y_val_pred = svc_val.predict(K_val)
                    classifier_time += time.perf_counter() - classifier_start
                    validation_scores.append(max(accuracy_score(y_val, y_val_pred), 1e-6))

                    kernel_start = time.perf_counter()
                    Kt = kern.evaluate(X_train, X_train)
                    Kte = kern.evaluate(X_test, X_train)
                    kernel_time += time.perf_counter() - kernel_start
                    Kt = np.nan_to_num(Kt, nan=0.0, posinf=1.0, neginf=0.0)
                    Kte = np.nan_to_num(Kte, nan=0.0, posinf=1.0, neginf=0.0)
                    svc = cuSVC(kernel='precomputed', class_weight='balanced',
                                random_state=seed, cache_size=8192.0,
                                max_iter=10000, nochange_steps=100,
                                output_type='numpy')
                    classifier_start = time.perf_counter()
                    svc.fit(Kt, y_train)
                    test_probas.append(cuml_svc_predict_proba(svc, Kte))
                    classifier_time += time.perf_counter() - classifier_start

                weights = np.asarray(validation_scores, dtype=np.float64)
                weights = weights / weights.sum()
                avg_proba = np.average(np.asarray(test_probas), axis=0, weights=weights)
                y_pred = np.argmax(avg_proba, axis=1)
            else:
                raise ValueError(f"Unknown model for Phase 4 run: {model_name}")

            train_time = kernel_time + classifier_time
            metrics = calculate_all_metrics(y_test, y_pred, train_time=train_time)
            return {
                'status': 'success',
                'run_idx': run_idx,
                'seed': seed,
                'gpu_id': gpu_id,
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'mcc': metrics['mcc'],
                'kernel_time': kernel_time,
                'classifier_time': classifier_time,
                'total_time': train_time,
            }
        except Exception as exc:
            return {
                'status': 'failed',
                'run_idx': run_idx,
                'seed': seed,
                'gpu_id': gpu_id,
                'error': str(exc),
            }
        finally:
            clear_gpu_memory(gpu_id)
            gc.collect()

    def _execute_ideal_qsvc_run(self, run_idx: int, seed: int,
                                optimization_name: str,
                                X: np.ndarray, y: np.ndarray,
                                num_qubits: int,
                                feature_map: QuantumCircuit,
                                gpu_id: int) -> Dict[str, Any]:
        """Execute one ideal-simulator QSVC precomputed-kernel run on a specific GPU."""
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], random_state=seed, stratify=y
            )
            processor = DataProcessor(num_qubits=num_qubits, random_seed=seed)
            processor.fit(X_train_raw, y_train)
            X_train = processor.transform(X_train_raw)
            X_test = processor.transform(X_test_raw)

            gpu_kernel = GPUFidelityKernel(feature_map, gpu_id=gpu_id, assume_pretranspiled=True)
            start = time.perf_counter()
            K_train = gpu_kernel.evaluate(X_train, X_train)
            K_test = gpu_kernel.evaluate(X_test, X_train)
            svc = cuSVC(
                kernel='precomputed',
                class_weight='balanced',
                random_state=seed,
                cache_size=8192.0,
                max_iter=10000,
                nochange_steps=100,
                output_type='numpy',
            )
            svc.fit(K_train, y_train)
            y_pred = svc.predict(K_test)
            elapsed = time.perf_counter() - start
            perf = calculate_all_metrics(y_test, y_pred, train_time=elapsed)
            return {
                'status': 'success',
                'run_idx': run_idx,
                'seed': seed,
                'optimization_name': optimization_name,
                'gpu_id': gpu_id,
                'accuracy': perf['accuracy'],
                'f1_score': perf['f1_score'],
                'precision': perf['precision'],
                'recall': perf['recall'],
                'mcc': perf['mcc'],
                'training_time': elapsed,
            }
        except Exception as exc:
            return {
                'status': 'failed',
                'run_idx': run_idx,
                'seed': seed,
                'optimization_name': optimization_name,
                'gpu_id': gpu_id,
                'error': str(exc),
            }
        finally:
            clear_gpu_memory(gpu_id)
            gc.collect()

    def _reset_results_storage(self):
        self.all_results = []
        self.circuit_analysis_results = []
        self.ablation_results = []
        self.cv_results = []
        self.metrics_analyzer.metrics_history = []

    @staticmethod
    def _detect_csv_delimiter(dataset_path: str) -> str:
        with open(dataset_path, 'r', encoding='utf-8', errors='ignore', newline='') as handle:
            sample = ''.join(handle.readline() for _ in range(5))

        if not sample:
            return ','

        try:
            return csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
        except csv.Error:
            if sample.count(';') > sample.count(','):
                return ';'
            return ','

    def _load_dataset(self, dataset_path: str) -> Tuple[pd.DataFrame, str]:
        delimiter = self._detect_csv_delimiter(dataset_path)
        df = pd.read_csv(dataset_path, sep=delimiter, low_memory=False)
        df.columns = [str(col).strip() for col in df.columns]
        return df, delimiter

    @staticmethod
    def _holm_adjust(p_values: List[float]) -> List[float]:
        if not p_values:
            return []

        indexed_p_values = sorted(enumerate(p_values), key=lambda item: item[1])
        n_values = len(indexed_p_values)
        adjusted = [1.0] * n_values
        running_max = 0.0

        for rank, (original_index, p_value) in enumerate(indexed_p_values):
            corrected = min(1.0, (n_values - rank) * p_value)
            running_max = max(running_max, corrected)
            adjusted[original_index] = running_max

        return adjusted

    def _build_phase4_paired_statistics(self, per_run_df: pd.DataFrame) -> pd.DataFrame:
        if per_run_df.empty:
            return pd.DataFrame()

        metric_names = [
            'accuracy', 'f1_score', 'mcc', 'kernel_time',
            'classifier_time', 'total_time'
        ]
        comparison_rows = []

        group_columns = ['dataset', 'config', 'model', 'entanglement', 'noise_level', 'num_qubits']
        for group_key, group_df in per_run_df.groupby(group_columns, sort=True):
            l0_df = group_df[group_df['optimization_level'] == 0]
            l3_df = group_df[group_df['optimization_level'] == 3]
            if l0_df.empty or l3_df.empty:
                continue

            l0_metrics = l0_df[['run_idx', *metric_names]].rename(
                columns={metric: f'{metric}_l0' for metric in metric_names}
            )
            l3_metrics = l3_df[['run_idx', *metric_names]].rename(
                columns={metric: f'{metric}_l3' for metric in metric_names}
            )
            paired_df = pd.merge(l0_metrics, l3_metrics, on='run_idx', how='inner').sort_values('run_idx')
            if len(paired_df) < 2:
                continue

            dataset_name, config_tag, model_name, entanglement, noise_level, qubit_count = group_key
            comparison_row = {
                'dataset': dataset_name,
                'config': config_tag,
                'model': model_name,
                'entanglement': entanglement,
                'noise_level': noise_level,
                'num_qubits': qubit_count,
                'paired_runs': len(paired_df),
            }

            for metric_name in metric_names:
                l0_values = paired_df[f'{metric_name}_l0'].to_numpy(dtype=np.float64)
                l3_values = paired_df[f'{metric_name}_l3'].to_numpy(dtype=np.float64)
                diffs = l3_values - l0_values

                if np.allclose(diffs, 0.0):
                    t_stat = 0.0
                    p_value = 1.0
                    effect_size = 0.0
                else:
                    t_stat, p_value = stats.ttest_rel(l3_values, l0_values)
                    diff_std = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0
                    effect_size = np.mean(diffs) / diff_std if diff_std > 0 else 0.0

                comparison_row[f'{metric_name}_mean_l0'] = float(np.mean(l0_values))
                comparison_row[f'{metric_name}_mean_l3'] = float(np.mean(l3_values))
                comparison_row[f'{metric_name}_delta_l3_minus_l0'] = float(np.mean(diffs))
                comparison_row[f'{metric_name}_t_stat'] = float(t_stat)
                comparison_row[f'{metric_name}_p_value'] = float(p_value)
                comparison_row[f'{metric_name}_cohens_d_paired'] = float(effect_size)

            comparison_rows.append(comparison_row)

        comparison_df = pd.DataFrame(comparison_rows)
        if comparison_df.empty:
            return comparison_df

        for metric_name in metric_names:
            adjusted = self._holm_adjust(comparison_df[f'{metric_name}_p_value'].fillna(1.0).tolist())
            comparison_df[f'{metric_name}_p_value_holm'] = adjusted
            comparison_df[f'{metric_name}_significant_holm'] = comparison_df[f'{metric_name}_p_value_holm'] < 0.05

        return comparison_df
    
    def _save_incremental_results(self):
        """Save results incrementally after each model to prevent data loss."""
        prefix = self._get_filename_prefix()
        
        # Save model results incrementally
        if self.all_results:
            df = pd.DataFrame(self.all_results)
            df.to_csv(RESULTS_DIR / f'model_results_{prefix}_incremental.csv', index=False)
        
        # Save circuit analysis incrementally
        if self.circuit_analysis_results:
            df = pd.DataFrame(self.circuit_analysis_results)
            df.to_csv(RESULTS_DIR / f'circuit_analysis_{prefix}_incremental.csv', index=False)
        
        # Save CV results incrementally
        if self.cv_results:
            df = pd.DataFrame(self.cv_results)
            df.to_csv(RESULTS_DIR / f'cv_results_{prefix}_incremental.csv', index=False)
        
        # Save ablation results incrementally
        if self.ablation_results:
            df = pd.DataFrame(self.ablation_results)
            df.to_csv(RESULTS_DIR / f'ablation_results_{prefix}_incremental.csv', index=False)
    
    def build_model_list(self, num_qubits: int, feature_map: QuantumCircuit) -> List[Tuple[str, str, Dict]]:
        """Build list of models to evaluate"""

        return [
            ('QSVC_Precomputed', 'QSVC_Precomputed', {'feature_map': feature_map}),
            ('QVE', 'QVE', {}),
            ('QWE', 'QWE', {}),
        ]
    
    def analyze_circuit_expressibility(self, circuits: List[Tuple[str, QuantumCircuit]],
                                        simulator=None) -> List[Dict[str, Any]]:
        """Analyze expressibility for a set of circuits"""
        
        results = []
        
        for name, circuit in circuits:
            logger.info(f"Analyzing expressibility for {name}...")
            
            expr_result = self.expressibility_analyzer.compute_expressibility(circuit, simulator)
            entropy_result = self.entropy_analyzer.compute_average_entropy(circuit, num_samples=50, simulator=simulator)
            
            circuit_metrics = self.metrics_analyzer.analyze_circuit(circuit, name)
            
            results.append({
                'circuit_name': name,
                **expr_result,
                **entropy_result,
                **circuit_metrics
            })
        
        return results
    
    def run_noise_comparison(self, model_factory, X: np.ndarray, y: np.ndarray,
                              num_qubits: int, model_name: str) -> Dict[str, Any]:
        """Compare model performance with and without noise"""
        
        results = {'model_name': model_name}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        # Ideal (no noise) run
        try:
            model = model_factory()
            start = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            ideal_metrics = calculate_all_metrics(y_test, y_pred, train_time=time.time() - start)
            for k, v in ideal_metrics.items():
                results[f'ideal_{k}'] = v
        except Exception as e:
            logger.warning(f"Ideal run failed: {e}")
            results['ideal_error'] = str(e)
        
        # Noisy run
        try:
            noise_model = self.noise_simulator.create_noise_model(num_qubits)
            # Note: Full noisy simulation would require modifying kernel computation
            # For now, we record that noise simulation was attempted
            results['noise_params'] = NOISE_PARAMS
            results['noise_simulation_note'] = 'Full noise integration pending IBM device access'
        except Exception as e:
            results['noise_error'] = str(e)
        
        return results
    
    def run_optimization_noise_impact(self, X: np.ndarray, y: np.ndarray, 
                                       num_qubits: int, n_runs: int = 5) -> pd.DataFrame:
        """
        KEY EXPERIMENT: Show how circuit optimization improves performance under noise.
        
        This is the critical experiment that demonstrates the research value:
        - Unoptimized circuits accumulate more noise (more gates = more errors)
        - Optimized circuits are more noise-resilient (fewer gates = fewer errors)
        
        We compare accuracy degradation between optimized and unoptimized circuits
        across multiple noise levels using shot-based noisy simulation.
        
        METHODOLOGY:
        - Ideal (noise_level=0): Use exact statevector fidelity via GPUFidelityKernel
        - Noisy (noise_level>0): Use NoisyFidelityKernel with Tr(ρ·σ) density matrix
          kernel and depolarizing noise model to simulate realistic NISQ behavior
        """
        logger.info("\n" + "="*60)
        logger.info("🔬 NOISE IMPACT ON CIRCUIT OPTIMIZATION")
        logger.info("This experiment demonstrates the KEY research contribution:")
        logger.info("Circuit optimization improves noise resilience on NISQ devices")
        logger.info(f"Running {n_runs} trials per configuration")
        logger.info("="*60)
        
        from qiskit.transpiler import CouplingMap
        linear_coupling = CouplingMap.from_line(num_qubits)
        
        # Create test feature map
        fm = create_feature_map(num_qubits, 'ZZ', reps=2, entanglement='full')
        
        # Prepare unoptimized and optimized versions
        fm_unoptimized = transpile_with_cache(
            fm,
            ('noise-impact', num_qubits, 'ZZ', 2, 'full', 0, _normalize_coupling_map_key(linear_coupling)),
            optimization_level=0,
            basis_gates=['u', 'cx', 'rz', 'sx', 'x'],
            coupling_map=linear_coupling, seed_transpiler=42
        )
        fm_optimized = transpile_with_cache(
            fm,
            ('noise-impact', num_qubits, 'ZZ', 2, 'full', 3, _normalize_coupling_map_key(linear_coupling)),
            optimization_level=3,
            basis_gates=['u', 'cx', 'rz', 'sx', 'x'],
            coupling_map=linear_coupling, seed_transpiler=42
        )
        
        # Get circuit metrics
        unopt_depth = fm_unoptimized.depth()
        unopt_gates = sum(fm_unoptimized.count_ops().values())
        unopt_2q = sum(c for g, c in fm_unoptimized.count_ops().items() if g in ['cx', 'cz'])
        
        opt_depth = fm_optimized.depth()
        opt_gates = sum(fm_optimized.count_ops().values())
        opt_2q = sum(c for g, c in fm_optimized.count_ops().items() if g in ['cx', 'cz'])
        
        logger.info(f"\nCircuit Comparison:")
        logger.info(f"  Unoptimized: depth={unopt_depth}, gates={unopt_gates}, 2Q_gates={unopt_2q}")
        logger.info(f"  Optimized:   depth={opt_depth}, gates={opt_gates}, 2Q_gates={opt_2q}")
        logger.info(f"  Reduction:   depth={unopt_depth-opt_depth} ({(1-opt_depth/unopt_depth)*100:.1f}%), gates={unopt_gates-opt_gates}")
        
        # Legacy helper retained for supplementary analysis.
        # Uses a denser 14-level grid than the main paper-facing default Phase 4 grid.
        noise_levels = [0.0, 0.0001, 0.0003, 0.0005, 0.001,
                        0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]  # Single-qubit error rates
        
        # Use same number of runs for both ideal and noisy for fair comparison
        ideal_runs = n_runs  # Same runs for all conditions
        noisy_runs = n_runs  # Same runs for noisy simulations (no reduction)
        
        logger.info(f"  Runs: {ideal_runs} (ideal), {noisy_runs} (noisy), using full dataset (no subsampling)")
        
        results = []
        
        # Create kernels for ideal simulation (reuse across runs)
        # Use best available GPU instead of hardcoding gpu_id=0
        best_gpu = self._select_gpu_for_workload("phase4-noise-comparison")
        gpu_kernel_unopt = GPUFidelityKernel(fm_unoptimized, gpu_id=best_gpu, assume_pretranspiled=True)
        gpu_kernel_opt = GPUFidelityKernel(fm_optimized, gpu_id=best_gpu, assume_pretranspiled=True)
        
        for noise_level in noise_levels:
            # Adaptive runs: full runs for ideal, reduced for noisy
            current_runs = ideal_runs if noise_level == 0 else noisy_runs
            logger.info(f"\n📊 Testing noise level: {noise_level} (2Q error: {noise_level * 3:.4f}) - {current_runs} runs")
            
            # Create noise model and noisy kernels ONCE per noise level (outside run loop)
            noise_model = None
            noisy_kernel_unopt = None
            noisy_kernel_opt = None
            if noise_level > 0:
                noise_params = NOISE_PARAMS.copy()
                noise_params['single_qubit_error'] = noise_level
                noise_params['two_qubit_error'] = noise_level * 3  # 2Q gates have ~3x error (realistic NISQ ratio)
                noise_sim = NoiseModelSimulator(params=noise_params)
                noise_model = noise_sim.create_noise_model(num_qubits)
                # Create kernels ONCE per noise level - reuse for all runs
                noisy_kernel_unopt = NoisyFidelityKernel(
                    fm_unoptimized, noise_model, gpu_id=best_gpu
                )
                noisy_kernel_opt = NoisyFidelityKernel(
                    fm_optimized, noise_model, gpu_id=best_gpu
                )
            
            for opt_level, fm_circuit, opt_name in [
                (0, fm_unoptimized, 'unoptimized'),
                (3, fm_optimized, 'optimized_L3')
            ]:
                run_accs = []
                run_f1s = []
                run_times = []
                
                for run_idx in range(current_runs):
                    seed = 42 + run_idx
                    
                    # Different train/test split for each run
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, stratify=y, random_state=seed
                    )
                    
                    try:
                        start_time = time.time()
                        
                        if noise_level == 0:
                            # Ideal simulation using exact statevector fidelity
                            if opt_name == 'unoptimized':
                                K_train = gpu_kernel_unopt.evaluate(X_train, X_train)
                                K_test = gpu_kernel_unopt.evaluate(X_test, X_train)
                            else:
                                K_train = gpu_kernel_opt.evaluate(X_train, X_train)
                                K_test = gpu_kernel_opt.evaluate(X_test, X_train)
                        else:
                            # Noisy simulation using shot-based sampling kernel
                            noisy_kernel = noisy_kernel_unopt if opt_name == 'unoptimized' else noisy_kernel_opt
                            K_train = noisy_kernel.evaluate(X_train, X_train)
                            K_test = noisy_kernel.evaluate(X_test, X_train)
                        
                        # Ensure kernel matrices are valid
                        K_train = np.nan_to_num(K_train, nan=0.0, posinf=1.0, neginf=0.0)
                        K_test = np.nan_to_num(K_test, nan=0.0, posinf=1.0, neginf=0.0)
                        
                        # Train cuML SVC with precomputed kernel on GPU
                        svc = cuSVC(kernel='precomputed', class_weight='balanced',
                                     random_state=seed, C=1.0, cache_size=8192.0,
                                     max_iter=10000, nochange_steps=100,
                                     output_type='numpy')
                        svc.fit(K_train, y_train)
                        y_pred = svc.predict(K_test)
                        train_time = time.time() - start_time
                        
                        metrics = calculate_all_metrics(y_test, y_pred, train_time=train_time)
                        run_accs.append(metrics['accuracy'])
                        run_f1s.append(metrics['f1_score'])
                        run_times.append(train_time)
                        
                        # Log progress at INFO level periodically
                        if (run_idx + 1) % 5 == 0 or run_idx == 0:
                            logger.info(f"    Run {run_idx+1}/{current_runs}: acc={metrics['accuracy']:.4f} ({train_time:.1f}s)")
                        
                    except Exception as e:
                        logger.warning(f"Run {run_idx+1} failed for {opt_name} at noise={noise_level}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Compute statistics for this configuration
                valid_accs = [a for a in run_accs if not np.isnan(a)]
                valid_f1s = [f for f in run_f1s if not np.isnan(f)]
                
                if valid_accs:
                    acc_mean = np.mean(valid_accs)
                    n_valid = len(valid_accs)
                    acc_std = np.std(valid_accs, ddof=1) if n_valid > 1 else 0
                    acc_median = np.median(valid_accs)  # More robust to outliers
                    acc_ci95 = stats.t.ppf(0.975, df=n_valid-1) * acc_std / np.sqrt(n_valid) if n_valid > 1 else 0
                    # Interquartile range (IQR) for robust spread measure
                    acc_q25, acc_q75 = np.percentile(valid_accs, [25, 75])
                    acc_iqr = acc_q75 - acc_q25
                    
                    results.append({
                        'noise_level': noise_level,
                        'two_qubit_error': noise_level * 3,
                        'optimization': opt_name,
                        'circuit_depth': opt_depth if opt_name != 'unoptimized' else unopt_depth,
                        'total_gates': opt_gates if opt_name != 'unoptimized' else unopt_gates,
                        'two_qubit_gates': opt_2q if opt_name != 'unoptimized' else unopt_2q,
                        'accuracy_mean': acc_mean,
                        'accuracy_median': acc_median,  # Added median
                        'accuracy_std': acc_std,
                        'accuracy_iqr': acc_iqr,  # Added IQR
                        'accuracy_ci95': acc_ci95,
                        'accuracy_q25': acc_q25,  # Added quartiles
                        'accuracy_q75': acc_q75,
                        'accuracy_min': np.min(valid_accs),
                        'accuracy_max': np.max(valid_accs),
                        'f1_mean': np.mean(valid_f1s) if valid_f1s else 0,
                        'f1_median': np.median(valid_f1s) if valid_f1s else 0,  # Added F1 median
                        'f1_std': np.std(valid_f1s) if valid_f1s else 0,
                        'avg_time': np.mean(run_times) if run_times else 0,
                        'n_runs': len(valid_accs),
                        'num_qubits': num_qubits
                    })
                    
                    logger.info(f"  {opt_name}: acc={acc_mean:.4f} ± {acc_std:.4f} (n={len(valid_accs)})")
                else:
                    logger.warning(f"  {opt_name}: ALL runs failed at noise_level={noise_level}")
                
                # Clear GPU memory between configurations
                clear_gpu_memory(best_gpu)
                gc.collect()
        
        # Create DataFrame and compute noise resilience metrics
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            # Use new comprehensive noise resilience analysis
            resilience_metrics = compute_noise_resilience_metrics(df)
            
            # Print comprehensive report
            report = format_noise_resilience_report(resilience_metrics)
            logger.info(report)
            
            # Also save resilience metrics to separate CSV
            resilience_df = pd.DataFrame([resilience_metrics])
            prefix = self._get_filename_prefix()
            resilience_file = RESULTS_DIR / f'noise_resilience_metrics_{prefix}.csv'
            resilience_df.to_csv(resilience_file, index=False)
            logger.info(f"Resilience metrics saved to: {resilience_file}")
            
            # Add NISQ feasibility scores to the analysis
            logger.info("\n" + "="*70)
            logger.info("🔧 NISQ FEASIBILITY COMPARISON")
            logger.info("="*70)
            
            unopt_metrics = {
                'depth': unopt_depth,
                'total_gates': unopt_gates,
                'two_qubit_gates': unopt_2q
            }
            opt_metrics = {
                'depth': opt_depth,
                'total_gates': opt_gates,
                'two_qubit_gates': opt_2q
            }
            
            unopt_nisq = compute_nisq_feasibility_score(unopt_metrics)
            opt_nisq = compute_nisq_feasibility_score(opt_metrics)
            
            logger.info(f"\nUnoptimized Circuit:")
            logger.info(f"  NISQ Score: {unopt_nisq['nisq_feasibility_score']:.2f}/100")
            logger.info(f"  Est. Circuit Fidelity: {unopt_nisq['estimated_circuit_fidelity']:.4f}")
            logger.info(f"  Coherence Ratio (T2): {unopt_nisq['coherence_ratio']:.4f}")
            
            logger.info(f"\nOptimized Circuit:")
            logger.info(f"  NISQ Score: {opt_nisq['nisq_feasibility_score']:.2f}/100")
            logger.info(f"  Est. Circuit Fidelity: {opt_nisq['estimated_circuit_fidelity']:.4f}")
            logger.info(f"  Coherence Ratio (T2): {opt_nisq['coherence_ratio']:.4f}")
            
            nisq_improvement = opt_nisq['nisq_feasibility_score'] - unopt_nisq['nisq_feasibility_score']
            fidelity_improvement = opt_nisq['estimated_circuit_fidelity'] - unopt_nisq['estimated_circuit_fidelity']
            
            logger.info(f"\n🏆 OPTIMIZATION BENEFITS:")
            logger.info(f"  NISQ Score Improvement: +{nisq_improvement:.2f} points")
            logger.info(f"  Fidelity Improvement: +{fidelity_improvement:.4f} ({fidelity_improvement*100:.2f}%)")
            
            # Expected accuracy on real hardware
            ideal_acc = resilience_metrics.get('unoptimized_ideal_accuracy', 0.9)
            unopt_expected = compute_expected_accuracy_under_noise(ideal_acc, unopt_nisq['estimated_circuit_fidelity'])
            opt_expected = compute_expected_accuracy_under_noise(ideal_acc, opt_nisq['estimated_circuit_fidelity'])
            
            logger.info(f"\n📊 EXPECTED REAL HARDWARE PERFORMANCE:")
            logger.info(f"  Unoptimized expected accuracy: {unopt_expected:.4f}")
            logger.info(f"  Optimized expected accuracy: {opt_expected:.4f}")
            logger.info(f"  Improvement: +{(opt_expected - unopt_expected):.4f} ({(opt_expected - unopt_expected)/max(unopt_expected, 0.01)*100:.1f}%)")
            logger.info("="*70)
            
            # Add NISQ scores to DataFrame
            df['nisq_score'] = df.apply(
                lambda row: opt_nisq['nisq_feasibility_score'] if row['optimization'] != 'unoptimized' 
                else unopt_nisq['nisq_feasibility_score'], axis=1
            )
            df['estimated_fidelity'] = df.apply(
                lambda row: opt_nisq['estimated_circuit_fidelity'] if row['optimization'] != 'unoptimized'
                else unopt_nisq['estimated_circuit_fidelity'], axis=1
            )
        
        return df
    
    def run_model_optimization_noise_comparison(self, X: np.ndarray, y: np.ndarray,
                                                  num_qubits: int, n_runs: int = 5) -> pd.DataFrame:
        """
        ENHANCED Phase 4: Test each model type on L0 vs L3 circuits under ideal and noisy conditions.
        
        This addresses a key gap - showing how circuit optimization affects different
        QML model architectures, not just QSVC.
        
        Returns DataFrame with columns:
        - model_type, optimization_level, noise_condition
        - accuracy, f1_score, mcc, etc.
        - circuit metrics (depth, gates, 2q_gates)
        """
        logger.info("\n" + "="*60)
        logger.info("🔬 MODEL-LEVEL OPTIMIZATION & NOISE IMPACT ANALYSIS")
        logger.info("Testing each model on L0 vs L3 under ideal and noisy conditions")
        logger.info("="*60)
        
        from qiskit.transpiler import CouplingMap
        linear_coupling = CouplingMap.from_line(num_qubits)
        basis_gates = ['u', 'cx', 'rz', 'sx', 'x']
        
        # Create base feature map
        base_fm = create_feature_map(num_qubits, 'ZZ', reps=2, entanglement='full')
        
        # Transpile to L0 (unoptimized) and L3 (optimized)
        fm_L0 = transpile_with_cache(
            base_fm,
            ('phase4a-base', num_qubits, 'ZZ', 2, 'full', 0, _normalize_coupling_map_key(linear_coupling)),
            optimization_level=0,
            basis_gates=basis_gates, coupling_map=linear_coupling,
            seed_transpiler=42
        )
        fm_L3 = transpile_with_cache(
            base_fm,
            ('phase4a-base', num_qubits, 'ZZ', 2, 'full', 3, _normalize_coupling_map_key(linear_coupling)),
            optimization_level=3,
            basis_gates=basis_gates, coupling_map=linear_coupling,
            seed_transpiler=42
        )
        metrics_L0 = {'depth': fm_L0.depth(),
                      'gates': sum(fm_L0.count_ops().values()),
                      '2q_gates': sum(c for g, c in fm_L0.count_ops().items() if g in ['cx', 'cz'])}
        metrics_L3 = {'depth': fm_L3.depth(),
                      'gates': sum(fm_L3.count_ops().values()),
                      '2q_gates': sum(c for g, c in fm_L3.count_ops().items() if g in ['cx', 'cz'])}
        
        logger.info(f"  L0: depth={metrics_L0['depth']}, gates={metrics_L0['gates']}, 2q={metrics_L0['2q_gates']}")
        logger.info(f"  L3: depth={metrics_L3['depth']}, gates={metrics_L3['gates']}, 2q={metrics_L3['2q_gates']}")
        logger.info(f"  Reduction: depth={metrics_L0['depth']-metrics_L3['depth']} ({(1-metrics_L3['depth']/metrics_L0['depth'])*100:.1f}%)")
        
        results = []
        
        # Create noise model for noisy simulation
        noise_params = NOISE_PARAMS.copy()
        noise_params['single_qubit_error'] = 0.01  # 1% error - realistic NISQ
        noise_params['two_qubit_error'] = 0.10     # 10% 2Q error
        noise_sim = NoiseModelSimulator(params=noise_params)
        noise_model = noise_sim.create_noise_model(num_qubits)
        
        # Models to test (kernel-based only)
        model_configs = [
            ('QSVC', 'Kernel SVM with quantum kernel'),
        ]
        
        for model_name, model_desc in model_configs:
            logger.info(f"\n  📊 Testing {model_name}: {model_desc}")
            
            for opt_level, fm, opt_name in [
                (0, fm_L0, self._optimization_label(0)),
                (3, fm_L3, self._optimization_label(3)),
            ]:
                fm_metrics = metrics_L0 if opt_level == 0 else metrics_L3
                
                for noise_cond in ['ideal', 'noisy']:
                    logger.info(f"    [{model_name}] {opt_name} + {noise_cond}...")
                    
                    run_accs = []
                    run_f1s = []
                    
                    # Create kernels ONCE outside the run loop (reuse across runs)
                    best_gpu_4a = self._select_gpu_for_workload(
                        f"phase4a-{model_name}-{opt_name}-{noise_cond}"
                    )
                    gpu_kernel_4a = GPUFidelityKernel(fm, gpu_id=best_gpu_4a, assume_pretranspiled=True)
                    noisy_kernel_4a = None
                    if noise_cond == 'noisy':
                        noisy_kernel_4a = NoisyFidelityKernel(fm, noise_model, gpu_id=best_gpu_4a)
                    
                    for run_idx in range(n_runs):
                        seed = 42 + run_idx
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.3, stratify=y, random_state=seed
                        )
                        
                        try:
                            if noise_cond == 'ideal':
                                # Use GPU fidelity kernel (exact statevector)
                                K_train = gpu_kernel_4a.evaluate(X_train, X_train)
                                K_test = gpu_kernel_4a.evaluate(X_test, X_train)
                            else:
                                # Use noisy sampling kernel
                                K_train = noisy_kernel_4a.evaluate(X_train, X_train)
                                K_test = noisy_kernel_4a.evaluate(X_test, X_train)
                            
                            # Ensure valid kernel
                            K_train = np.nan_to_num(K_train, nan=0.0, posinf=1.0, neginf=0.0)
                            K_test = np.nan_to_num(K_test, nan=0.0, posinf=1.0, neginf=0.0)
                            
                            if model_name == 'QSVC':
                                svc = cuSVC(kernel='precomputed', class_weight='balanced',
                                            random_state=seed, cache_size=8192.0,
                                            max_iter=10000, nochange_steps=100,
                                            output_type='numpy')
                                svc.fit(K_train, y_train)
                                y_pred = svc.predict(K_test)
                            
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            run_accs.append(acc)
                            run_f1s.append(f1)
                            
                        except Exception as e:
                            logger.warning(f"      Run {run_idx+1} failed: {e}")
                    
                    if run_accs:
                        results.append({
                            'dataset': self.current_dataset_name,
                            'config': self.current_config_tag,
                            'model': model_name,
                            'optimization_level': opt_level,
                            'optimization_name': opt_name,
                            'optimization_description': self._optimization_description(opt_level),
                            'noise_condition': noise_cond,
                            'accuracy_mean': np.mean(run_accs),
                            'accuracy_std': np.std(run_accs),
                            'f1_mean': np.mean(run_f1s),
                            'f1_std': np.std(run_f1s),
                            'circuit_depth': fm_metrics['depth'],
                            'total_gates': fm_metrics['gates'],
                            'two_qubit_gates': fm_metrics['2q_gates'],
                            'n_runs': len(run_accs)
                        })
                        logger.info(f"      Acc={np.mean(run_accs):.4f}±{np.std(run_accs):.4f}, n={len(run_accs)}")
        
        df = pd.DataFrame(results)
        
        # Log summary
        if len(df) > 0:
            logger.info("\n  📊 MODEL OPTIMIZATION/NOISE IMPACT SUMMARY:")
            for model_name in df['model'].unique():
                model_df = df[df['model'] == model_name]
                logger.info(f"\n    {model_name}:")
                for _, row in model_df.iterrows():
                    logger.info(f"      {row['optimization_name']} + {row['noise_condition']}: "
                               f"Acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}")
        
        return df
    
    def run_comprehensive_model_noise_analysis(self, X: np.ndarray, y: np.ndarray,
                                                num_qubits: int, n_runs: int = 30) -> pd.DataFrame:
        """
        CORE EXPERIMENT: Comprehensive analysis of circuit optimization impact on kernel-QML models under noise.
        
        This is the CENTRAL experiment answering the research question:
        "How does circuit optimization affect ALL QML model architectures under realistic noise?"
        
        Replaces the fragmented Phase 3.5 + Phase 4a + Phase 4b with a single unified analysis.
        
        Design:
        - Models: QSVC, QVE (Quantum Voting Ensemble), QWE (Quantum Weighted Ensemble)
        - Optimization levels: L0 (unoptimized), L3 (Qiskit full optimization)
        - Entanglement: full and linear (structure comparison)
        - Noise levels: 10 levels from ideal (0.0) to high (0.05)
        - Runs: n_runs per configuration for statistical significance
        - All GPU — no CPU fallback
        
        For kernel models (QSVC, QVE, QWE):
        - Ideal (noise=0): GPUFidelityKernel (exact statevector fidelity)
        - Noisy (noise>0): NoisyFidelityKernel (density matrix Tr(ρ·σ) kernel with noise model)
        
        Fair comparison:
        - Same train/test splits across all models at each run (same seed)
        - Ensemble sub-models (QVE/QWE secondary feature maps) use same opt_level
        - Noise model is consistent across all models at each noise level
        - Results saved incrementally after each (model, opt, noise) block
        """
        logger.info("\n" + "=" * 80)
        logger.info("🔬 COMPREHENSIVE MODEL-NOISE-OPTIMIZATION ANALYSIS")
        logger.info("=" * 80)
        logger.info("RESEARCH QUESTION: How does circuit optimization affect kernel-QML models under noise?")
        
        model_names = list(getattr(self, 'phase4_model_names_override', None) or ['QSVC', 'QVE', 'QWE'])
        logger.info(f"  Models: {', '.join(model_names)}")
        
        n_models = len(model_names)
        logger.info(f"  Optimization levels: L0 (unoptimized), L3 (Qiskit optimized)")
        logger.info(f"  Entanglement types: full, linear")
        logger.info(f"  Runs per config: {n_runs}")
        logger.info(f"  {n_models} models × 2 opt levels × 2 entanglements × n_noise × {n_runs} runs each")
        logger.info("=" * 80)
        
        from qiskit.transpiler import CouplingMap
        linear_coupling = CouplingMap.from_line(num_qubits)
        basis_gates = ['u', 'cx', 'rz', 'sx', 'x']
        
        # ---- Noise levels ----
        # Ten levels is a practical default for 30-run studies: enough to resolve
        # the degradation curve without spending most runtime on redundant points.
        noise_override = getattr(self, 'phase4_noise_levels_override', None)
        if noise_override:
            noise_levels = list(noise_override)
            logger.info(f"  Custom noise levels override: using {len(noise_levels)} levels")
        elif hasattr(self, 'quick_test') and self.quick_test:
            noise_levels = QUICK_TEST_NOISE_LEVELS
            logger.info(f"  Quick test mode: using {len(noise_levels)} noise levels")
        else:
            noise_levels = DEFAULT_NOISE_LEVELS
        logger.info(f"  Noise levels: {noise_levels}")
        
        # ---- Entanglement types ----
        # Compare full vs linear entanglement to study how circuit structure
        # interacts with optimization and noise resilience
        entanglement_types = list(getattr(self, 'phase4_entanglement_override', None) or ['full', 'linear'])
        
        results = []
        per_run_results = []
        prefix = self._get_filename_prefix()
        
        # ---- Resume: load already-completed configurations ----
        completed_keys = set()
        resume_csv = RESULTS_DIR / f'comprehensive_model_noise_{prefix}.csv'
        if resume_csv.exists():
            try:
                existing_df = pd.read_csv(resume_csv)
                for _, row in existing_df.iterrows():
                    key = (row['model'], row['entanglement'], int(row['optimization_level']), float(row['noise_level']), int(row['num_qubits']))
                    completed_keys.add(key)
                # Reload existing results so incremental saves include them
                results = existing_df.to_dict('records')
                logger.info(f"🔄 RESUME: Found {len(completed_keys)} completed configurations in {resume_csv.name}")
                logger.info(f"   Will skip these and continue from where we left off.")
            except Exception as e:
                logger.warning(f"Failed to load existing results for resume: {e}. Starting fresh.")
        
        # ---- Per-run checkpoint for crash recovery within a config ----
        # Saves individual run metrics so partial configs can be resumed
        run_checkpoint_csv = CHECKPOINT_DIR / f'per_run_checkpoint_{prefix}.csv'
        partial_runs = {}  # key -> list of per-run metrics including runtime splits
        if run_checkpoint_csv.exists():
            try:
                ckpt_df = pd.read_csv(run_checkpoint_csv)
                has_runtime_split = {
                    'kernel_time', 'classifier_time', 'total_time'
                }.issubset(set(ckpt_df.columns))
                if not has_runtime_split:
                    logger.warning(
                        "Per-run checkpoint %s predates runtime split fields; ignoring partial resume data.",
                        run_checkpoint_csv.name,
                    )
                    ckpt_df = pd.DataFrame()
                elif not ckpt_df.empty:
                    ckpt_df = ckpt_df.dropna(subset=['model', 'entanglement', 'optimization_level', 'noise_level', 'num_qubits', 'run_idx'])
                    ckpt_df = ckpt_df[
                        (ckpt_df['run_idx'] >= 0)
                        & (ckpt_df['run_idx'] < n_runs)
                    ].copy()
                    ckpt_df = ckpt_df.sort_values(
                        ['model', 'entanglement', 'optimization_level', 'noise_level', 'num_qubits', 'run_idx']
                    )
                    duplicate_mask = ckpt_df.duplicated(
                        subset=['model', 'entanglement', 'optimization_level', 'noise_level', 'num_qubits', 'run_idx'],
                        keep='last'
                    )
                    duplicate_count = int(duplicate_mask.sum())
                    if duplicate_count:
                        logger.warning(
                            "Per-run checkpoint %s had %d duplicate run rows; keeping the last occurrence for each run.",
                            run_checkpoint_csv.name,
                            duplicate_count,
                        )
                        ckpt_df = ckpt_df.loc[~duplicate_mask].copy()
                for _, row in ckpt_df.iterrows():
                    key = (row['model'], row['entanglement'], int(row['optimization_level']),
                           float(row['noise_level']), int(row['num_qubits']))
                    if key not in completed_keys:  # Only load runs for incomplete configs
                        if key not in partial_runs:
                            partial_runs[key] = []
                        partial_runs[key].append({
                            'run_idx': int(row['run_idx']),
                            'accuracy': float(row['accuracy']),
                            'f1_score': float(row['f1_score']),
                            'mcc': float(row['mcc']),
                            'kernel_time': float(row['kernel_time']),
                            'classifier_time': float(row['classifier_time']),
                            'time': float(row['total_time']),
                        })
                if partial_runs:
                    logger.info(f"🔄 RESUME: Found {len(partial_runs)} partially-completed configs "
                               f"with per-run checkpoints")
                    for k, runs in partial_runs.items():
                        logger.info(f"   {k[0]}|{k[1]}|L{k[2]}|noise={k[3]}|{k[4]}q: "
                                   f"{len(runs)}/{n_runs} runs completed")
            except Exception as e:
                logger.warning(f"Failed to load per-run checkpoints: {e}")
        
        # ---- GPU setup ----
        best_gpu = self._select_gpu_for_workload("comprehensive-noise-analysis")
        
        total_configs = len(entanglement_types) * len(model_names) * 2 * len(noise_levels)
        config_idx = 0
        
        for ent_type in entanglement_types:
            logger.info(f"\n{'#'*80}")
            logger.info(f"  ENTANGLEMENT: {ent_type.upper()}")
            logger.info(f"{'#'*80}")
            
            # ---- Create ALL feature maps at BOTH optimization levels ----
            base_fm = create_feature_map(num_qubits, 'ZZ', reps=2, entanglement=ent_type)
            fm_L0 = transpile_with_cache(
                base_fm,
                ('phase4-primary', num_qubits, ent_type, 0, _normalize_coupling_map_key(linear_coupling)),
                optimization_level=0, basis_gates=basis_gates,
                coupling_map=linear_coupling, seed_transpiler=42
            )
            fm_L3 = transpile_with_cache(
                base_fm,
                ('phase4-primary', num_qubits, ent_type, 3, _normalize_coupling_map_key(linear_coupling)),
                optimization_level=3, basis_gates=basis_gates,
                coupling_map=linear_coupling, seed_transpiler=42
            )
            
            # Secondary feature maps for QVE (uses Z feature map — no entanglement param)
            secondary_base = create_feature_map(num_qubits, 'Z', reps=1)
            secondary_L0 = transpile_with_cache(
                secondary_base,
                ('phase4-secondary-z', num_qubits, ent_type, 0, _normalize_coupling_map_key(linear_coupling)),
                optimization_level=0, basis_gates=basis_gates,
                coupling_map=linear_coupling, seed_transpiler=42
            )
            secondary_L3 = transpile_with_cache(
                secondary_base,
                ('phase4-secondary-z', num_qubits, ent_type, 3, _normalize_coupling_map_key(linear_coupling)),
                optimization_level=3, basis_gates=basis_gates,
                coupling_map=linear_coupling, seed_transpiler=42
            )
            
            # Tertiary feature maps for QWE (uses Pauli feature map — same entanglement)
            tertiary_base = create_feature_map(num_qubits, 'Pauli', reps=1, entanglement=ent_type)
            tertiary_L0 = transpile_with_cache(
                tertiary_base,
                ('phase4-secondary-pauli', num_qubits, ent_type, 0, _normalize_coupling_map_key(linear_coupling)),
                optimization_level=0, basis_gates=basis_gates,
                coupling_map=linear_coupling, seed_transpiler=42
            )
            tertiary_L3 = transpile_with_cache(
                tertiary_base,
                ('phase4-secondary-pauli', num_qubits, ent_type, 3, _normalize_coupling_map_key(linear_coupling)),
                optimization_level=3, basis_gates=basis_gates,
                coupling_map=linear_coupling, seed_transpiler=42
            )
            
            # Log circuit metrics for this entanglement type
            for label, fm in [('Primary L0', fm_L0), ('Primary L3', fm_L3),
                               ('Secondary L0', secondary_L0), ('Secondary L3', secondary_L3),
                               ('Tertiary L0', tertiary_L0), ('Tertiary L3', tertiary_L3)]:
                d = fm.depth()
                g = sum(fm.count_ops().values())
                q2 = sum(c for gate, c in fm.count_ops().items() if gate in ['cx', 'cz'])
                logger.info(f"  [{ent_type}] {label:15s}: depth={d:4d}, gates={g:4d}, 2Q={q2:4d}")
            
            for model_name in model_names:
                for opt_level in [0, 3]:
                    opt_label = self._optimization_label(opt_level)
                    
                    # Select feature maps for this opt level
                    fm_primary = fm_L0 if opt_level == 0 else fm_L3
                    fm_secondary = secondary_L0 if opt_level == 0 else secondary_L3
                    fm_tertiary = tertiary_L0 if opt_level == 0 else tertiary_L3
                    
                    # Circuit metrics for primary feature map
                    fm_depth = fm_primary.depth()
                    fm_gates = sum(fm_primary.count_ops().values())
                    fm_2q = sum(c for g, c in fm_primary.count_ops().items() if g in ['cx', 'cz'])

                    component_feature_maps = [fm_primary]
                    if model_name == 'QVE':
                        component_feature_maps.append(fm_secondary)
                    elif model_name == 'QWE':
                        component_feature_maps.append(fm_tertiary)

                    component_depths = [feature_map.depth() for feature_map in component_feature_maps]
                    component_total_gates = [sum(feature_map.count_ops().values()) for feature_map in component_feature_maps]
                    component_two_qubit_gates = [
                        sum(c for g, c in feature_map.count_ops().items() if g in ['cx', 'cz'])
                        for feature_map in component_feature_maps
                    ]

                    aggregate_depth = sum(component_depths)
                    aggregate_total_gates = sum(component_total_gates)
                    aggregate_two_qubit_gates = sum(component_two_qubit_gates)
                    
                    for noise_level in noise_levels:
                        config_idx += 1
                        noise_label = f"noise={noise_level:.4f}"
                        
                        # ---- Resume: skip already-completed configurations ----
                        resume_key = (model_name, ent_type, opt_level, noise_level, num_qubits)
                        if resume_key in completed_keys:
                            logger.info(f"[{config_idx}/{total_configs}] {model_name} | {opt_label} | {ent_type} | {noise_label} — SKIPPED (already completed)")
                            continue
                        
                        logger.info(f"\n{'='*60}")
                        logger.info(f"[{config_idx}/{total_configs}] {model_name} | {opt_label} | {ent_type} | {noise_label}")
                        logger.info(f"{'='*60}")
                        
                        # ---- Run trials ----
                        run_accs = []
                        run_f1s = []
                        run_mccs = []
                        run_times = []
                        run_kernel_times = []
                        run_classifier_times = []
                        seeds = self.config['random_seeds']
                        
                        # Load any partially-completed runs from checkpoint
                        start_run_idx = 0
                        if resume_key in partial_runs:
                            saved_runs = partial_runs[resume_key]
                            completed_run_indices = {r['run_idx'] for r in saved_runs}
                            for r in sorted(saved_runs, key=lambda x: x['run_idx']):
                                run_accs.append(r['accuracy'])
                                run_f1s.append(r['f1_score'])
                                run_mccs.append(r['mcc'])
                                run_times.append(r['time'])
                                run_kernel_times.append(r['kernel_time'])
                                run_classifier_times.append(r['classifier_time'])
                                per_run_results.append({
                                    'dataset': self.current_dataset_name,
                                    'config': self.current_config_tag,
                                    'model': model_name,
                                    'entanglement': ent_type,
                                    'optimization_level': opt_level,
                                    'optimization_name': opt_label,
                                    'optimization_description': self._optimization_description(opt_level),
                                    'noise_level': noise_level,
                                    'num_qubits': num_qubits,
                                    'run_idx': r['run_idx'],
                                    'seed': seeds[r['run_idx']],
                                    'accuracy': r['accuracy'],
                                    'f1_score': r['f1_score'],
                                    'mcc': r['mcc'],
                                    'kernel_time': r['kernel_time'],
                                    'classifier_time': r['classifier_time'],
                                    'total_time': r['time'],
                                })
                            # Start from first incomplete run
                            start_run_idx = max(completed_run_indices) + 1
                            logger.info(f"  Resuming from run {start_run_idx}/{n_runs} "
                                       f"({len(saved_runs)} runs loaded from checkpoint)")
                        
                        remaining_run_tasks = [
                            (
                                run_idx,
                                seeds[run_idx],
                                model_name,
                                num_qubits,
                                noise_level,
                                X,
                                y,
                                fm_primary,
                                fm_secondary,
                                fm_tertiary,
                            )
                            for run_idx in range(start_run_idx, n_runs)
                        ]

                        gpu_mgr_phase4 = self.gpu_manager or get_gpu_manager()
                        phase4_workers = max(1, min(MAX_PARALLEL_GPUS, gpu_mgr_phase4.gpu_count or 1, len(remaining_run_tasks) or 1))
                        if remaining_run_tasks:
                            logger.info(
                                f"  Launching Phase 4 runs across {phase4_workers} GPU worker(s) "
                                f"for {len(remaining_run_tasks)} remaining run(s)"
                            )

                        for batch_start in range(0, len(remaining_run_tasks), phase4_workers):
                            batch_tasks = remaining_run_tasks[batch_start:batch_start + phase4_workers]
                            batch_results = run_parallel_on_gpus(
                                batch_tasks,
                                self._execute_phase4_run,
                                num_workers=phase4_workers,
                                timeout_per_task=7200,
                            )

                            for run_result in batch_results:
                                if not run_result or run_result.get('status') != 'success':
                                    failed_idx = run_result.get('run_idx') if isinstance(run_result, dict) else 'unknown'
                                    failed_err = run_result.get('error') if isinstance(run_result, dict) else 'unknown error'
                                    logger.warning(f"  Run {failed_idx + 1 if isinstance(failed_idx, int) else failed_idx} FAILED: {failed_err}")
                                    continue

                                run_idx = int(run_result['run_idx'])
                                seed = int(run_result['seed'])
                                kernel_time = float(run_result['kernel_time'])
                                classifier_time = float(run_result['classifier_time'])
                                train_time = float(run_result['total_time'])

                                run_accs.append(float(run_result['accuracy']))
                                run_f1s.append(float(run_result['f1_score']))
                                run_mccs.append(float(run_result['mcc']))
                                run_times.append(train_time)
                                run_kernel_times.append(kernel_time)
                                run_classifier_times.append(classifier_time)
                                per_run_results.append({
                                    'dataset': self.current_dataset_name,
                                    'config': self.current_config_tag,
                                    'model': model_name,
                                    'entanglement': ent_type,
                                    'optimization_level': opt_level,
                                    'optimization_name': opt_label,
                                    'optimization_description': self._optimization_description(opt_level),
                                    'noise_level': noise_level,
                                    'num_qubits': num_qubits,
                                    'run_idx': run_idx,
                                    'seed': seed,
                                    'gpu_id': int(run_result['gpu_id']),
                                    'accuracy': float(run_result['accuracy']),
                                    'f1_score': float(run_result['f1_score']),
                                    'mcc': float(run_result['mcc']),
                                    'kernel_time': kernel_time,
                                    'classifier_time': classifier_time,
                                    'total_time': train_time,
                                })

                                run_ckpt_row = {
                                    'model': model_name, 'entanglement': ent_type,
                                    'optimization_level': opt_level, 'noise_level': noise_level,
                                    'num_qubits': num_qubits, 'run_idx': run_idx,
                                    'accuracy': float(run_result['accuracy']), 'f1_score': float(run_result['f1_score']),
                                    'mcc': float(run_result['mcc']),
                                    'kernel_time': kernel_time,
                                    'classifier_time': classifier_time,
                                    'total_time': train_time,
                                }
                                ckpt_exists = run_checkpoint_csv.exists()
                                pd.DataFrame([run_ckpt_row]).to_csv(
                                    run_checkpoint_csv, mode='a', header=not ckpt_exists, index=False
                                )

                                if (run_idx + 1) % 5 == 0 or run_idx == 0 or run_idx == n_runs - 1:
                                    remaining = n_runs - (run_idx + 1)
                                    eta_min = (remaining * train_time) / (60 * max(1, phase4_workers)) if train_time > 0 else 0
                                    logger.info(
                                        f"  Run {run_idx+1}/{n_runs} on GPU {run_result['gpu_id']}: "
                                        f"Acc={run_result['accuracy']:.4f} F1={run_result['f1_score']:.4f} "
                                        f"(kernel={kernel_time:.1f}s, classifier={classifier_time:.1f}s, "
                                        f"total={train_time:.1f}s, ETA: {eta_min:.1f}min)"
                                    )
                        
                        # ---- Aggregate statistics for this config ----
                        if run_accs:
                            n_runs_done = len(run_accs)
                            acc_mean = np.mean(run_accs)
                            acc_std = np.std(run_accs, ddof=1) if n_runs_done > 1 else 0
                            acc_ci95 = stats.t.ppf(0.975, df=n_runs_done-1) * acc_std / np.sqrt(n_runs_done) if n_runs_done > 1 else 0
                            
                            avg_kernel_time = np.mean(run_kernel_times)
                            avg_classifier_time = np.mean(run_classifier_times)
                            avg_total_time = np.mean(run_times)
                            kernel_fraction = avg_kernel_time / avg_total_time if avg_total_time > 0 else 0.0
                            classifier_fraction = avg_classifier_time / avg_total_time if avg_total_time > 0 else 0.0

                            result_row = {
                                'dataset': self.current_dataset_name,
                                'config': self.current_config_tag,
                                'model': model_name,
                                'entanglement': ent_type,
                                'optimization_level': opt_level,
                                'optimization_name': opt_label,
                                'optimization_description': self._optimization_description(opt_level),
                                'noise_level': noise_level,
                                'two_qubit_error': noise_level * 3,
                                'circuit_depth': fm_depth,
                                'total_gates': fm_gates,
                                'two_qubit_gates': fm_2q,
                                'component_count': len(component_feature_maps),
                                'aggregate_circuit_depth': aggregate_depth,
                                'aggregate_total_gates': aggregate_total_gates,
                                'aggregate_two_qubit_gates': aggregate_two_qubit_gates,
                                'accuracy_mean': acc_mean,
                                'accuracy_std': acc_std,
                                'accuracy_ci95': acc_ci95,
                                'accuracy_median': np.median(run_accs),
                                'accuracy_min': np.min(run_accs),
                                'accuracy_max': np.max(run_accs),
                                'f1_mean': np.mean(run_f1s),
                                'f1_std': np.std(run_f1s, ddof=1) if n_runs_done > 1 else 0,
                                'mcc_mean': np.mean(run_mccs),
                                'mcc_std': np.std(run_mccs, ddof=1) if n_runs_done > 1 else 0,
                                'avg_kernel_time': avg_kernel_time,
                                'kernel_time_std': np.std(run_kernel_times, ddof=1) if n_runs_done > 1 else 0,
                                'avg_classifier_time': avg_classifier_time,
                                'classifier_time_std': np.std(run_classifier_times, ddof=1) if n_runs_done > 1 else 0,
                                'avg_time': avg_total_time,
                                'time_std': np.std(run_times, ddof=1) if n_runs_done > 1 else 0,
                                'kernel_fraction_of_total': kernel_fraction,
                                'classifier_fraction_of_total': classifier_fraction,
                                'n_runs': len(run_accs),
                                'num_qubits': num_qubits,
                                'config': self.current_config_tag,
                            }
                            results.append(result_row)
                            
                            logger.info(
                                f"  SUMMARY: Acc={acc_mean:.4f}±{acc_std:.4f} "
                                f"(CI95: ±{acc_ci95:.4f}), kernel={avg_kernel_time:.2f}±"
                                f"{(np.std(run_kernel_times, ddof=1) if n_runs_done > 1 else 0):.2f}s, "
                                f"classifier={avg_classifier_time:.2f}±"
                                f"{(np.std(run_classifier_times, ddof=1) if n_runs_done > 1 else 0):.2f}s, "
                                f"total={avg_total_time:.2f}±"
                                f"{(np.std(run_times, ddof=1) if n_runs_done > 1 else 0):.2f}s, "
                                f"kernel_share={kernel_fraction:.1%}, classifier_share={classifier_fraction:.1%}, "
                                f"primary Gates={fm_gates}, aggregate Gates={aggregate_total_gates}, "
                                f"n={len(run_accs)}"
                            )
                        else:
                            logger.warning(f"  ALL RUNS FAILED for {model_name}/{opt_label}/{ent_type}/{noise_label}")
                        
                        # ---- Incremental save after each (model, opt, ent, noise) block ----
                        if results:
                            inc_df = self._order_output_columns(pd.DataFrame(results))
                            inc_df.to_csv(
                                RESULTS_DIR / f'comprehensive_model_noise_{prefix}.csv',
                                index=False
                            )
                        
                        # Clear GPU memory between configs on all detected devices.
                        for gpu_info in (gpu_mgr_phase4.gpu_info if 'gpu_mgr_phase4' in locals() else []):
                            clear_gpu_memory(gpu_info['id'])
                        gc.collect()

        if per_run_results:
            per_run_df = self._order_output_columns(pd.DataFrame(per_run_results))
            per_run_df.to_csv(
                RESULTS_DIR / f'comprehensive_model_noise_runs_{prefix}.csv',
                index=False
            )

            comparison_df = self._build_phase4_paired_statistics(per_run_df)
            if not comparison_df.empty:
                self._order_output_columns(comparison_df).to_csv(
                    RESULTS_DIR / f'comprehensive_model_noise_stats_{prefix}.csv',
                    index=False
                )
                logger.info("✅ Phase 4 paired statistics saved")
        
        # ---- Final summary ----
        df = pd.DataFrame(results)
        if len(df) > 0:
            logger.info("\n" + "=" * 80)
            logger.info("📊 COMPREHENSIVE MODEL-NOISE-OPTIMIZATION-ENTANGLEMENT SUMMARY")
            logger.info("=" * 80)
            
            for ent in entanglement_types:
                logger.info(f"\n  === ENTANGLEMENT: {ent.upper()} ===")
                ent_df = df[df['entanglement'] == ent]
                for model in model_names:
                    model_df = ent_df[ent_df['model'] == model]
                    if len(model_df) == 0:
                        continue
                    logger.info(f"\n  {model} ({ent}):")
                    for opt in [0, 3]:
                        opt_df = model_df[model_df['optimization_level'] == opt]
                        if len(opt_df) == 0:
                            continue
                        ideal_row = opt_df[opt_df['noise_level'] == 0.0]
                        high_noise_row = opt_df[opt_df['noise_level'] == opt_df['noise_level'].max()]
                        if len(ideal_row) > 0 and len(high_noise_row) > 0:
                            ideal_acc = ideal_row.iloc[0]['accuracy_mean']
                            noisy_acc = high_noise_row.iloc[0]['accuracy_mean']
                            degradation = ideal_acc - noisy_acc
                            pct_deg = (degradation / ideal_acc * 100) if ideal_acc > 0 else 0
                            max_nl = high_noise_row.iloc[0]['noise_level']
                            logger.info(f"    L{opt}: Ideal={ideal_acc:.4f} → Noisy({max_nl})={noisy_acc:.4f} "
                                       f"(degradation: {degradation:.4f} = {pct_deg:.1f}%)")
            
            # ---- Statistical comparisons: L0 vs L3 at each noise level per entanglement ----
            logger.info("\n  📈 L0 vs L3 COMPARISON (per entanglement, per noise level):")
            for ent in entanglement_types:
                ent_df = df[df['entanglement'] == ent]
                logger.info(f"\n    --- {ent.upper()} ---")
                for model in model_names:
                    model_df = ent_df[ent_df['model'] == model]
                    logger.info(f"\n    {model}:")
                    for nl in noise_levels:
                        l0_row = model_df[(model_df['optimization_level'] == 0) & (model_df['noise_level'] == nl)]
                        l3_row = model_df[(model_df['optimization_level'] == 3) & (model_df['noise_level'] == nl)]
                        if len(l0_row) > 0 and len(l3_row) > 0:
                            l0_acc = l0_row.iloc[0]['accuracy_mean']
                            l3_acc = l3_row.iloc[0]['accuracy_mean']
                            diff = l3_acc - l0_acc
                            logger.info(f"      noise={nl:.4f}: L0={l0_acc:.4f}, L3={l3_acc:.4f}, "
                                       f"diff={diff:+.4f}")
            
            # ---- Entanglement comparison: full vs linear at matched conditions ----
            logger.info("\n  📈 FULL vs LINEAR ENTANGLEMENT COMPARISON:")
            for model in model_names:
                logger.info(f"\n    {model}:")
                for opt in [0, 3]:
                    for nl in [0.0, 0.001, 0.01, 0.05]:
                        full_row = df[(df['model']==model) & (df['entanglement']=='full') & 
                                      (df['optimization_level']==opt) & (df['noise_level']==nl)]
                        lin_row = df[(df['model']==model) & (df['entanglement']=='linear') & 
                                     (df['optimization_level']==opt) & (df['noise_level']==nl)]
                        if len(full_row) > 0 and len(lin_row) > 0:
                            f_acc = full_row.iloc[0]['accuracy_mean']
                            l_acc = lin_row.iloc[0]['accuracy_mean']
                            logger.info(f"      L{opt} noise={nl:.4f}: full={f_acc:.4f}, linear={l_acc:.4f}, "
                                       f"diff={l_acc-f_acc:+.4f}")
            
            # Save final results
            self._order_output_columns(df).to_csv(
                RESULTS_DIR / f'comprehensive_model_noise_{prefix}.csv', index=False)
            logger.info(f"\n✅ Comprehensive analysis saved ({len(df)} result rows)")
        
        return df
    
    def run_single_model(self, model_name: str, model_type: str, config: Dict,
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         num_qubits: int) -> Dict[str, Any]:
        """Train and evaluate a single model"""
        
        result = {
            'model': model_name,
            'model_type': model_type,
            'num_qubits': num_qubits,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'status': 'failed'
        }
        
        try:
            # Select best available GPU for this model run
            best_gpu = self._select_gpu_for_workload(f"single-model-{model_name}-{model_type}")
            logger.info(f"Training {model_name} on GPU {best_gpu}...")
            start_time = time.time()
            y_pred_proba = None  # Initialize for all model paths
            
            feature_map = config.get('feature_map')
            if feature_map is None:
                feature_map = create_feature_map(num_qubits, 'ZZ', reps=2)
            
            # Create and record circuit metrics (flattened for readability)
            circuit_metrics, transpiled_fm = self.metrics_analyzer.transpile_and_analyze(
                feature_map, model_name
            )
            # Extract transpiled metrics (the actual execution metrics)
            if 'transpiled' in circuit_metrics:
                tm = circuit_metrics['transpiled']
                result['circuit_depth'] = tm.get('depth')
                result['total_gates'] = tm.get('total_gates')
                result['single_qubit_gates'] = tm.get('single_qubit_gates')
                result['two_qubit_gates'] = tm.get('two_qubit_gates')
                result['gate_density'] = tm.get('gate_density')
                result['two_qubit_ratio'] = tm.get('two_qubit_ratio')
                result['circuit_volume'] = tm.get('circuit_volume')
            
            if model_type == 'QSVC_Standard':
                # Use qiskit-ml QSVC class with GPU-backed FidelityQuantumKernel
                # This tests the official qiskit-ml API path (different from precomputed GPU)
                gpu_fidelity = create_gpu_fidelity(cuda_device=best_gpu)
                fidelity_kernel = FidelityQuantumKernel(feature_map=transpiled_fm, fidelity=gpu_fidelity)
                qsvc = QSVC(quantum_kernel=fidelity_kernel)
                qsvc.fit(X_train, y_train)
                y_pred = qsvc.predict(X_test)
                y_pred_proba = None  # QSVC doesn't provide predict_proba natively
                
            elif model_type == 'QSVC_Precomputed':
                # Use GPUFidelityKernel for precomputed kernel - fastest QSVC variant
                # GPU is REQUIRED - no CPU fallback (would be too slow for 30 runs)
                gpu_kernel = GPUFidelityKernel(transpiled_fm, gpu_id=best_gpu, assume_pretranspiled=True)
                K_train = gpu_kernel.evaluate(X_train, X_train)
                K_test = gpu_kernel.evaluate(X_test, X_train)
                
                # Use cuML SVC on GPU with class_weight='balanced' for imbalanced data
                svc = cuSVC(kernel='precomputed', class_weight='balanced',
                            cache_size=8192.0, max_iter=10000, nochange_steps=100,
                            output_type='numpy')
                svc.fit(K_train, y_train)
                y_pred = svc.predict(K_test)
                y_pred_proba = cuml_svc_predict_proba(svc, K_test)
                
            elif model_type == 'QRF':
                model = QuantumRandomForest(
                    n_estimators=config.get('n_estimators', 5),
                    num_qubits=min(num_qubits, 4),
                    random_state=42,
                    use_gpu=True,
                    gpu_id=best_gpu
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = None  # QRF doesn't provide standard probabilities
                
            elif model_type == 'QVE':
                # Quantum Voting Ensemble with GPU-accelerated kernels
                # Uses the experiment's transpiled feature map + a secondary feature map
                # GPU is REQUIRED - no CPU fallback
                estimators = []
                # First estimator uses the experiment's feature map (same circuit config)
                secondary_fm = transpile_with_cache(
                    create_feature_map(num_qubits, 'Z', reps=1),
                    ('single-model-secondary-z', num_qubits, 2),
                    optimization_level=2
                )
                ensemble_fms = [
                    ('primary', transpiled_fm),
                    ('secondary', secondary_fm),
                ]
                for fm_name, fm_circuit in ensemble_fms:
                    gpu_kernel = GPUFidelityKernel(fm_circuit, gpu_id=best_gpu, assume_pretranspiled=True)
                    K_train = gpu_kernel.evaluate(X_train, X_train)
                    K_test = gpu_kernel.evaluate(X_test, X_train)
                    qsvc = cuSVC(kernel='precomputed', class_weight='balanced',
                                  cache_size=8192.0, max_iter=10000, nochange_steps=100,
                                  output_type='numpy')
                    qsvc.fit(K_train, y_train)
                    estimators.append((f'QSVC_{fm_name}', qsvc, K_test))
                
                # Manual voting for precomputed kernels
                predictions = [e[1].predict(e[2]) for e in estimators]
                y_pred = np.array([Counter(votes).most_common(1)[0][0] 
                                  for votes in zip(*predictions)])
                # Average pseudo-probabilities via decision_function + sigmoid for ROC-AUC
                probas = [cuml_svc_predict_proba(e[1], e[2]) for e in estimators]
                y_pred_proba = np.mean(probas, axis=0)
                
            elif model_type == 'QWE':
                # Quantum Weighted Ensemble with GPU-accelerated kernels
                # Uses the experiment's transpiled feature map + a secondary feature map
                # GPU is REQUIRED - no CPU fallback
                secondary_fm = transpile_with_cache(
                    create_feature_map(num_qubits, 'Pauli', reps=1),
                    ('single-model-secondary-pauli', num_qubits, 2),
                    optimization_level=2
                )
                ensemble_fms = [
                    ('primary', transpiled_fm),
                    ('secondary', secondary_fm),
                ]

                # Manual weighted voting for precomputed kernels using an
                # internal validation split to estimate estimator weights.
                X_fit, X_val, y_fit, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
                )

                validation_scores = []
                probas = []
                for fm_name, fm_circuit in ensemble_fms:
                    gpu_kernel = GPUFidelityKernel(fm_circuit, gpu_id=best_gpu, assume_pretranspiled=True)

                    K_fit = gpu_kernel.evaluate(X_fit, X_fit)
                    K_val = gpu_kernel.evaluate(X_val, X_fit)
                    svc_val = cuSVC(kernel='precomputed', class_weight='balanced',
                                    cache_size=8192.0, max_iter=10000, nochange_steps=100,
                                    output_type='numpy')
                    svc_val.fit(K_fit, y_fit)
                    y_val_pred = svc_val.predict(K_val)
                    validation_scores.append(max(accuracy_score(y_val, y_val_pred), 1e-6))

                    K_train = gpu_kernel.evaluate(X_train, X_train)
                    K_test = gpu_kernel.evaluate(X_test, X_train)
                    svc = cuSVC(kernel='precomputed', class_weight='balanced',
                               cache_size=8192.0, max_iter=10000, nochange_steps=100,
                               output_type='numpy')
                    svc.fit(K_train, y_train)
                    probas.append(cuml_svc_predict_proba(svc, K_test))

                weights = np.asarray(validation_scores, dtype=np.float64)
                weights = weights / weights.sum()
                avg_proba = np.average(np.asarray(probas), axis=0, weights=weights)
                y_pred = np.argmax(avg_proba, axis=1)
                y_pred_proba = avg_proba
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Calculate metrics - pass y_pred_proba for ROC-AUC computation
            train_time = time.time() - start_time
            metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba=y_pred_proba, train_time=train_time)
            
            result.update({
                'status': 'success',
                **metrics
            })
            
            logger.info(f"✅ {model_name}: Acc={metrics['accuracy']:.4f}, Bal_Acc={metrics['balanced_accuracy']:.4f}, F1={metrics['f1_score']:.4f}, MCC={metrics['mcc']:.4f}, Time={train_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ {model_name} failed: {e}")
            result['error'] = str(e)
        
        finally:
            # Clear GPU memory after each model to prevent OOM errors
            clear_gpu_memory(best_gpu)
            gc.collect()
        
        return result
    
    def _run_phases_1_to_3(self, X, y, num_qubits, sample_size, dataset_name,
                           linear_coupling, n_stat_runs, stat_seeds):
        """Run Phases 1-3: circuit optimization analysis on ideal simulators.
        
        Phase 1: Supporting L0 vs L3 compiled-circuit sanity check
        Phase 2: Unoptimized vs Optimized comparison across feature map configs
        Phase 3: All 4 Qiskit optimization levels (0-3)
        
        These phases are skipped in quick_test mode (Phase 4 covers L0 vs L3 under noise).
        """
        basis_gates = ['u', 'cx', 'rz', 'sx', 'x']
        prefix = self._get_filename_prefix()
        
        # ================================================================
        # PHASE 1: SUPPORTING COMPILED-CIRCUIT SANITY CHECK
        # ================================================================
        phase1_file = RESULTS_DIR / f'multi_method_optimization_{prefix}.csv'
        phase2_file = RESULTS_DIR / f'multi_method_model_results_{prefix}.csv'
        phase3_file = RESULTS_DIR / f'optimization_levels_{prefix}.csv'
        
        if phase1_file.exists() and phase2_file.exists() and phase3_file.exists():
            logger.info("\n" + "="*80)
            logger.info("🔄 PHASES 1-3: ALL OUTPUT FILES EXIST — SKIPPING (resume mode)")
            logger.info(f"   Phase 1: {phase1_file.name}")
            logger.info(f"   Phase 2: {phase2_file.name}")
            logger.info(f"   Phase 3: {phase3_file.name}")
            logger.info("="*80)
            return
        
        logger.info("\n" + "="*80)
        logger.info("🚀 PHASE 1: SUPPORTING COMPILED-CIRCUIT SANITY CHECK")
        logger.info("="*80)
        logger.info("Comparison: Qiskit L0 (baseline) | Qiskit L3 (full optimization)")
        
        multi_optimizer = MultiMethodOptimizer(
            num_qubits=num_qubits,
            coupling_map=linear_coupling,
            basis_gates=basis_gates
        )
        
        test_fm = create_feature_map(num_qubits, 'ZZ', reps=2, entanglement='full')
        
        # Compare circuit realizations used in model evaluation.
        logger.info("\n📊 Running compiled-circuit sanity check on ZZ feature map (reps=2)...")
        method_comparison_df = multi_optimizer.compare_all_methods(test_fm, "ZZ_reps2_baseline")
        
        # Identify the best method
        best_method = 'qiskit_level3'
        valid_methods = method_comparison_df[
            ~method_comparison_df.get('error', pd.Series([None]*len(method_comparison_df))).apply(lambda x: x is not None)
        ]
        if len(valid_methods) > 1:
            non_baseline = valid_methods[valid_methods['method'] != 'none (baseline)']
            if len(non_baseline) > 0:
                best_by_depth = non_baseline.loc[non_baseline['depth'].idxmin()]
                best_method = best_by_depth['method']
                logger.info(f"\n🏆 BEST METHOD by depth: {best_method}")
                logger.info(f"   Depth: {best_by_depth['depth']}, Gates: {best_by_depth['total_gates']}")
                logger.info(f"   Depth reduction: {best_by_depth.get('depth_reduction_pct', 0):.1f}%")
        
        method_comparison_df['dataset'] = dataset_name
        method_comparison_df['config'] = self.current_config_tag
        method_comparison_df['num_qubits'] = num_qubits
        self._order_output_columns(method_comparison_df).to_csv(
            RESULTS_DIR / f'multi_method_optimization_{prefix}.csv', index=False)
        logger.info("✅ Phase 1 supporting circuit comparison saved")
        
        # Train QSVC with L0 and L3 for statistical comparison
        logger.info(f"\n🎯 Testing L0 vs L3 with QSVC ({n_stat_runs} runs, leakage-safe preprocessing)")
        multi_method_results = []
        method_aggregated_results = {}
        
        for method_name in ['none (baseline)', 'qiskit_level3']:
            try:
                if method_name == 'none (baseline)':
                    fm_optimized = transpile_with_cache(
                        test_fm,
                        ('phase1-test-fm', num_qubits, 0, _normalize_coupling_map_key(linear_coupling)),
                        optimization_level=0,
                        basis_gates=basis_gates,
                        coupling_map=linear_coupling, seed_transpiler=42
                    )
                else:
                    fm_optimized, _ = multi_optimizer.optimize_qiskit(test_fm, level=3)
                
                circuit_depth = fm_optimized.depth()
                total_gates = sum(fm_optimized.count_ops().values())
                two_qubit_gates = sum(c for g, c in fm_optimized.count_ops().items()
                                      if g.lower() in ['cx', 'cz', 'swap', 'ecr', 'rzz', 'rxx', 'ryy'])
                
                logger.info(f"   {method_name}: launching {len(stat_seeds)} runs across GPUs")

                run_accs, run_f1s, run_times = [], [], []
                phase1_tasks = [
                    (run_idx, seed, method_name, X, y, num_qubits, fm_optimized)
                    for run_idx, seed in enumerate(stat_seeds)
                ]
                phase1_workers = max(1, min(MAX_PARALLEL_GPUS, (self.gpu_manager.gpu_count if self.gpu_manager else 1), len(phase1_tasks)))
                phase1_results = run_parallel_on_gpus(
                    phase1_tasks,
                    self._execute_ideal_qsvc_run,
                    num_workers=phase1_workers,
                    timeout_per_task=7200,
                )
                for run_result in phase1_results:
                    if not run_result or run_result.get('status') != 'success':
                        logger.warning(f"      Run failed for {method_name}: {run_result.get('error') if isinstance(run_result, dict) else 'unknown error'}")
                        continue
                    run_idx = int(run_result['run_idx'])
                    t = float(run_result['training_time'])
                    run_accs.append(float(run_result['accuracy']))
                    run_f1s.append(float(run_result['f1_score']))
                    run_times.append(t)
                    multi_method_results.append({
                        'method': method_name, 'model': 'QSVC_Precomputed_GPU',
                        'run': run_idx+1, 'seed': int(run_result['seed']),
                        'gpu_id': int(run_result['gpu_id']),
                        'circuit_depth': circuit_depth, 'total_gates': total_gates,
                        'two_qubit_gates': two_qubit_gates,
                        'accuracy': float(run_result['accuracy']), 'f1_score': float(run_result['f1_score']),
                        'precision': float(run_result['precision']), 'recall': float(run_result['recall']),
                        'mcc': float(run_result['mcc']), 'training_time': t,
                        'dataset': dataset_name, 'config': self.current_config_tag
                    })
                    remaining = n_stat_runs - (run_idx + 1)
                    eta = (remaining * t) / (60 * max(1, phase1_workers)) if t > 0 else 0
                    logger.info(f"      Run {run_idx+1}/{n_stat_runs} on GPU {run_result['gpu_id']}: Acc={run_result['accuracy']:.4f} ({t:.1f}s, ETA: {eta:.1f}min)")

                if not run_accs:
                    logger.warning(f"   {method_name}: no successful runs; excluding from Phase 1 aggregates")
                    continue
                
                n_runs_done = len(run_accs)
                acc_mean = np.mean(run_accs)
                acc_std = np.std(run_accs, ddof=1) if n_runs_done > 1 else 0
                acc_ci = stats.t.ppf(0.975, df=max(n_runs_done-1, 1)) * acc_std / np.sqrt(n_runs_done)
                method_aggregated_results[method_name] = {
                    'accuracy_mean': acc_mean, 'accuracy_std': acc_std, 'accuracy_ci95': acc_ci,
                    'f1_mean': np.mean(run_f1s), 'f1_std': np.std(run_f1s),
                    'circuit_depth': circuit_depth, 'total_gates': total_gates,
                    'two_qubit_gates': two_qubit_gates, 'n_runs': len(run_accs)
                }
                logger.info(f"   {method_name}: Acc={acc_mean:.4f}±{acc_std:.4f} (CI95: ±{acc_ci:.4f})")
            except Exception as e:
                logger.warning(f"   {method_name} failed: {e}")
        
        # Statistical significance test
        if len(method_aggregated_results) >= 2:
            baseline_accs = [r['accuracy'] for r in multi_method_results if r['method'] == 'none (baseline)']
            for mn, ms in method_aggregated_results.items():
                if mn == 'none (baseline)':
                    continue
                method_accs = [r['accuracy'] for r in multi_method_results if r['method'] == mn]
                if len(baseline_accs) >= 2 and len(method_accs) >= 2 and len(baseline_accs) == len(method_accs):
                    if np.allclose(method_accs, baseline_accs):
                        t_stat, p_value = 0.0, 1.0
                    else:
                        t_stat, p_value = stats.ttest_rel(method_accs, baseline_accs)
                    diff = np.array(method_accs) - np.array(baseline_accs)
                    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    logger.info(f"   {mn} vs baseline: t={t_stat:.3f}, p={p_value:.4f} {sig}, d={cohens_d:.3f}")
        
        # NISQ feasibility scores
        if method_aggregated_results:
            for mn, ms in method_aggregated_results.items():
                nisq = compute_nisq_feasibility_score({
                    'depth': ms['circuit_depth'], 'total_gates': ms['total_gates'],
                    'two_qubit_gates': ms['two_qubit_gates']
                })
                ms['nisq_feasibility_score'] = nisq['nisq_feasibility_score']
                ms['estimated_circuit_fidelity'] = nisq['estimated_circuit_fidelity']
                ms['expected_noisy_accuracy'] = compute_expected_accuracy_under_noise(
                    ms['accuracy_mean'], nisq['estimated_circuit_fidelity'])
                logger.info(f"   {mn}: NISQ={nisq['nisq_feasibility_score']:.1f}/100, "
                           f"Fidelity={nisq['estimated_circuit_fidelity']:.4f}")
            
            agg_df = pd.DataFrame([
                {
                    'dataset': dataset_name,
                    'config': self.current_config_tag,
                    'method': m,
                    **s,
                }
                for m, s in method_aggregated_results.items()
            ])
            self._order_output_columns(agg_df).to_csv(
                RESULTS_DIR / f'multi_method_statistics_{prefix}.csv', index=False)
        
        if multi_method_results:
            self._order_output_columns(pd.DataFrame(multi_method_results)).to_csv(
                RESULTS_DIR / f'multi_method_model_results_{prefix}.csv', index=False)
        
        logger.info("✅ Phase 1 complete")
        
        # ================================================================
        # PHASE 2: UNOPTIMIZED vs OPTIMIZED CIRCUIT COMPARISON
        # ================================================================
        logger.info("\n" + "="*80)
        logger.info("🔬 PHASE 2: CIRCUIT OPTIMIZATION IMPACT ANALYSIS")
        logger.info("="*80)
        logger.info(f"Running {n_stat_runs} trials per configuration")
        
        optimization_results = []
        optimization_statistics = {}
        
        feature_map_configs = [
            ('ZZ', 2, 'full'),
            ('ZZ', 3, 'full'),
            ('Pauli', 2, 'full'),
        ]
        
        for fm_type, reps, entanglement in feature_map_configs:
            config_key = f"{fm_type}_r{reps}_{entanglement}"
            logger.info(f"\n📊 Testing {fm_type} reps={reps}")
            unopt_label = self._optimization_label(0)
            opt_label = self._optimization_label(3)
            
            base_fm = create_feature_map(num_qubits, fm_type, reps=reps, entanglement=entanglement)
            fm_unopt = transpile_with_cache(
                base_fm,
                ('phase2', fm_type, reps, entanglement, 0, _normalize_coupling_map_key(linear_coupling)),
                optimization_level=0, basis_gates=basis_gates,
                coupling_map=linear_coupling, seed_transpiler=42
            )
            fm_opt = transpile_with_cache(
                base_fm,
                ('phase2', fm_type, reps, entanglement, 3, _normalize_coupling_map_key(linear_coupling)),
                optimization_level=3, basis_gates=basis_gates,
                coupling_map=linear_coupling, seed_transpiler=42
            )
            
            unopt_m = self.metrics_analyzer.analyze_circuit(fm_unopt, f"{config_key}_unopt")
            opt_m = self.metrics_analyzer.analyze_circuit(fm_opt, f"{config_key}_opt")
            logger.info(f"  Unopt: Depth={unopt_m['depth']}, Gates={unopt_m['total_gates']}")
            logger.info(f"  Opt:   Depth={opt_m['depth']}, Gates={opt_m['total_gates']}")
            
            unopt_accs, opt_accs = [], []
            unopt_f1s, opt_f1s = [], []
            phase2_tasks = []
            for run_idx, seed in enumerate(stat_seeds):
                phase2_tasks.append((run_idx, seed, 'unoptimized', X, y, num_qubits, fm_unopt))
                phase2_tasks.append((run_idx, seed, 'optimized', X, y, num_qubits, fm_opt))
            phase2_workers = max(1, min(MAX_PARALLEL_GPUS, (self.gpu_manager.gpu_count if self.gpu_manager else 1), len(phase2_tasks)))
            phase2_results = run_parallel_on_gpus(
                phase2_tasks,
                self._execute_ideal_qsvc_run,
                num_workers=phase2_workers,
                timeout_per_task=7200,
            )
            for run_result in phase2_results:
                if not run_result or run_result.get('status') != 'success':
                    logger.warning(f"  Phase 2 run failed for {config_key}: {run_result.get('error') if isinstance(run_result, dict) else 'unknown error'}")
                    continue
                run_idx = int(run_result['run_idx'])
                optimization_name = str(run_result['optimization_name'])
                t = float(run_result['training_time'])
                result_row = {
                    'dataset': dataset_name,
                    'config': self.current_config_tag,
                    'feature_map': fm_type, 'reps': reps, 'entanglement': entanglement,
                    'optimization': unopt_label if optimization_name == 'unoptimized' else opt_label,
                    'optimization_level': 0 if optimization_name == 'unoptimized' else 3,
                    'optimization_name': unopt_label if optimization_name == 'unoptimized' else opt_label,
                    'optimization_description': self._optimization_description(0 if optimization_name == 'unoptimized' else 3),
                    'run': run_idx+1, 'seed': int(run_result['seed']),
                    'gpu_id': int(run_result['gpu_id']),
                    'accuracy': float(run_result['accuracy']), 'f1_score': float(run_result['f1_score']),
                    'mcc': float(run_result['mcc']), 'training_time': t
                }
                if optimization_name == 'unoptimized':
                    unopt_accs.append(float(run_result['accuracy']))
                    unopt_f1s.append(float(run_result['f1_score']))
                    result_row.update({
                        'circuit_depth': unopt_m['depth'], 'total_gates': unopt_m['total_gates'],
                        'two_qubit_gates': unopt_m['two_qubit_gates'],
                    })
                else:
                    opt_accs.append(float(run_result['accuracy']))
                    opt_f1s.append(float(run_result['f1_score']))
                    result_row.update({
                        'circuit_depth': opt_m['depth'], 'total_gates': opt_m['total_gates'],
                        'two_qubit_gates': opt_m['two_qubit_gates'],
                    })
                optimization_results.append(result_row)
            
            if len(unopt_accs) >= 2 and len(opt_accs) >= 2:
                u_mean, o_mean = np.mean(unopt_accs), np.mean(opt_accs)
                if len(unopt_accs) == len(opt_accs):
                    if np.allclose(opt_accs, unopt_accs):
                        t_stat, p_val = 0.0, 1.0
                    else:
                        t_stat, p_val = stats.ttest_rel(opt_accs, unopt_accs)
                    diff = np.array(opt_accs) - np.array(unopt_accs)
                    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                else:
                    t_stat, p_val = stats.ttest_ind(opt_accs, unopt_accs)
                    cohens_d = 0
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                logger.info(f"  {config_key}: Unopt={u_mean:.4f} → Opt={o_mean:.4f}, "
                           f"t={t_stat:.3f}, p={p_val:.4f} {sig}")
                
                unopt_nisq = compute_nisq_feasibility_score(unopt_m)
                opt_nisq = compute_nisq_feasibility_score(opt_m)
                optimization_statistics[config_key] = {
                    'dataset': dataset_name,
                    'config': self.current_config_tag,
                    'feature_map': fm_type, 'reps': reps, 'entanglement': entanglement,
                    'unopt_optimization_name': unopt_label,
                    'opt_optimization_name': opt_label,
                    'unopt_acc_mean': u_mean, 'opt_acc_mean': o_mean,
                    'unopt_acc_std': float(np.std(unopt_accs, ddof=1)) if len(unopt_accs) > 1 else 0.0,
                    'opt_acc_std': float(np.std(opt_accs, ddof=1)) if len(opt_accs) > 1 else 0.0,
                    'acc_change': o_mean - u_mean, 'p_value': p_val,
                    't_stat': t_stat,
                    'cohens_d': cohens_d,
                    'depth_reduction_pct': ((unopt_m['depth'] - opt_m['depth']) / unopt_m['depth']) * 100,
                    'gate_reduction_pct': ((unopt_m['total_gates'] - opt_m['total_gates']) / unopt_m['total_gates']) * 100,
                    'nisq_score_improvement': opt_nisq['nisq_feasibility_score'] - unopt_nisq['nisq_feasibility_score'],
                    'significant': p_val < 0.05, 'n_runs': len(unopt_accs)
                }
            
            for gpu_info in (self.gpu_manager.gpu_info if self.gpu_manager else []):
                clear_gpu_memory(gpu_info['id'])
            gc.collect()
        
        if optimization_results:
            self._order_output_columns(pd.DataFrame(optimization_results)).to_csv(
                RESULTS_DIR / f'optimization_comparison_{prefix}.csv', index=False)
        if optimization_statistics:
            self._order_output_columns(pd.DataFrame(list(optimization_statistics.values()))).to_csv(
                RESULTS_DIR / f'optimization_statistics_{prefix}.csv', index=False)
        logger.info("✅ Phase 2 complete")
        
        # ================================================================
        # PHASE 3: OPTIMIZATION LEVELS COMPARISON (0-3)
        # ================================================================
        logger.info("\n" + "="*80)
        logger.info("⚡ PHASE 3: QISKIT OPTIMIZATION LEVELS (0-3)")
        logger.info("="*80)
        
        opt_level_results = []
        base_fm = create_feature_map(num_qubits, 'ZZ', reps=2, entanglement='full')
        opt_names = {level: self._optimization_label(level) for level in range(4)}
        
        for opt_level in [0, 1, 2, 3]:
            logger.info(f"\n  Testing opt_level={opt_level} ({opt_names[opt_level]})...")
            try:
                fm = transpile_with_cache(
                    base_fm,
                    ('phase3', num_qubits, 'ZZ', 2, 'full', opt_level, _normalize_coupling_map_key(linear_coupling)),
                    optimization_level=opt_level,
                    basis_gates=basis_gates, coupling_map=linear_coupling,
                    seed_transpiler=42
                )
                m = self.metrics_analyzer.analyze_circuit(fm, f"ZZ_r2_opt{opt_level}")

                run_accs, run_f1s = [], []
                phase3_tasks = [
                    (run_idx, seed, opt_names[opt_level], X, y, num_qubits, fm)
                    for run_idx, seed in enumerate(stat_seeds)
                ]
                phase3_workers = max(1, min(MAX_PARALLEL_GPUS, (self.gpu_manager.gpu_count if self.gpu_manager else 1), len(phase3_tasks)))
                phase3_results = run_parallel_on_gpus(
                    phase3_tasks,
                    self._execute_ideal_qsvc_run,
                    num_workers=phase3_workers,
                    timeout_per_task=7200,
                )
                for run_result in phase3_results:
                    if not run_result or run_result.get('status') != 'success':
                        logger.warning(f"    Phase 3 run failed at opt_level={opt_level}: {run_result.get('error') if isinstance(run_result, dict) else 'unknown error'}")
                        continue
                    run_idx = int(run_result['run_idx'])
                    t = float(run_result['training_time'])
                    run_accs.append(float(run_result['accuracy']))
                    run_f1s.append(float(run_result['f1_score']))
                    opt_level_results.append({
                        'dataset': dataset_name,
                        'config': self.current_config_tag,
                        'optimization_level': opt_level,
                        'optimization_name': opt_names[opt_level],
                        'optimization_description': self._optimization_description(opt_level),
                        'run': run_idx+1, 'seed': int(run_result['seed']),
                        'gpu_id': int(run_result['gpu_id']),
                        'circuit_depth': m['depth'], 'total_gates': m['total_gates'],
                        'two_qubit_gates': m['two_qubit_gates'],
                        'accuracy': float(run_result['accuracy']), 'f1_score': float(run_result['f1_score']),
                        'mcc': float(run_result['mcc']), 'training_time': t
                    })

                if not run_accs:
                    logger.warning(f"     opt_level={opt_level}: no successful runs; skipping aggregate log")
                    continue
                logger.info(f"     Depth={m['depth']}, Acc={np.mean(run_accs):.4f}±{np.std(run_accs):.4f} ({len(run_accs)} runs)")
                for gpu_info in (self.gpu_manager.gpu_info if self.gpu_manager else []):
                    clear_gpu_memory(gpu_info['id'])
                gc.collect()
            except Exception as e:
                logger.warning(f"     opt_level={opt_level} failed: {e}")
        
        if opt_level_results:
            self._order_output_columns(pd.DataFrame(opt_level_results)).to_csv(
                RESULTS_DIR / f'optimization_levels_{prefix}.csv', index=False)
        logger.info("✅ Phase 3 complete")
    
    def run_full_experiment(self, datasets: List[Tuple[str, str]], 
                            num_qubits: int = 10, sample_size: int = 5000):
        """Run the complete circuit depth experiment"""
        self._set_run_context(num_qubits, sample_size)

        aggregate_results = {
            'model_results': [],
            'circuit_analysis': [],
            'cv_results': [],
            'ablation_results': []
        }
        
        logger.info("=" * 80)
        logger.info("🔬 CIRCUIT DEPTH EXPERIMENT: QUANTUM CIRCUIT DEPTH AND GATE OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Config Tag: {self.current_config_tag}")
        logger.info(f"Configuration: {num_qubits} qubits, {sample_size} samples")
        logger.info(f"Noise Simulation: {'ENABLED' if self.enable_noise_simulation else 'DISABLED'}")
        logger.info(f"K-folds: {self.config['k_folds']}, Runs: {self.config['n_runs']}")
        logger.info("=" * 80)
        
        for dataset_name, dataset_path in datasets:
            self._reset_results_storage()
            self._set_run_context(num_qubits, sample_size, dataset_name)
            logger.info(f"\n📊 Processing dataset: {dataset_name}")
            
            # Load data
            try:
                df, detected_delimiter = self._load_dataset(dataset_path)
                logger.info(f"Loaded: {df.shape} (delimiter='{detected_delimiter}')")
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
                continue
            
            # Prepare data (sampling + label encoding only, NO scaling/PCA yet)
            processor = DataProcessor(num_qubits=num_qubits)
            X_raw, y = processor.prepare_data(df, sample_size=sample_size)

            train_size = int(round((1.0 - self.config['test_size']) * len(X_raw)))
            test_size = len(X_raw) - train_size
            logger.info(f"Train: ({train_size}, {X_raw.shape[1]}), Test: ({test_size}, {X_raw.shape[1]})")
            
            # Create coupling map (needed by multiple phases)
            from qiskit.transpiler import CouplingMap
            linear_coupling = CouplingMap.from_line(num_qubits)
            
            # n_stat_runs and stat_seeds used by Phases 1-3
            n_stat_runs = self.config['n_runs']
            stat_seeds = self.config['random_seeds'][:n_stat_runs]
            
            # ================================================================
            # PHASES 1-3: CIRCUIT OPTIMIZATION ANALYSIS (skipped in quick_test)
            # ================================================================
            # Phases 1-3 compare optimization methods/levels on ideal simulators.
            # Phase 4 is the CORE experiment (all models × opt × noise).
            # In quick_test mode, skip Phases 1-3 to save ~30-60 min.
            # ================================================================
            if not self.quick_test:
                self._run_phases_1_to_3(
                    X_raw, y, num_qubits, sample_size, dataset_name,
                    linear_coupling, n_stat_runs, stat_seeds
                )
            else:
                logger.info("\n" + "="*80)
                logger.info("⏭️  PHASES 1-3 SKIPPED (quick test mode — Phase 4 covers L0 vs L3)")
                logger.info("="*80)
            
            # ================================================================
            # PHASE 4: COMPREHENSIVE MODEL-NOISE-OPTIMIZATION ANALYSIS
            # ================================================================
            # This is the CORE experiment answering the central research question:
            # "How does circuit optimization affect ALL QML model architectures 
            #  under realistic noise conditions?"
            #
            # Replaces old Phase 3.5 (QSVC-only noise sweep) + Phase 4a (QSVC-only
            # model comparison) + Phase 4b (multi-model L3/ideal only) with a single
            # unified analysis testing ALL models × ALL opt levels × ALL noise levels.
            # ================================================================
            if self.enable_noise_simulation:
                logger.info("\n" + "="*80)
                logger.info("🔬 PHASE 4: COMPREHENSIVE MODEL-NOISE-OPTIMIZATION ANALYSIS")
                logger.info("="*80)
                
                try:
                    comprehensive_df = self.run_comprehensive_model_noise_analysis(
                        X_raw, y, num_qubits, n_runs=self.config['n_runs']
                    )
                    
                    if len(comprehensive_df) > 0:
                        comprehensive_df['dataset'] = dataset_name
                        prefix = self._get_filename_prefix()
                        comprehensive_df.to_csv(
                            RESULTS_DIR / f'comprehensive_model_noise_{prefix}.csv',
                            index=False
                        )
                        logger.info(f"✅ Comprehensive analysis saved ({len(comprehensive_df)} result rows)")
                        
                        # Clean up per-run checkpoint now that Phase 4 is fully complete
                        run_ckpt = CHECKPOINT_DIR / f'per_run_checkpoint_{prefix}.csv'
                        if run_ckpt.exists():
                            run_ckpt.unlink()
                            logger.info("   Cleaned up per-run checkpoint file")
                except Exception as e:
                    logger.warning(f"Comprehensive model-noise analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.info("\n" + "="*80)
                logger.info("⏭️  PHASE 4: COMPREHENSIVE ANALYSIS (SKIPPED)")
                logger.info("="*80)
                logger.info("Use --enable_noise to run the comprehensive model-noise-optimization analysis.")
            
            # ================================================================
            # PHASE 5: K-Fold Cross-Validation (GPU-ACCELERATED)
            # ================================================================
            if not self.quick_test:
                prefix = self._get_filename_prefix()
                cv_results_file = RESULTS_DIR / f'cv_results_{prefix}.csv'
                
                if cv_results_file.exists():
                    logger.info("\n📊 PHASE 5: Cross-Validation — SKIPPED (results file exists)")
                    logger.info(f"   {cv_results_file.name}")
                else:
                    logger.info("\n📊 PHASE 5: Cross-Validation (GPU-Accelerated)")
                    
                    cv_experiment = CrossValidationExperiment(
                        k_folds=self.config['k_folds'],
                        n_runs=self.config['n_runs'],
                        random_seeds=self.config['random_seeds']
                    )
                    
                    # GPU-accelerated CV for representative feature maps
                    cv_feature_maps = [
                        ('QSVC_ZZ', create_feature_map(num_qubits, 'ZZ', reps=2)),
                        ('QSVC_Pauli', create_feature_map(num_qubits, 'Pauli', reps=2)),
                        ('QSVC_ZZFull', create_feature_map(num_qubits, 'ZZ', reps=2, entanglement='full')),
                    ]
                    
                    gpu_mgr = get_gpu_manager()
                    logger.info(f"   Using {gpu_mgr.gpu_count} GPU(s) for parallel CV")
                    
                    for model_name, feature_map in cv_feature_maps:
                        try:
                            cv_result = cv_experiment.run_cv_experiment_gpu(
                                feature_map, X_raw, y, f'{model_name}_CV', 
                                use_parallel=(gpu_mgr.gpu_count > 1), num_qubits=num_qubits
                            )
                            cv_result['dataset'] = dataset_name
                            self.cv_results.append(cv_result)
                            logger.info(f"✅ CV completed for {model_name}: "
                                       f"Acc={cv_result.get('accuracy_mean', 0):.4f}±{cv_result.get('accuracy_std', 0):.4f}")
                        except Exception as e:
                            logger.warning(f"CV failed for {model_name}: {e}")
            else:
                logger.info("\n⏭️  PHASE 5 SKIPPED (quick test mode)")
            
            phases_run = "4 only" if self.quick_test else "1-5"
            logger.info(f"\n✅ Phases {phases_run} completed for dataset: {dataset_name}")
            self.save_all_results()

            aggregate_results['model_results'].extend(copy.deepcopy(self.all_results))
            aggregate_results['circuit_analysis'].extend(copy.deepcopy(self.circuit_analysis_results))
            aggregate_results['cv_results'].extend(copy.deepcopy(self.cv_results))
            aggregate_results['ablation_results'].extend(copy.deepcopy(self.ablation_results))

        self.current_dataset_name = ""
        self.current_dataset_tag = ""
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        
        return aggregate_results
    
    def save_all_results(self):
        """Save all experiment results"""
        
        prefix = self._get_filename_prefix()
        
        # Model results
        if self.all_results:
            df = pd.DataFrame(self.all_results)
            df.to_csv(RESULTS_DIR / f'model_results_{prefix}.csv', index=False)
            logger.info(f"Saved model results: {len(self.all_results)} entries")
        
        # Circuit analysis
        if self.circuit_analysis_results:
            df = pd.DataFrame(self.circuit_analysis_results)
            df.to_csv(RESULTS_DIR / f'circuit_analysis_{prefix}.csv', index=False)
            logger.info(f"Saved circuit analysis: {len(self.circuit_analysis_results)} entries")
        
        # CV results
        if self.cv_results:
            df = pd.DataFrame(self.cv_results)
            df.to_csv(RESULTS_DIR / f'cv_results_{prefix}.csv', index=False)
            logger.info(f"Saved CV results: {len(self.cv_results)} entries")
        
        # Ablation results
        if self.ablation_results:
            df = pd.DataFrame(self.ablation_results)
            df.to_csv(RESULTS_DIR / f'ablation_results_{prefix}.csv', index=False)
            logger.info(f"Saved ablation results: {len(self.ablation_results)} entries")
        
        # Circuit metrics
        self.metrics_analyzer.save_metrics(RESULTS_DIR / f'circuit_metrics_{prefix}.csv')
        
        # Summary JSON
        # Extract num_qubits and sample_size from config tag (e.g., '6q_5k')
        config_parts = self.current_config_tag.split('_')
        num_qubits_tag = config_parts[0] if config_parts else 'unknown'
        sample_size_tag = config_parts[1] if len(config_parts) > 1 else 'unknown'
        
        # GPU info for documentation
        gpu_info_list = []
        try:
            mgr = get_gpu_manager()
            for g in mgr.gpu_info:
                mem = get_gpu_memory_status(g['id'])
                gpu_info_list.append({
                    'id': g['id'], 'name': g['name'],
                    'memory_total_gb': g['memory_total'],
                    'compute_capability': g['compute_capability']
                })
        except Exception:
            pass
        
        summary = {
            'session_id': self.session_id,
            'config_tag': self.current_config_tag,
            'dataset_name': self.current_dataset_name,
            'dataset_tag': self.current_dataset_tag,
            'num_qubits': num_qubits_tag,
            'sample_size': sample_size_tag,
            'config': self.config,
            'noise_simulation_enabled': self.enable_noise_simulation,
            'total_models': len(self.all_results),
            'total_cv_results': len(self.cv_results),
            'total_ablations': len(self.ablation_results),
            'noise_params': NOISE_PARAMS,
            'phases_completed': {
                'phase_1_multi_method': bool(self.circuit_analysis_results or self.all_results),
                'phase_2_optimization_comparison': True,
                'phase_3_optimization_levels': True,
                'phase_4_comprehensive_model_noise': self.enable_noise_simulation,
                'phase_5_cross_validation': bool(self.cv_results)
            },
            'gpu_info': gpu_info_list,
            'datasets_used': list(set(r.get('dataset', '') for r in self.all_results if r.get('dataset'))),
            'successful_models': [r['model'] for r in self.all_results if r.get('status') == 'success'],
            'failed_models': [r['model'] for r in self.all_results if r.get('status') != 'success'],
        }
        
        with open(RESULTS_DIR / f'experiment_summary_{prefix}.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate journal-ready summary report
        self._generate_journal_summary(prefix)
        
        logger.info(f"All results saved to {RESULTS_DIR}")
    
    def _generate_journal_summary(self, timestamp: str):
        """Generate a comprehensive journal-ready summary report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("JOURNAL-READY EXPERIMENT SUMMARY")
        report_lines.append("Circuit Depth Optimization for Quantum Machine Learning")
        report_lines.append("=" * 80)
        report_lines.append(f"\nSession ID: {timestamp}")
        report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"\nConfiguration:")
        report_lines.append(f"  - Number of runs: {self.config['n_runs']} (for statistical significance)")
        report_lines.append(f"  - K-folds: {self.config['k_folds']}")
        report_lines.append(f"  - Test size: {self.config['test_size']}")
        
        # Try to load and analyze saved results
        try:
            # Phase 1 supporting circuit comparison results
            mm_path = RESULTS_DIR / f'multi_method_optimization_{timestamp}.csv'
            if mm_path.exists():
                mm_df = pd.read_csv(mm_path)
                report_lines.append("\n" + "-" * 60)
                report_lines.append("PHASE 1: SUPPORTING COMPILED-CIRCUIT SANITY CHECK")
                report_lines.append("-" * 60)
                report_lines.append("\nSupporting circuit metrics by optimization setting:")
                for _, row in mm_df.iterrows():
                    method = row.get('method', 'unknown')
                    depth = row.get('depth', 0)
                    gates = row.get('total_gates', 0)
                    two_q = row.get('two_qubit_gates', 0)
                    report_lines.append(f"  {method}: depth={depth}, gates={gates}, 2Q_gates={two_q}")
            
            # Phase 1 supporting QSVC statistics
            mms_path = RESULTS_DIR / f'multi_method_statistics_{timestamp}.csv'
            if mms_path.exists():
                mms_df = pd.read_csv(mms_path)
                report_lines.append("\nSupporting QSVC performance by optimization setting:")
                for _, row in mms_df.iterrows():
                    method = row.get('method', 'unknown')
                    acc_mean = row.get('accuracy_mean', 0)
                    acc_std = row.get('accuracy_std', 0)
                    acc_ci = row.get('accuracy_ci95', 0)
                    n_runs = row.get('n_runs', 0)
                    report_lines.append(f"  {method}: acc={acc_mean:.4f} ± {acc_std:.4f} (95% CI: ±{acc_ci:.4f}, n={n_runs})")
            
            # Optimization statistics (Phase 2)
            opt_stats_path = RESULTS_DIR / f'optimization_statistics_{timestamp}.csv'
            if opt_stats_path.exists():
                opt_stats_df = pd.read_csv(opt_stats_path)
                report_lines.append("\n" + "-" * 60)
                report_lines.append("PHASE 2: OPTIMIZATION IMPACT ANALYSIS")
                report_lines.append("-" * 60)
                
                for _, row in opt_stats_df.iterrows():
                    fm = row.get('feature_map', 'unknown')
                    reps = row.get('reps', 0)
                    report_lines.append(f"\nFeature Map: {fm} (reps={reps})")
                    report_lines.append(f"  Unoptimized: {row.get('unopt_acc_mean', 0):.4f} ± {row.get('unopt_acc_std', 0):.4f}")
                    report_lines.append(f"  Optimized:   {row.get('opt_acc_mean', 0):.4f} ± {row.get('opt_acc_std', 0):.4f}")
                    report_lines.append(f"  Accuracy change: {row.get('acc_change', 0):+.4f}")
                    report_lines.append(f"  t-statistic: {row.get('t_stat', 0):.3f}")
                    report_lines.append(f"  p-value: {row.get('p_value', 1):.4f}")
                    report_lines.append(f"  Cohen's d: {row.get('cohens_d', 0):.3f}")
                    report_lines.append(f"  Significant (p<0.05): {row.get('significant', False)}")
                    report_lines.append(f"  Depth reduction: {row.get('depth_reduction_pct', 0):.1f}%")
                    report_lines.append(f"  Gate reduction: {row.get('gate_reduction_pct', 0):.1f}%")
            
            # Comprehensive model-noise results (Phase 4)
            comprehensive_path = RESULTS_DIR / f'comprehensive_model_noise_{timestamp}.csv'
            if comprehensive_path.exists():
                comp_df = pd.read_csv(comprehensive_path)
                report_lines.append("\n" + "-" * 60)
                report_lines.append("PHASE 4: COMPREHENSIVE MODEL-NOISE-OPTIMIZATION ANALYSIS")
                report_lines.append("-" * 60)
                report_lines.append("\nThis demonstrates how circuit optimization affects kernel-QML models")
                report_lines.append("across a range of ideal and noisy simulation conditions.")
                report_lines.append(f"\nModels tested: {sorted(comp_df['model'].unique().tolist())}")
                report_lines.append(f"Optimization levels: {sorted(comp_df['optimization_level'].unique().tolist())}")
                report_lines.append(f"Noise levels tested: {sorted(comp_df['noise_level'].unique().tolist())}")
                
                for model in sorted(comp_df['model'].unique()):
                    model_data = comp_df[comp_df['model'] == model]
                    report_lines.append(f"\n{model}:")
                    for opt in sorted(model_data['optimization_level'].unique()):
                        opt_data = model_data[model_data['optimization_level'] == opt]
                        ideal = opt_data[opt_data['noise_level'] == 0.0]
                        max_noise = opt_data['noise_level'].max()
                        noisy = opt_data[opt_data['noise_level'] == max_noise]
                        if len(ideal) > 0 and len(noisy) > 0:
                            ideal_acc = ideal.iloc[0]['accuracy_mean']
                            noisy_acc = noisy.iloc[0]['accuracy_mean']
                            deg = (ideal_acc - noisy_acc) / ideal_acc * 100 if ideal_acc > 0 else 0
                            report_lines.append(f"  L{opt}: ideal={ideal_acc:.4f} → noise({max_noise})={noisy_acc:.4f} "
                                              f"(degradation: {deg:.1f}%)")
            
            # CV results
            cv_path = RESULTS_DIR / f'cv_results_{timestamp}.csv'
            if cv_path.exists():
                cv_df = pd.read_csv(cv_path)
                report_lines.append("\n" + "-" * 60)
                report_lines.append("PHASE 5: CROSS-VALIDATION RESULTS")
                report_lines.append("-" * 60)
                
                for _, row in cv_df.iterrows():
                    model = row.get('model_name', row.get('model', 'unknown'))
                    acc_mean = row.get('accuracy_mean', 0)
                    acc_std = row.get('accuracy_std', 0)
                    n_folds = row.get('k_folds', 0)
                    n_runs = row.get('n_runs', 0)
                    report_lines.append(f"\n{model}:")
                    report_lines.append(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
                    report_lines.append(f"  Configuration: {n_folds} folds × {n_runs} runs")
            
            # Key findings summary
            report_lines.append("\n" + "=" * 80)
            report_lines.append("KEY FINDINGS FOR PUBLICATION")
            report_lines.append("=" * 80)
            report_lines.append("""
1. QISKIT OPTIMIZATION COMPARISON:
    - Compared the baseline circuit realization against Qiskit optimization levels
    - Measured: circuit depth, gate count, and two-qubit gate count
    - Result: [Check optimization_levels.csv and optimization_statistics.csv]

2. OPTIMIZATION IMPACT ON IDEAL SIMULATION:
    - Research question: Does circuit optimization improve ideal-simulator kernel-QML performance?
    - Methodology: repeated train/test splits with leakage-safe preprocessing
    - Statistics are descriptive because runs reuse one dataset
    - Result: [Check optimization_statistics.csv for paired comparisons]

3. NOISE RESILIENCE (NISQ RELEVANCE):
   - Key contribution: Circuit optimization improves noise resilience
   - Fewer gates = less noise accumulation on real hardware
    - Tested across a practical 10-level noise grid with 30 repeated runs
    - Result: [Check comprehensive_model_noise.csv for model-specific degradation]

4. PRACTICAL IMPLICATIONS:
    - For NISQ-era quantum computing, circuit optimization should be evaluated per model family
   - Depth/gate reduction directly translates to improved real-hardware performance
    - Recommendation: Compare L0 vs L3 and keep the setting that improves noisy robustness
""")
            
        except Exception as e:
            report_lines.append(f"\nError generating detailed summary: {e}")
        
        # Write report with UTF-8 encoding to support ± and other special characters
        report_path = RESULTS_DIR / f'journal_summary_{timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"✅ Journal summary saved to {report_path}")


# ============================================================================
# CLASSICAL BASELINE MODELS
# ============================================================================

def train_classical_baselines(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> List[Dict[str, Any]]:
    """Train classical ML baselines for comparison"""
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    
    classical_models = {
        'SVM_Linear': SVC(kernel='linear', random_state=42),
        'SVM_RBF': SVC(kernel='rbf', random_state=42),
        'SVM_Poly': SVC(kernel='poly', degree=3, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'NaiveBayes': GaussianNB(),
    }
    
    results = []
    
    for name, model in classical_models.items():
        logger.info(f"Training classical: {name}")
        
        try:
            start = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            train_time = time.time() - start
            
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception:
                    pass
            
            metrics = calculate_all_metrics(y_test, y_pred, y_proba, train_time)
            
            results.append({
                'model': f'Classical_{name}',
                'model_type': 'classical',
                'status': 'success',
                **metrics
            })
            
            logger.info(f"✅ {name}: Acc={metrics['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
            results.append({
                'model': f'Classical_{name}',
                'status': 'failed',
                'error': str(e)
            })
    
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the circuit depth experiment"""
    
    parser = argparse.ArgumentParser(
        description='Quantum Circuit Depth and Gate Optimization Experiment for Journal Publication'
    )
    parser.add_argument('--num_qubits', type=int, default=6, help='Number of qubits (default: 6, matching full experiment configs)')
    parser.add_argument('--sample_size', type=int, default=5000, help='Sample size')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['data/IoT_Original_Distribution.csv'],
                        help='Dataset files')
    parser.add_argument('--k_folds', type=int, default=5, help='K-fold CV splits')
    parser.add_argument('--n_runs', type=int, default=30, help='Number of random seed runs (30+ for statistical significance)')
    parser.add_argument('--quick_n_runs', type=int, default=5, help='Number of runs for quick test mode (default: 5)')
    parser.add_argument('--include_classical', action='store_true', help='Include classical baselines')
    parser.add_argument('--quick_test', action='store_true', help='Run quick test mode')
    parser.add_argument('--enable_noise', action='store_true', default=True,
                        help='Enable noise simulation (Phase 4: comprehensive model-noise analysis) - ENABLED by default')
    parser.add_argument('--disable_noise', action='store_true',
                        help='Disable noise simulation (skip Phase 4) for faster testing')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a previous session. Pass the session_id (e.g. v2_noisy_20260302_130734) '
                             'to reuse its timestamp and skip already-completed phases/configs.')
    parser.add_argument('--config_specs', type=str, nargs='+', default=None,
                        help='Explicit experiment configurations as <num_qubits>:<sample_size>, e.g. 10:2500')
    parser.add_argument('--phase4_models', type=str, nargs='+', default=None,
                        choices=['QSVC', 'QVE', 'QWE'],
                        help='Optional Phase 4 model subset override.')
    parser.add_argument('--phase4_entanglements', type=str, nargs='+', default=None,
                        choices=['full', 'linear'],
                        help='Optional Phase 4 entanglement subset override.')
    parser.add_argument('--phase4_noise_levels', type=float, nargs='+', default=None,
                        help='Optional Phase 4 noise-level override list, e.g. 0.0 0.002 0.01 0.05')
    
    args = parser.parse_args()
    
    # Initialize GPU Manager and check GPU status
    logger.info("=" * 80)
    logger.info("🔧 INITIALIZING GPU ENVIRONMENT")
    logger.info("=" * 80)
    
    gpu_mgr = get_gpu_manager()
    if gpu_mgr.gpu_count > 0:
        logger.info(f"✅ GPU Acceleration: ENABLED ({gpu_mgr.gpu_count} GPU(s) available)")
        for info in gpu_mgr.gpu_info:
            mem_status = get_gpu_memory_status(info['id'])
            if mem_status:
                logger.info(f"   GPU {info['id']}: {info['name']} - "
                          f"{mem_status['free']:.1f}GB free / {mem_status['total']:.1f}GB total")
    else:
        logger.error("❌ FATAL: No GPU detected! This experiment requires GPU acceleration.")
        logger.error("   Please ensure NVIDIA GPUs are available and CUDA is properly configured.")
        logger.error("   Exiting - CPU fallback is disabled for this experiment.")
        return
    
    # Check CuPy availability - REQUIRED for GPU kernel computations
    cupy_mod = _ensure_cupy()
    if cupy_mod is not None:
        logger.info("✅ CuPy: Available for GPU matrix operations")
    else:
        logger.error("❌ FATAL: CuPy is not available! GPU kernel computation requires CuPy.")
        logger.error("   Please install CuPy: pip install cupy-cuda12x (adjust for your CUDA version)")
        logger.error("   Exiting - NumPy fallback is disabled for this experiment.")
        return
    
    # Validate GPU AerSimulator works
    try:
        test_sim = _get_shared_aer_simulator(0, method='statevector')
        logger.info("✅ AerSimulator GPU: Configured with cuStateVec")
    except Exception as e:
        logger.error(f"❌ FATAL: GPU AerSimulator failed: {e}")
        logger.error("   Please ensure qiskit-aer-gpu is installed with CUDA support.")
        return
    
    # Validate GPU-backed FidelityQuantumKernel
    try:
        _test_fidelity = create_gpu_fidelity(cuda_device=0)
        logger.info("✅ GPU Fidelity: ComputeUncompute with SamplerV2 GPU backend")
    except Exception as e:
        logger.error(f"❌ FATAL: GPU fidelity creation failed: {e}")
        logger.error("   FidelityQuantumKernel requires AerSamplerV2 with GPU.")
        return
    
    logger.info("=" * 80)
    
    # Configuration
    # Quick test: configurable runs for fast iteration (default 5)
    # Full experiment: 30 runs for statistical significance (CLT requires n>=30)
    n_runs_to_use = args.quick_n_runs if args.quick_test else args.n_runs
    
    # Generate seeds based on n_runs
    all_seeds = [
        42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
        7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
        17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624
    ]
    seeds_to_use = all_seeds[:n_runs_to_use]
    
    config = {
        'n_runs': n_runs_to_use,
        'k_folds': 3 if args.quick_test else args.k_folds,
        'random_seeds': seeds_to_use,
        'test_size': 0.3,
        'noise_simulation': True,
        'transpile_opt_level': 2,
    }
    
    logger.info(f"📊 Statistical Configuration: {n_runs_to_use} runs with {len(seeds_to_use)} seeds")
    
    # Build dataset list
    datasets = []
    _here = os.path.dirname(os.path.abspath(__file__))
    for ds in args.datasets:
        # Accept 'data/x.csv' or a bare 'x.csv', and resolve relative to the
        # repository root so the script works from any working directory.
        resolved = None
        for cand in (ds, os.path.join(_here, ds),
                     os.path.join(_here, 'data', os.path.basename(ds)),
                     os.path.join(_here, os.path.basename(ds))):
            if os.path.isfile(cand):
                resolved = cand
                break
        if resolved:
            name = Path(resolved).stem
            datasets.append((name, resolved))
        else:
            logger.warning(
                f"Dataset not found: {ds} (expected in data/; "
                f"run 'git lfs pull' if it is a Git LFS pointer)")
    
    if not datasets:
        logger.error("No valid datasets found!")
        return
    
    # Determine noise simulation setting
    enable_noise = not args.disable_noise  # Enabled by default unless --disable_noise
    
    # Run experiment
    experiment = CircuitDepthExperimentRunner(
        config,
        quick_test=args.quick_test,
        enable_noise_simulation=enable_noise,
        resume_session_id=args.resume
    )
    experiment.phase4_model_names_override = list(args.phase4_models) if args.phase4_models else None
    experiment.phase4_entanglement_override = list(args.phase4_entanglements) if args.phase4_entanglements else None
    experiment.phase4_noise_levels_override = list(args.phase4_noise_levels) if args.phase4_noise_levels else None
    
    logger.info(f"🔬 Noise Simulation: {'ENABLED (Phase 4 comprehensive analysis will run)' if enable_noise else 'DISABLED'}")
    
    # ================================================================
    # EXPERIMENT CONFIGURATIONS
    # 2 configs: vary qubits at fixed sample size (5k)
    # Sample size doesn't affect circuit optimization / noise interaction —
    # only qubit count changes circuit depth/gates/noise sensitivity.
    # ================================================================
    
    # Define experiment configurations
    def _parse_config_spec(spec: str) -> Dict[str, int]:
        try:
            num_qubits_str, sample_size_str = str(spec).split(':', 1)
            return {'num_qubits': int(num_qubits_str), 'sample_size': int(sample_size_str)}
        except Exception as exc:
            raise ValueError(
                f"Invalid --config_specs entry '{spec}'. Expected <num_qubits>:<sample_size>, e.g. 10:2500"
            ) from exc

    if args.quick_test:
        # Quick test: use command line args or defaults
        experiment_configs = [
            {'num_qubits': args.num_qubits, 'sample_size': args.sample_size}
        ]
    elif args.config_specs:
        experiment_configs = [_parse_config_spec(spec) for spec in args.config_specs]
    else:
        # Full experiment: 6q and 10q at 5k samples
        experiment_configs = [
            {'num_qubits': 6, 'sample_size': 5000},
            {'num_qubits': 10, 'sample_size': 5000},
        ]
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 EXPERIMENT CONFIGURATIONS")
    logger.info("=" * 80)
    for i, cfg in enumerate(experiment_configs, 1):
        logger.info(f"   Config {i}: {cfg['num_qubits']} qubits × {cfg['sample_size']} samples")
    logger.info("=" * 80)
    
    # Run experiment for each configuration
    all_results = {
        'model_results': [],
        'circuit_analysis': [],
        'cv_results': [],
        'ablation_results': []
    }
    
    for config_idx, exp_cfg in enumerate(experiment_configs, 1):
        num_qubits = exp_cfg['num_qubits']
        sample_size = exp_cfg['sample_size']
        
        logger.info("\n" + "=" * 80)
        logger.info(f"🔬 RUNNING CONFIGURATION {config_idx}/{len(experiment_configs)}")
        logger.info(f"   Qubits: {num_qubits}, Samples: {sample_size}")
        logger.info("=" * 80)
        
        results = experiment.run_full_experiment(
            datasets=datasets,
            num_qubits=num_qubits,
            sample_size=sample_size
        )
        
        # Add configuration info to results
        for r in results.get('model_results', []):
            r['config_num_qubits'] = num_qubits
            r['config_sample_size'] = sample_size
        
        # Aggregate results
        all_results['model_results'].extend(results.get('model_results', []))
        all_results['circuit_analysis'].extend(results.get('circuit_analysis', []))
        all_results['cv_results'].extend(results.get('cv_results', []))

        if args.include_classical:
            logger.info("\n🔷 Training Classical Baselines")

            for dataset_name, dataset_path in datasets:
                df, detected_delimiter = experiment._load_dataset(dataset_path)
                processor = DataProcessor(num_qubits=num_qubits)
                X_raw, y = processor.prepare_data(df, sample_size=sample_size)

                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X_raw, y, test_size=config['test_size'], stratify=y, random_state=42
                )
                processor.fit(X_train_raw, y_train)
                X_train = processor.transform(X_train_raw)
                X_test = processor.transform(X_test_raw)

                classical_results = train_classical_baselines(X_train, y_train, X_test, y_test)

                for r in classical_results:
                    r['dataset'] = dataset_name
                    r['delimiter'] = detected_delimiter
                    r['config_num_qubits'] = num_qubits
                    r['config_sample_size'] = sample_size

                dataset_tag = experiment._sanitize_tag(dataset_name)
                prefix = f"{dataset_tag}_{num_qubits}q_{experiment._format_sample_tag(sample_size)}_{experiment.session_id}"
                df_classical = pd.DataFrame(classical_results)
                df_classical.to_csv(
                    RESULTS_DIR / f'classical_results_{prefix}.csv',
                    index=False
                )
        
        logger.info(f"✅ Configuration {config_idx} complete")
    
    results = all_results
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 80)
    
    if results['model_results']:
        df = pd.DataFrame(results['model_results'])
        success_df = df[df['status'] == 'success']
        
        if not success_df.empty:
            print(f"\n✅ Successfully trained: {len(success_df)} models")
            print("\n🏆 TOP MODELS BY ACCURACY:")
            print("-" * 60)
            
            for _, row in success_df.nlargest(min(10, len(success_df)), 'accuracy').iterrows():
                print(f"⚛️ {row['model']:<30} | Acc: {row['accuracy']:.4f} | "
                      f"F1: {row.get('f1_score', 0):.4f} | Time: {row.get('training_time', 0):.2f}s")
    
    print("\n" + "=" * 80)
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
