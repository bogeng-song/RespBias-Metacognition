"""
msdt_model.py
Multi-Alternative Signal Detection Theory (M-SDT) parameter estimation using PyMC.

Estimates perceptual sensitivity (d') and category-specific response bias (b)
for multi-alternative perceptual decision-making tasks.

Features:
- Dynamically adapts to n-AFC (e.g., 4-choice or 8-choice).
- Supports both single-noise datasets (Experiment 1) and multi-noise datasets (Experiment 2).
- When multiple noise levels are present, it estimates condition-specific d' parameters
  while sharing a single bias vector (b) per participant.
- Executes MCMC sampling in parallel across participants using joblib.
"""

import argparse
import os
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import warnings
from joblib import Parallel, delayed
import multiprocessing as mp

warnings.filterwarnings('ignore')

def prepare_global_mappings(df):
    """
    Generates mappings across the entire dataset to ensure label consistency 
    across all subjects, even if a subject omitted a certain response category.
    """
    # 1. Map Stimulus and Response labels to contiguous 0-indexed integers
    unique_labels = np.sort(np.unique(df[['Stimulus', 'Response']].dropna().values))
    label_map = {val: i for i, val in enumerate(unique_labels)}
    
    # 2. Handle Noise Levels (If 'Noise' column doesn't exist, assume single noise level)
    if 'Noise' not in df.columns:
        df['Noise'] = 'Standard'
        
    unique_noise = sorted(df['Noise'].unique())
    noise_map = {val: i for i, val in enumerate(unique_noise)}
    
    return label_map, noise_map, unique_labels, unique_noise

def prepare_individual_data(subject_df, label_map, noise_map, n_alternatives):
    """Formats behavioral dataframe into 0-indexed tensors for a single subject."""
    df = subject_df.copy()
    if 'Noise' not in df.columns:
        df['Noise'] = 'Standard'

    df['Stimulus_idx'] = df['Stimulus'].map(label_map)
    df['Response_idx'] = df['Response'].map(label_map)
    df['noise_idx'] = df['Noise'].map(noise_map)

    # Drop any rows with NaN mappings
    df = df.dropna(subset=['Stimulus_idx', 'Response_idx', 'noise_idx'])

    data_dict = {
        'n_trials': len(df),
        'n_categories': n_alternatives,
        'n_noise_levels': len(noise_map),
        'stimulus': df['Stimulus_idx'].values.astype(int),
        'response': df['Response_idx'].values.astype(int),
        'noise_idx': df['noise_idx'].values.astype(int)
    }
    return data_dict

def build_sdt_model(data_dict):
    """Constructs the PyMC generative model for M-SDT."""
    n_categories = data_dict['n_categories']
    n_bias_params = n_categories - 1  # Reference category fixed to 0
    n_noise = data_dict['n_noise_levels']

    with pm.Model() as model:
        # 1. Sensitivity (d') - Separate parameter for each noise level
        d_raw = pm.Normal('d_raw', mu=1.0, sigma=1.0, shape=n_noise)
        d = pm.Deterministic('d', pm.math.exp(d_raw)) # Constrain d' to be positive

        # 2. Bias (b) - SHARED across all noise levels
        b_raw = pm.Normal('b_raw', mu=0.0, sigma=1.0, shape=n_bias_params)
        b_zeros = pt.zeros(1)
        b = pm.Deterministic('b', pt.concatenate([b_raw, b_zeros]))

        # Convert data to PyTensor variables
        stim_idx = pt.as_tensor_variable(data_dict['stimulus'])
        noise_idx = pt.as_tensor_variable(data_dict['noise_idx'])

        # 3. Compute Utilities
        # One-hot encode stimulus position
        stim_onehot = pt.eye(n_categories)[stim_idx]

        # Select the correct d' parameter for the noise level of the current trial
        trial_d = d[noise_idx][:, None]

        # Total Utility = (Signal * d') + Shared Bias
        utilities = (stim_onehot * trial_d) + b[None, :]

        # 4. Likelihood (SoftMax choice rule)
        pm.Categorical('response', logit_p=utilities, observed=data_dict['response'])

    return model

def fit_individual_subject(subject_data, subject_id, label_map, noise_map, n_alternatives, draws, tune, chains):
    """Fits the M-SDT model for a single subject."""
    try:
        data_dict = prepare_individual_data(subject_data, label_map, noise_map, n_alternatives)
        model = build_sdt_model(data_dict)

        with model:
            # Note: cores=1 ensures PyMC doesn't conflict with outer Joblib parallelization
            trace = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=1,
                target_accept=0.95, random_seed=42,
                progressbar=False, return_inferencedata=True,
                compute_convergence_checks=False
            )

        # Extract posterior means
        d_means = trace.posterior['d'].mean(axis=(0, 1)).values
        b_means = trace.posterior['b'].mean(axis=(0, 1)).values

        # Convergence check (R-hat)
        r_hat = az.rhat(trace)
        max_rhat = max([float(r_hat[v].max()) for v in r_hat.data_vars])

        return {
            'subject_id': subject_id, 
            'success': True,
            'd_means': d_means, 
            'b_means': b_means,
            'max_rhat': max_rhat,
        }
    except Exception as e:
        return {'subject_id': subject_id, 'success': False, 'error': str(e)}

def analyze_dataset(df, n_alternatives=None, draws=2000, tune=1000, chains=4, n_jobs=-1):
    """Distributes the subject-level PyMC fitting across available CPU cores."""
    sub_ids = df['Sub_id'].unique()
    
    # Generate unified mappings
    label_map, noise_map, unique_labels, unique_noise = prepare_global_mappings(df)
    
    if n_alternatives is None:
        n_alternatives = len(unique_labels)
        
    n_cores = mp.cpu_count() if n_jobs == -1 else n_jobs
    print(f"🚀 Processing {len(sub_ids)} subjects in parallel utilizing {n_cores} cores... (This may take a while)")

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_individual_subject)(
            df[df['Sub_id'] == sid], sid, label_map, noise_map, n_alternatives, draws, tune, chains
        ) for sid in sub_ids
    )

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if failed:
        print(f"⚠️ {len(failed)} subjects failed to converge:")
        for f in failed: print(f"   - Subject {f['subject_id']}: {f['error']}")

    if not successful:
        print("❌ No subjects fitted successfully!")
        return None

    print(f"✅ Successfully fitted {len(successful)}/{len(sub_ids)} subjects.")
    
    summary = {
        'subject_ids': [r['subject_id'] for r in successful],
        'b_means': np.array([r['b_means'] for r in successful]),
        'd_means': [r['d_means'] for r in successful],
        'max_rhats': np.array([r['max_rhat'] for r in successful]),
        'unique_labels': unique_labels,
        'unique_noise': unique_noise
    }
    return summary

def save_results(summary, output_path):
    """Saves the extracted d' and bias parameters to a long-format CSV."""
    if not summary:
        return
        
    n_alts = summary['b_means'].shape[1]
    labels = summary['unique_labels']
    noise_levels = summary['unique_noise']
    rows = []
    
    for i, sub_id in enumerate(summary['subject_ids']):
        d_vals = summary['d_means'][i]
        
        for alt_idx in range(n_alts):
            row = {
                'Sub_id': sub_id,
                'Alternative': labels[alt_idx] if alt_idx < len(labels) else alt_idx + 1,
                'Bias_Value': summary['b_means'][i, alt_idx],
                'Max_Rhat': summary['max_rhats'][i]
            }
            # Add d_prime columns dynamically based on how many noise levels existed
            for n_idx, d_val in enumerate(d_vals):
                label = noise_levels[n_idx]
                row[f'd_prime_Noise_{label}'] = d_val

            rows.append(row)

    df_long = pd.DataFrame(rows)
    df_long.to_csv(output_path, index=False)
    print(f"💾 Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Alternative SDT Estimation using PyMC")
    parser.add_argument('--data_path', type=str, required=True, help="Path to behavioral data CSV")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save M-SDT results CSV")
    parser.add_argument('--n_alts', type=int, default=None, help="Number of choices (e.g., 4 or 8)")
    parser.add_argument('--draws', type=int, default=2000, help="MCMC draws")
    parser.add_argument('--tune', type=int, default=1000, help="MCMC tuning steps")
    parser.add_argument('--chains', type=int, default=4, help="MCMC chains")
    parser.add_argument('--jobs', type=int, default=-1, help="Number of parallel jobs (-1 for all cores)")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    
    # Required columns check (Standardizes to Sub_id, Stimulus, Response, Noise automatically)
    col_map = {c.lower(): c for c in df.columns}
    if 'subject' in col_map and 'Sub_id' not in df.columns: 
        df.rename(columns={col_map['subject']: 'Sub_id'}, inplace=True)
    if 'stim' in col_map and 'Stimulus' not in df.columns: 
        df.rename(columns={col_map['stim']: 'Stimulus'}, inplace=True)
    if 'response' in col_map and 'Response' not in df.columns: 
        df.rename(columns={col_map['response']: 'Response'}, inplace=True)
    if 'difficulty' in col_map and 'Noise' not in df.columns: 
        df.rename(columns={col_map['difficulty']: 'Noise'}, inplace=True)
        
    for col in ['Sub_id', 'Stimulus', 'Response']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Ensure your CSV has Sub_id, Stimulus, Response.")

    # Run Analysis
    summary = analyze_dataset(
        df, 
        n_alternatives=args.n_alts, 
        draws=args.draws, 
        tune=args.tune, 
        chains=args.chains, 
        n_jobs=args.jobs
    )
    
    # Save Outputs
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_results(summary, args.output_path)

if __name__ == "__main__":
    # Protects PyMC multiprocessing in script executions
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()