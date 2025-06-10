import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(42)

import numpy as np
import pandas as pd

def normalize(arr):
    return arr / arr.sum() * 100

def generate_scenario(option_count, method, param=None, param2=None):
    rank = np.arange(1, option_count + 1)
    if method == 'exp_decay':
        values = np.exp(-param * (rank - 1))
    elif method == 'powerlaw':
        values = 1 / (rank ** param)
    elif method == 'reverse_powerlaw':
        values = 1 / ((rank[::-1] + 1) ** param)
    elif method == 'linear_decay':
        values = np.linspace(1, 0.1, option_count)
    elif method == 'uniform':
        values = np.ones(option_count)
    elif method == 'dirichlet':
        values = np.random.dirichlet(alpha=np.full(option_count, param))
    elif method == 'gaussian_bump':
        values = np.exp(-((rank - param) ** 2) / (2 * param2 ** 2))
    elif method == 'step_function':
        step_size = option_count // param
        values = np.ones(option_count)
        for i in range(param):
            start = i * step_size
            end = (i + 1) * step_size if i < param - 1 else option_count
            values[start:end] *= (param - i) / param
    elif method == 'sigmoid_decay':
        values = 1 / (1 + np.exp(param * (rank - param2)))
    elif method == 'periodic_oscillation':
        decay = np.exp(-0.1 * (rank - 1))
        oscillation = param2 * np.sin(param * rank)
        values = decay * (1 + oscillation)
        values = np.maximum(values, 0.01)
    elif method == 'hybrid_decay':
        values = np.zeros(option_count)
        split = int(param)
        values[:split] = np.exp(-param2 * (rank[:split] - 1))
        values[split:] = np.linspace(values[split-1], 0.1, option_count - split)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    pct = normalize(values)
    cum_pct = np.cumsum(pct)
    
    return pd.DataFrame({
        'option_rank': rank,
        'per_option_revenue_pct': pct,
        'cum_revenue_pct': cum_pct
    })

# Updated scenario_specs with explicit param2 for all entries
scenario_specs = [
    # Original scenarios
    ('exp_decay', 10, 0.3, None),
    ('exp_decay', 20, 0.1, None),
    ('powerlaw', 15, 1.5, None),
    ('reverse_powerlaw', 12, 2.0, None),
    ('linear_decay', 18, None, None),
    ('uniform', 25, None, None),
    ('dirichlet', 30, 0.5, None),
    ('dirichlet', 25, 2, None),
    # New scenarios
    ('gaussian_bump', 20, 10, 3),
    ('step_function', 15, 3, None),
    ('sigmoid_decay', 25, 0.5, 12),
    ('periodic_oscillation', 30, 0.5, 0.3),
    ('hybrid_decay', 20, 10, 0.2),
    ('exp_decay', 15, 0.5, None),
    ('powerlaw', 25, 2.5, None),
    ('reverse_powerlaw', 20, 1.0, None),
    ('linear_decay', 30, None, None),
    ('uniform', 10, None, None),
    ('dirichlet', 20, 1.0, None),
    ('gaussian_bump', 30, 5, 2),
]

# Generate DataFrames for each scenario with metadata
scenario_dfs = []
for idx, (method, count, param, param2) in enumerate(scenario_specs, start=1):
    df = generate_scenario(count, method, param, param2)
    df['scenario_name'] = f'Scenario {idx}'
    
    total_pct = df['per_option_revenue_pct'].sum()
    if not np.isclose(total_pct, 100, rtol=1e-5):
        print(f"Warning: Scenario {idx} per_option_revenue_pct sums to {total_pct}, not 100")
    
    scenario_dfs.append(df)

df_all = pd.concat(scenario_dfs, ignore_index=True)


def inverse_decay(x, a, b):
    return a / (x + b)

def extend_curve_inverse_decay(df, total_revenue, future_revenue, decay_func, params, scale_factor, max_extend=200, tol=1e-5):
    x_full = df['option_rank']
    fitted_per_option_rev_pct = decay_func(x_full, *params) * scale_factor

    extended_ranks = list(x_full)
    extended_per_option_rev = list(fitted_per_option_rev_pct)

    cum_rev_abs = np.cumsum(extended_per_option_rev) / 100 * total_revenue
    current_cum_abs = cum_rev_abs[-1]
    rank = x_full.max()

    while current_cum_abs < future_revenue and rank < x_full.max() + max_extend:
        rank += 1
        next_rev_pct = decay_func(rank, *params) * scale_factor
        if next_rev_pct < tol:
            break
        extended_ranks.append(rank)
        extended_per_option_rev.append(next_rev_pct)
        current_cum_abs += (next_rev_pct / 100) * total_revenue

    extended_cum_pct = np.cumsum(extended_per_option_rev)
    extended_cum_abs = extended_cum_pct / 100 * total_revenue

    return pd.DataFrame({
        'option_rank': extended_ranks,
        'per_option_revenue_pct': extended_per_option_rev,
        'cum_revenue_pct': extended_cum_pct,
        'cum_revenue_abs': extended_cum_abs
    })

def find_required_options(extended_df, future_revenue):
    candidates = extended_df[extended_df['cum_revenue_abs'] >= future_revenue]
    if candidates.empty:
        return None
    return int(candidates['option_rank'].iloc[0])

# ----- 3. Streamlit UI -----



st.set_page_config(layout="wide")

st.title("📈 Long-Tail Option Solver")

# Layout with two columns
col_left, col_right = st.columns([1, 3])  # Left is narrower, right is wider

with col_left:


    scenario_list = df_all['scenario_name'].unique()
    selected_scenario = st.selectbox("Select a Scenario", scenario_list)

    # Revenue inputs
    col1, col2 = st.columns(2)
    with col1:
        # total_revenue = st.number_input("Total Historical Revenue", value=10000.0)
        total_revenue = int(st.number_input("Total Historical Revenue", 
                                   value=10000, 
                                   step=100, 
                                   min_value=0, 
                                   format="%d"))
    
    with col2:
        # future_revenue = st.number_input("Target Future Revenue", value=14000.0)
        future_revenue = int(st.number_input("Target Future Revenue", 
                                   value=12000, 
                                   step=100, 
                                   min_value=0, 
                                   format="%d"))
    # Get selected scenario data
    df = df_all[df_all['scenario_name'] == selected_scenario]

    # Fit inverse decay
    x_full = df['option_rank']
    y_full = df['per_option_revenue_pct']
    params_inv, _ = curve_fit(inverse_decay, x_full, y_full, p0=[50, 0.5], maxfev=10000)

    # Scale fitted curve
    fitted_per_option_rev_pct = inverse_decay(x_full, *params_inv)
    scale_factor = y_full.sum() / fitted_per_option_rev_pct.sum()
    fitted_per_option_rev_pct *= scale_factor

    # Extend curve
    extended_df = extend_curve_inverse_decay(df, total_revenue, future_revenue,
                                            inverse_decay, params_inv, scale_factor)
    required_options = find_required_options(extended_df, future_revenue)


    # Add long-tail cutoff at 90% of historical total cumulative revenue
    historical_total = total_revenue  # $1,000,000
    long_tail_target = 0.9 * historical_total  # 90% of 1,000,000 = 900,000
    long_tail_candidates = df[df['cum_revenue_pct'] / 100 * total_revenue >= long_tail_target]
    long_tail_rank = int(long_tail_candidates['option_rank'].iloc[0])

    # Options summary
    historical_options = df['option_rank'].nunique()
    st.markdown("#### 📌 Options Summary")
    if required_options:
        st.success(
            f"""
            - 📉 **Historical Options**: {historical_options}
            - 📊 **Long-Tail Cutoff**: {long_tail_rank}  
            - 🚀 **Required Top-N Options**: {required_options}  
            - ➕ **Additional Options Needed**: {required_options - historical_options}
            """
        )
    else:
        st.error(
            f"""
            ❌ Could not reach the future revenue target with extrapolation.  
            - 📉 **Historical Options**: {historical_options}
            """
        )

with col_right:


    st.markdown("##### Target Revenue and Options")
    fig1 = plt.figure(figsize=(10, 3))
    
    plt.plot(df['option_rank'], df['cum_revenue_pct'] / 100 * total_revenue,
         'o-', label='Actual Revenue')
    plt.plot(x_full, fitted_per_option_rev_pct.cumsum() / 100 * total_revenue,
            's--', label='Fitted Curve')
    plt.plot(extended_df['option_rank'], extended_df['cum_revenue_abs'],
            'd-.', label='Extended Curve')
    plt.axhline(future_revenue, color='red', linestyle='--', label='Future Revenue')
    if required_options:
        plt.axvline(required_options, color='green', linestyle=':', label=f'Required Options = {required_options}')

    if not long_tail_candidates.empty:
        long_tail_rank = int(long_tail_candidates['option_rank'].iloc[0])
        plt.axvline(long_tail_rank, color='purple', linestyle=':', label=f'Long-Tail Cutoff = {long_tail_rank}')

    plt.xlabel('Options')
    plt.ylabel('Cumulative Revenue')
    plt.grid(True)
    plt.gca().xaxis.get_major_locator().set_params(integer=True)

    plt.legend(fontsize=5)  
    st.pyplot(fig1)
    plt.close(fig1)


    st.markdown(f"##### Historical Options")
    fig2 = plt.figure(figsize=(10, 3))
    cum_rev_abs_target = df['cum_revenue_pct'] / 100 * total_revenue
    plt.plot(df['option_rank'], cum_rev_abs_target, marker='o')
    plt.xlabel('Options')
    plt.ylabel('Cumulative Revenue')
    # plt.title(f'Cumulative Revenue - {selected_scenario}')
    plt.grid(True)
    plt.gca().xaxis.get_major_locator().set_params(integer=True)
    st.pyplot(fig2)
    plt.close(fig2)
