import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----- 1. Scenario Generation -----

np.random.seed(42)


def normalize(arr):
    return arr / arr.sum() * 100

import os

#os.chdir(r"C:\Users\pramod.kumar\OneDrive - o9 Solutions\Documents\URBN_DATA\HML_data")

weekly_sales = pd.read_csv('pag_weekly_sales_data_with_inventory.csv')


# Step 1: Filter 2023 data
data_2023 = weekly_sales[weekly_sales['year'] == 2023]

# Step 2: Identify PAGs with at least 3 stylecolor_snum where qty > 0 in 2023
valid_pags = (
    data_2023[data_2023['qty'] > 0]
    .groupby('pag')['stylecolor_snum']
    .nunique()
    .reset_index(name='nonzero_stylecolor_count')
)

valid_pags = valid_pags[valid_pags['nonzero_stylecolor_count'] >= 3]['pag']

# Step 3: Identify PAGs that exist in both 2023 and 2024
pags_2023 = set(weekly_sales[weekly_sales['year'] == 2023]['pag'].unique())
pags_2024 = set(weekly_sales[weekly_sales['year'] == 2024]['pag'].unique())
common_pags = pags_2023 & pags_2024

# Step 4: Filter weekly_sales where pag is in both years AND passes 2023 logic
final_pags = valid_pags[valid_pags.isin(common_pags)]
upd_historical_data = weekly_sales[weekly_sales['pag'].isin(final_pags)]

top_pags = (
    upd_historical_data
    .groupby('pag', as_index=False)['qty']
    .sum()
    .query('qty > 0')
    .sort_values('qty', ascending=False)
    .head(30)  # top_pag is your desired count (e.g., 10)
)

# 2. Filter weekly_sales to include only those top PAGs
weekly_sales_top_pag = upd_historical_data[upd_historical_data['pag'].isin(top_pags['pag'])]


# df_historical_revenue = pd.read_csv('historical_revenue.csv')

# df_future_revenue = pd.read_csv('future_revenue.csv')

# df_input = pd.read_csv('option_input_df.csv')

# df_all = pd.concat([df_input])

def historical_option_data(df):
    filtered_historical_pag = df.copy()

    option_input_df = filtered_historical_pag[['pag','stylecolor_snum', 'qty']].drop_duplicates()

    agg_df = option_input_df.groupby(['pag', 'stylecolor_snum'])['qty'].sum().reset_index()
    agg_df['total_qty_pag'] = agg_df.groupby('pag')['qty'].transform('sum')

    agg_df['per_option_revenue_pct'] = agg_df['qty'] / agg_df['total_qty_pag'] * 100

    agg_df['option_rank'] = agg_df.groupby('pag')['qty'].rank(ascending=False, method='first')

    agg_df = agg_df.sort_values(['pag', 'option_rank'])

    agg_df['cum_revenue_pct'] = agg_df.groupby('pag')['per_option_revenue_pct'].cumsum()

    option_input_df = option_input_df.merge(
        agg_df[['pag', 'stylecolor_snum', 'option_rank', 'per_option_revenue_pct', 'cum_revenue_pct']],
        on=['pag', 'stylecolor_snum'],
        how='left'
    )
    agg_df = agg_df[['pag', 'option_rank', 'per_option_revenue_pct', 'cum_revenue_pct']].drop_duplicates()

    return agg_df


def width_data_preparation(df, top_pag = 100):
    df_sales = df.copy()

    historical_pag = df_sales[df_sales['year']== 2023]
    future_pag = df_sales[df_sales['year']== 2024]

    top_pags = historical_pag.groupby(['pag']).agg({'qty':'sum'}).reset_index().sort_values('qty', ascending=False)
    # top_pags = top_pags[top_pags['qty']>0]
    # top_pags = top_pags[0:top_pag]

    
    filtered_future_pag = future_pag[future_pag['pag'].isin(top_pags['pag'].unique())]
    
    # future_revenue = filtered_future_pag.groupby(['pag']).agg({'qty':'sum', 
    #                 'stylecolor_snum': lambda x:x.nunique()}).reset_index()
    
    future_revenue = (
    filtered_future_pag
    .groupby(['pag'])
    .agg(
        qty=('qty', 'sum'),
        stylecolor_snum_total=('stylecolor_snum', 'nunique'),
        stylecolor_snum_nonzero=('stylecolor_snum', lambda x: x[filtered_future_pag.loc[x.index, 'qty'] > 0].nunique())
    )
    .reset_index()
    )


    filtered_historical_pag = historical_pag[historical_pag['pag'].isin(top_pags['pag'].unique())]

    historical_revenue = (
    filtered_historical_pag
    .groupby(['pag'])
    .agg(
        qty=('qty', 'sum'),
        stylecolor_snum_total=('stylecolor_snum', 'nunique'),
        stylecolor_snum_nonzero=('stylecolor_snum', lambda x: x[filtered_historical_pag.loc[x.index, 'qty'] > 0].nunique())
    )
    .reset_index()
    )


    option_df = historical_option_data(filtered_historical_pag)


    return historical_revenue, future_revenue, option_df



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




def extend_curve_polynomial_with_inverse_decay(total_revenue, future_revenue, poly_func, fitted_per_option_rev_pct, x_full, df, decay_rate_length=5, max_extend=200, tol=1e-5):
    current_max_rank = df['option_rank'].max()
    
    extended_ranks = list(x_full)
    extended_per_option_rev = list(fitted_per_option_rev_pct)
    
    # Current cumulative revenue absolute
    cum_rev_abs = np.cumsum(extended_per_option_rev) / 100 * total_revenue
    current_cum_abs = cum_rev_abs[-1]
    
    # Step 1: Fit an exponential decay to the tail (ranks 6-10)
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    def inverse_decay(x, a, b):
        return a / (x + b)
    
    # Use the last 5 ranks for fitting the tail (adjustable)
    tail_length = min(decay_rate_length, len(extended_ranks))

    if len(x_full) >= tail_length:
        tail_ranks = x_full[-tail_length:]  # e.g., [6, 7, 8, 9, 10]
        print('tail_ranks', tail_ranks)

        tail_values = fitted_per_option_rev_pct[-tail_length:]  # e.g., [9.8, 3.2, 2.3, 2.1, 2.4]
        print('tail_values', tail_values)


        # Convert tail_ranks to numpy array to ensure positional indexing
        if isinstance(tail_ranks, pd.Series):
            tail_ranks = tail_ranks.to_numpy()
        
        # Normalize x to start at 0 for fitting
        tail_ranks_shifted = tail_ranks - tail_ranks[0]  # Now works with numpy array

        # Initial guesses: a = first tail value, b = small decay rate
        try:
            params, _ = curve_fit(inverse_decay, tail_ranks_shifted, tail_values, p0=[tail_values[0], 0.1], maxfev=10000)
            a, b = params
            print(f"Fitted exponential decay parameters: a={a:.3f}, b={b:.3f}")
        except RuntimeError:
            # Fallback if curve_fit fails: use last value with a small decay
            print("Exponential fit failed, using fallback method.")
            a = tail_values[-1]
            b = 0.05  # Small decay rate as fallback
    else:
        # Fallback for small datasets
        a = fitted_per_option_rev_pct[-1]
        b = 0.05
    
    # Step 2: Extrapolate using the fitted exponential decay
    rank = current_max_rank
    while current_cum_abs < future_revenue and rank < current_max_rank + max_extend:
        rank += 1
        # Compute the rank offset relative to the start of the tail
        rank_offset = rank - tail_ranks[0]
        next_rev_pct = max(inverse_decay(rank_offset, a, b), tol)
        
        if next_rev_pct < tol:
            print(f"Revenue increment too small at rank={rank}. Stopping extrapolation.")
            break
        
        extended_ranks.append(rank)
        extended_per_option_rev.append(next_rev_pct)
        current_cum_abs += (next_rev_pct / 100) * total_revenue
    
    extended_cum_pct = np.cumsum(extended_per_option_rev)
    extended_cum_abs = extended_cum_pct / 100 * total_revenue
    
    extended_df = pd.DataFrame({
        'option_rank': extended_ranks,
        'per_option_revenue_pct': extended_per_option_rev,
        'cum_revenue_pct': extended_cum_pct,
        'cum_revenue_abs': extended_cum_abs
    })
    
    return extended_df




# ----- 3. Streamlit UI -----



st.set_page_config(layout="wide")

st.title("📈 Long-Tail Option Solver")

# Layout with two columns
col_left, col_right = st.columns([1, 3])  # Left is narrower, right is wider

with col_left:


    # # Input weeks
    # col1, col2 = st.columns(2)
    # with col1:
    #     start_week = int(st.number_input(
    #         "Start Week",
    #         value=1,
    #         step=1,
    #         min_value=1,
    #         max_value=40,  # So end_week = start_week + 12 ≤ 52
    #         format="%d"
    #     ))

    # with col2:
    #     min_end_week = start_week + 12
    #     end_week = int(st.number_input(
    #         "End Week",
    #         value=min_end_week,
    #         step=1,
    #         min_value=min_end_week,
    #         max_value=52,
    #         format="%d"
    #     ))

    # filtered_data = weekly_sales[(weekly_sales['week']>=start_week)&(weekly_sales['week']<=end_week)]



    # df_historical_revenue, df_future_revenue, df_input = width_data_preparation(filtered_data)
    
    # df_all = pd.concat([df_input])

    # scenario_list = df_all['pag'].unique()
    # selected_scenario = st.selectbox("Select a PAG", scenario_list)



    # Get all unique scenarios from full dataset
    scenario_list = weekly_sales_top_pag['pag'].unique()
    selected_scenario = st.selectbox("Select a PAG", scenario_list)

    # Filter data based on selected scenario
    scenario_data = weekly_sales_top_pag[weekly_sales_top_pag['pag'] == selected_scenario]

    # print(scenario_data.head(2))

    # # Get min and max week values for this scenario
    # min_week = int(scenario_data['week'].min())
    # max_week = int(scenario_data['week'].max() - 4)  # ensure 12-week gap is possible

    # # Handle edge case where there's not enough data
    # if max_week < min_week:
    #     st.warning("Not enough data for 12-week selection in this scenario.")
    #     st.stop()

    # # Input weeks AFTER scenario is selected
    # col1, col2 = st.columns(2)
    # with col1:
    #     start_week = int(st.number_input(
    #         "Start Week",
    #         value=min_week,
    #         step=1,
    #         min_value=min_week,
    #         max_value=max_week,  # So end_week = start_week + 12 ≤ actual max
    #         format="%d"
    #     ))

    # with col2:
    #     min_end_week = start_week + 4
    #     actual_max_week = int(scenario_data['week'].max())
    #     end_week = int(st.number_input(
    #         "End Week",
    #         value=min_end_week,
    #         step=1,
    #         min_value=min_end_week,
    #         max_value=actual_max_week,
    #         format="%d"
    #     ))

    # # Filter based on selected scenario and weeks
    # filtered_data = scenario_data[(scenario_data['week'] >= start_week) & 
    #                             (scenario_data['week'] <= end_week)]

    filtered_data = scenario_data.copy()

    # Continue processing
    df_historical_revenue, df_future_revenue, df_input = width_data_preparation(filtered_data)
    df_all = pd.concat([df_input])

    print(df_historical_revenue[df_historical_revenue['pag'] == selected_scenario])
    # print(df_historical_revenue.loc[df_historical_revenue['pag'] == selected_scenario, 'qty'])

    # Get selected scenario data
    df = df_all[df_all['pag'] == selected_scenario]
    df = df.sort_values(['option_rank'], ascending=True)

    historical_revenue = df_historical_revenue.loc[df_historical_revenue['pag'] == selected_scenario, 'qty'].iloc[0]
    target_revenue = df_future_revenue.loc[df_future_revenue['pag'] == selected_scenario, 'qty'].iloc[0]
    target_selected_options = df_future_revenue.loc[df_future_revenue['pag'] == selected_scenario, 'stylecolor_snum_total'].iloc[0]
    target_selected_nonzero_options = df_future_revenue.loc[df_future_revenue['pag'] == selected_scenario, 'stylecolor_snum_nonzero'].iloc[0]

    
    # Revenue inputs
    col1, col2 = st.columns(2)
    with col1:
        # total_revenue = st.number_input("Total Historical Revenue", value=10000.0)
        total_revenue = int(st.number_input("Total Historical Revenue", 
                                   value=historical_revenue, 
                                   step=100, 
                                   min_value=0, 
                                   format="%d"))
    
    with col2:
        # future_revenue = st.number_input("Target Future Revenue", value=14000.0)
        future_revenue = int(st.number_input("Target Future Revenue", 
                                   value=target_revenue, 
                                   step=100, 
                                   min_value=0, 
                                   format="%d"))


    # Method 1

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

    st.markdown("###### 📌 Method 1 (Inverse Decay)")

    if required_options:
        st.success(
            f"""
            - 🔢 **Historical Options**: {historical_options}
            - 🪃 **Long-Tail Cutoff**: {long_tail_rank}  
            - 🎯 **Required Options**: {required_options}  
            - ➕ **Additional Options Needed**: {required_options - historical_options}
            - 🧮 **Acutal Options**:{target_selected_options}
            - 📉 **Acutal Options with Sales**:{target_selected_nonzero_options}
            """
        )
    else:
        st.error(
            f"""
            ❌ Could not reach the future revenue target with extrapolation.  
            - 📉 **Historical Options**: {historical_options}
            """
        )

    # Method 2

    # col3, col4 = st.columns(2)
    
    # with col3:
    
    #     degree = int(st.number_input("Polynomial Degree", 
    #                                 value=3, 
    #                                 step=1, 
    #                                 min_value=2,
    #                                 max_value=6, 
    #                                 format="%d"))
                            
    # with col4:
    #     decay_length = int(st.number_input("#Options for Decay", 
    #                                 value=5, 
    #                                 step=1, 
    #                                 min_value=1,
    #                                 max_value=int(df['option_rank'].max()), 
    #                                 format="%d"))


    # poly_coeffs = np.polyfit(df['option_rank'], df['per_option_revenue_pct'], degree)
    # poly_func = np.poly1d(poly_coeffs)
    # print(f"Polynomial coefficients (degree {degree}): {poly_coeffs}")

    # # Calculate fitted values over historical range
    # x_full_poly = np.arange(1, df['option_rank'].max() + 1)
    # fitted_per_option_rev_pct_poly = poly_func(x_full_poly)

    # # Scale factor to ensure fitted values sum to 100%
    # scale_factor_poly = df['per_option_revenue_pct'].sum() / fitted_per_option_rev_pct_poly.sum()
    # print(f"Scale factor: {scale_factor_poly:.2f}")
    # fitted_per_option_rev_pct_poly *= scale_factor_poly


    # extended_df_poly =  extend_curve_polynomial_with_inverse_decay(total_revenue, future_revenue, poly_func, fitted_per_option_rev_pct_poly, 
    #                     x_full, df, 
    #                     decay_rate_length=decay_length,
    #                     max_extend=200, tol=1e-5)

    # # extended_df_poly = extend_curve_inverse_decay(df, total_revenue, future_revenue,fitted_per_option_rev_pct_poly
    # #                                         inverse_decay, params_inv, scale_factor)
    # required_options_poly = find_required_options(extended_df_poly, future_revenue)
    
    # st.markdown("###### 📌 Method 2 (Poly Fit)")

    # if required_options_poly:
    #     st.success(
    #         f"""
    #         - 🔢 **Historical Options**: {historical_options}
    #         - 🪃 **Long-Tail Cutoff**: {long_tail_rank}  
    #         - 🎯 **Required Options**: {required_options_poly}  
    #         - ➕ **Additional Options Needed**: {required_options_poly - historical_options}
    #         - 🧮 **Acutal Options**:{target_selected_options}
    #         """
    #     )
    # else:
    #     st.error(
    #         f"""
    #         ❌ Could not reach the future revenue target with extrapolation.  
    #         - 📉 **Historical Options**: {historical_options}
    #         """
    #     )
    

with col_right:


    st.markdown("##### Target Revenue and Options (Method 1)")
    fig1 = plt.figure(figsize=(10, 3))
    
    plt.plot(df['option_rank'], df['cum_revenue_pct'] / 100 * total_revenue,
         'o-', label='Actual Revenue', markersize=3,)
    plt.plot(x_full, fitted_per_option_rev_pct.cumsum() / 100 * total_revenue,
            's--', label='Fitted Curve', markersize=3,)
    plt.plot(extended_df['option_rank'], extended_df['cum_revenue_abs'],
            'd-.', label='Extended Curve', markersize=3,)
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



    # st.markdown("##### Target Revenue and Options (Method 2)")
    # fig2 = plt.figure(figsize=(10, 3))
    
    # plt.plot(df['option_rank'], df['cum_revenue_pct'] / 100 * total_revenue,
    #      'o-', label='Actual Revenue', markersize=3,)
    # plt.plot(x_full, fitted_per_option_rev_pct_poly.cumsum() / 100 * total_revenue,
    #         's--', label='Fitted Curve', markersize=3,)
    # plt.plot(extended_df_poly['option_rank'], extended_df_poly['cum_revenue_abs'],
    #         'd-.', label='Extended Curve', markersize=3,)
    # plt.axhline(future_revenue, color='red', linestyle='--', label='Future Revenue')
    # if required_options_poly:
    #     plt.axvline(required_options_poly, color='green', linestyle=':', label=f'Required Options = {required_options_poly}')

    # if not long_tail_candidates.empty:
    #     long_tail_rank = int(long_tail_candidates['option_rank'].iloc[0])
    #     plt.axvline(long_tail_rank, color='purple', linestyle=':', label=f'Long-Tail Cutoff = {long_tail_rank}')

    # plt.xlabel('Options')
    # plt.ylabel('Cumulative Revenue')
    # plt.grid(True)
    # plt.gca().xaxis.get_major_locator().set_params(integer=True)

    # plt.legend(fontsize=5)  
    # st.pyplot(fig2)
    # plt.close(fig2)
