import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# ----------------------------------------
# STEP 1: Preprocessing - Discretise Wt
# ----------------------------------------
def discretise_Wt(df, k=200, bins=(-0.5, 0.5), labels=(-1, 0, 1), plot=True):
    """
    Discretise scaled Wt into categories -1, 0, +1 based on thresholds.
    Returns updated DataFrame and empirical probabilities.
    Optionally plots a histogram grouped by intervals (in %).
    """

    df = df.copy()
    df['scaled_Wt'] = df['Wt'] * k
    bin_edges = [-np.inf] + list(bins) + [np.inf]
    df['e_t'] = pd.cut(df['scaled_Wt'], bins=bin_edges, labels=labels, include_lowest=True).astype(int)

    # Generate interval labels dynamically for plotting
    plot_labels = []
    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        # Format label
        if left == -np.inf:
            label = f"(-inf, {right}]"
        elif right == np.inf:
            label = f"({left}, inf)"
        else:
            label = f"({left}, {right}]"
        plot_labels.append(label)

    # Assign intervals for plotting
    df['scaled_Wt_bin'] = pd.cut(df['scaled_Wt'], bins=bin_edges, labels=plot_labels, include_lowest=True, right=True)

    # Empirical probabilities
    prob_dist = df['e_t'].value_counts(normalize=True).sort_index()
    p = {state: prob_dist.get(state, 0) for state in labels}

    print("Empirical probabilities:")
    print(f"P(-1) = {p[-1]:.4f}, P(0) = {p[0]:.4f}, P(+1) = {p[1]:.4f}")

    if plot:
        # Count the frequency of each bin and convert to percentage
        counts = df['scaled_Wt_bin'].value_counts(sort=False)
        percentages = 100 * counts / counts.sum()

        # Plot the histogram grouped by categories
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(plot_labels, percentages, edgecolor='black')

        # Add percentage labels on each bar, with a small offset to avoid overlap
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(percentages) * 0.03,  # 3% of max height as offset
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=11
            )

        ax.set_xlabel('Intervals of: ΔWt*k')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Histogram of scaled ΔWt (grouped by intervals)')
        ax.grid(axis='y')
        plt.xticks(rotation=45)
        plt.ylim(0, max(percentages) * 1.15)  # Add extra space on top
        plt.tight_layout()
        plt.show()

    return df, p

# ----------------------------------------
# STEP 2: Simulate discrete random walk
# ----------------------------------------
def simulate_multiple_discrete_paths(start_value, p_dict, horizon=252, n_paths=1000, seed=None):
    """
    Simulate multiple discrete random walk paths.

    Parameters:
    - start_value: starting point for all paths (usually last scaled_Wt)
    - p_dict: dictionary with keys -1, 0, +1 and their associated probabilities
    - horizon: number of time steps to simulate
    - n_paths: number of paths to generate
    - seed: random seed for reproducibility

    Returns:
    - paths: np.ndarray of shape (n_paths, horizon + 1)
    [
        [start_value, sv+1, sv+1-1, sv+1-1-1, ++0...] trayectoria 1
        [start_value, sv+1, sv+1-1, sv+1-1+1, +0...] trayectoria 2
        [...] trayectoria ...
    ]
    """
    if seed is not None:
        random.seed(seed)

    probs = [-1, 0, 1]
    weights = [p_dict[-1], p_dict[0], p_dict[1]]
    
    paths = np.zeros((n_paths, horizon + 1))
    paths[:, 0] = start_value

    for t in range(1, horizon + 1):
        steps = random.choices(probs, weights=weights, k=n_paths)
        paths[:, t] = paths[:, t - 1] + steps

    return paths # 1000 simulated scaled Wt paths 

# ----------------------------------------
# STEP 3: Forecast from multiple paths
# ----------------------------------------
def forecast_from_multiple_paths(train_df, paths, k, mu, log_rate_col='log_rate'):
    """
    Convert multiple simulated scaled_Wt paths to forecasted log_rates and rates.
    Returns a DataFrame with mean forecast, and confidence intervals (2.5%, 97.5%).
    
    Parameters:
    - train_df: historical DataFrame with last log_rate
    - paths: array of shape (n_paths, horizon + 1) with simulated scaled_Wt
    - k: scale factor used in Wt discretisation
    - mu: estimated drift
    - log_rate_col: name of log_rate column in train_df

    Returns:
    - df_summary: DataFrame with index of future dates and columns:
    #   shape (horizon,6)
        ['log_rate_mean', 'log_rate_lower', 'log_rate_upper',
         'rate_mean', 'rate_lower', 'rate_upper']
    """
    X_o = train_df[log_rate_col].iloc[-1]
    n_paths, T = paths.shape
    horizon = T - 1  # exclude initial value

    # Steps = diff / k (convert scaled_Wt to e_t)
    steps = np.diff(paths, axis=1) / k  # shape (n_paths, horizon)
    W_t_paths = np.cumsum(steps, axis=1)  # shape (n_paths, horizon)

    time_steps = np.arange(1, horizon + 1)
    drift = mu * time_steps  # shape (horizon,)

    # Add drift and X_o to each path 
    # predicted log-rates = X_o + drift + W_t_paths
    predicted_log_rates = X_o + drift + W_t_paths  # shape (n_paths, horizon)
    predicted_rates = np.exp(predicted_log_rates)  # shape (n_paths, horizon)

    # Create DataFrame
    last_date = train_df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    df_paths = pd.DataFrame(predicted_rates, columns=future_dates)

    return df_paths

# ----------------------------------------
# STEP 4: Plot some forecast paths vs historical data
# ----------------------------------------
def plot_multiple_paths_vs_history(train_df, df_paths, rate_col='rate', n_show=30):
    """
    Plot historical data and multiple forecasted paths.
    
    Parameters:
    - train_df: DataFrame with historical data (datetime index)
    - df_paths: DataFrame with simulated forecast paths (n_paths x horizon)
    - rate_col: column with historical exchange rate
    - n_show: number of paths to display
    """
    df_hist = train_df[[rate_col]].copy()
    
    plt.figure(figsize=(12, 6))
    
    # Historical
    plt.plot(df_hist.index, df_hist[rate_col], label='Historical', linewidth=2)

    # Show up to n_show paths
    for i in range(min(n_show, df_paths.shape[0])):
        plt.plot(df_paths.columns, df_paths.iloc[i], color='gray', alpha=0.5, linewidth=1)

    plt.axvline(x=df_hist.index[-1], color='gray', linestyle=':', label='Forecast start')
    plt.title(f'Historical vs {df_paths.shape[0]} Monte Carlo Paths')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()