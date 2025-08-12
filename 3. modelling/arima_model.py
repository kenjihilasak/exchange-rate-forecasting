from cmath import exp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simulate_paths_and_plot(
    res, train_df, test_df, steps, n_sims=1000, n_show=30, 
    rate_col='rate', base_seed=42
):
    """
    Simulates multiple trajectories from an ARIMA model and compares them to the test set.
    Each simulated path uses a distinct random seed = base_seed + i for reproducibility.
    
    Parameters
    ----------
    res : fitted statsmodels ARIMA/ARMA model. The model used for simulation.
    train_df : pd.DataFrame. Training data with datetime index and exchange rate column.
    test_df : pd.DataFrame. Test data with datetime index and exchange rate column.
    steps : int. Number of forecast steps.
    n_sims : int, default=1000. Number of simulated paths.
    n_show : int, default=30. Number of paths to plot.
    rate_col : str, default='rate'. Column name for the exchange rate.
    base_seed : int, default=42. Base value for deterministic seeds (path i uses seed = base_seed + i).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Forecast dates starting from last training date
    last_train_date = train_df.index[-1]
    forecast_dates = pd.date_range(
        start=last_train_date,
        periods=steps + 1,
        freq='D'
    )

    all_paths = pd.DataFrame(index=forecast_dates)
    last_log_value = np.log(train_df[rate_col].iloc[-1])
    np.random.seed(base_seed)  # control all aleatory sequences
    
    for i in range(n_sims):
        # Deterministic seed for each path
        # np.random.seed(base_seed + i)

        # Simulate in log scale
        sim_log = res.simulate(nsimulations=steps, anchor='end')

        # Back-transform to rate
        sim_rate = np.concatenate([[np.exp(last_log_value)], np.exp(sim_log)])
        all_paths[f'path_{i+1}'] = pd.Series(sim_rate, index=forecast_dates)

    # ---------------- Plotting ----------------
    plt.figure(figsize=(14, 6))

    # Last 30 days of training data
    plt.plot(train_df.index[-30:], train_df[rate_col].iloc[-30:], 
             color='blue', label='Train (interpolated rate)', linewidth=2)

    # Test period
    mask = (test_df.index >= forecast_dates[0]) & (test_df.index <= forecast_dates[-1])
    test_period = test_df[mask]
    plt.plot(test_period.index, test_period['rate'], 
             color='orange', label='Test (forecast period)', linewidth=2)

    # Forecast start marker
    plt.axvline(x=last_train_date, color='gray', linestyle=':', alpha=0.5, label='Forecast start')

    # Simulated paths
    for idx, col in enumerate(all_paths.columns[:n_show]):
        if idx == 0:
            plt.plot(all_paths[col], color='gray', alpha=0.5, label='Simulated paths')
        else:
            plt.plot(all_paths[col], color='gray', alpha=0.5)

    plt.title('Last 30 days + Simulated Paths (ARIMA)')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

    return all_paths


from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_simulations(all_paths, test_df, rate_col='rate', steps=None, verbose=True):
    """
    Evalúa RMSE y MAE para cada trayectoria simulada y calcula sus medias.

    Parameters:
    - all_paths: DataFrame con trayectorias simuladas (de simulate_paths_and_plot)
    - test_df: DataFrame con los datos reales (contiene rate_col)
    - rate_col: columna con valores reales
    - steps: número de pasos a evaluar (opcional)
    - verbose: si True imprime métricas globales

    Returns:
    - metrics: dict con medias y listas de RMSE y MAE
    """
    # Drop rows in test_df where rate is NaN
    df_eval = test_df[[rate_col]].dropna().copy()

    # Find overlapping dates between test_df and forecast paths
    eval_dates = df_eval.index.intersection(all_paths.index)

    if len(eval_dates) == 0:
        print("⚠️ No hay fechas en común entre predicciones y datos de prueba")
        return {
            'mean_rmse': np.nan,
            'mean_mae': np.nan,
            'rmse_list': [],
            'mae_list': []
        }

    # Extract true values only for valid dates
    y_true = df_eval.loc[eval_dates, rate_col].values

    # Evaluate RMSE and MAE per path <--
    rmses, maes = [], []
    
    for col in all_paths.columns:
        y_pred = all_paths.loc[eval_dates, col].values
            
        rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        maes.append(mean_absolute_error(y_true, y_pred))

    metrics = {
        'expected_rmse': np.mean(rmses) if rmses else np.nan,
        'expected_mae': np.mean(maes) if maes else np.nan,
        'rmse_list': rmses,
        'mae_list': maes
    }

    if verbose:
        print(f"Evaluated dates: {len(eval_dates)}")
        print(f"Evaluated paths: {len(rmses)} of {len(all_paths.columns)}")
        print(f"✅ Expected RMSE: {metrics['expected_rmse']:.6f}")
        print(f"✅ Expected MAE: {metrics['expected_mae']:.6f}")

    return metrics
