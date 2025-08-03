from cmath import exp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simulate_paths_and_plot(res, train_df, test_df, steps, n_sims=1000, n_show=30, rate_col='rate'):
    """
    Simula múltiples trayectorias desde un modelo ARIMA y las compara con el test set.
    """
    # Get correct forecast dates starting from last training date
    last_train_date = train_df.index[-1]
    forecast_dates = pd.date_range(
        start=last_train_date,
        periods=steps+1, # +1 to include the last date
        freq='D'
    )

    all_paths = pd.DataFrame(index=forecast_dates)
    last_log_value = np.log(train_df[rate_col].iloc[-1])
    
    for i in range(n_sims):
        # Simulación en log
        sim_log = res.simulate(nsimulations=steps, anchor='end')
        # sim_log = np.clip(sim_log, -5, 5)
        
        # Back-transform      
        sim_rate = np.concatenate([[np.exp(last_log_value)], np.exp(sim_log)])
        all_paths[f'path_{i+1}'] = pd.Series(sim_rate, index=forecast_dates)

    # Plotting
    plt.figure(figsize=(14,6))
    
    # Plot training data (last 30 few points)
    plt.plot(train_df.index[-30:], train_df[rate_col].iloc[-30:], 
             color='blue', label='Train (interpolated rate)', linewidth=2)
        
    # Plot the test data only for forecast period
    mask = (test_df.index >= forecast_dates[0]) & (test_df.index <= forecast_dates[-1])
    test_period = test_df[mask]
    plt.plot(test_period.index, test_period['rate'], 
            color='orange', label='Test (forecast period)', 
            linewidth=2)

    # Add vertical line at forecast start
    plt.axvline(x=last_train_date, 
                color='gray', linestyle=':', alpha=0.5,
                label='Forecast start')
            
    # Plot simulated paths
    for idx, col in enumerate(all_paths.columns[:n_show]):
        if idx == 0:
            plt.plot(all_paths[col], color='gray', alpha=0.5, 
                     label='Simulated paths')
        else:
            plt.plot(all_paths[col], color='gray', alpha=0.5)

    plt.title(f'Last 30 days + Simulated Paths (ARIMA)')
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
        'mean_rmse': np.mean(rmses) if rmses else np.nan,
        'mean_mae': np.mean(maes) if maes else np.nan,
        'rmse_list': rmses,
        'mae_list': maes
    }

    if verbose:
        print(f"Evaluated dates: {len(eval_dates)}")
        print(f"Evaluated paths: {len(rmses)} of {len(all_paths.columns)}")
        print(f"✅ Mean RMSE: {metrics['mean_rmse']:.6f}")
        print(f"✅ Mean MAE: {metrics['mean_mae']:.6f}")

    return metrics
