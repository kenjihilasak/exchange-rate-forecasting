def evaluate_rmse_forecast(df_forecast, df_real, col_forecast='rate_pred', col_actual='rate'):
    """
    Compute RMSE between forecasted and actual exchange rates (or log_rates).

    Parameters:
    - df_forecast: DataFrame with predicted values (indexed by date)
    - df_real: DataFrame with actual values (indexed by date)
    - col_forecast: column name in df_forecast (e.g. 'rate_pred' or 'log_rate_pred')
    - col_actual: column name in df_real (e.g. 'rate' or 'log_rate')

    Returns:
    - rmse: root mean square error
    """
    # Align both dataframes on index (date)
    df_eval = df_forecast[[col_forecast]].join(df_real[[col_actual]], how='inner')

    # Extract arrays
    y_pred = df_eval[col_forecast].values
    y_true = df_eval[col_actual].values

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"✅ RMSE between simulated forecast and real data: {rmse:.6f}")
    
    return rmse

def evaluate_rmse_horizons(df_forecast, df_real,
                           col_forecast='rate_pred', col_actual='rate'):
    """
    Compute RMSE over 6 months, 12 months, and full forecast period.
    """
    # Align the dataframes
    df = df_forecast[[col_forecast]].join(df_real[[col_actual]], how='inner')
    df = df.sort_index()
    
    # Define horizon lengths (in days)
    horizons = {
        '6_months':  6 * 30,   # ≈180 days
        '12_months': 12 * 30,  # ≈360 days
        'all':       len(df)
    }
    results = {}

    for name, days in horizons.items():
        sub = df.iloc[:days]
        if sub.empty:
            results[name] = np.nan
            continue

        rmse_val = np.sqrt(mean_squared_error(sub[col_actual], sub[col_forecast]))
        results[name] = rmse_val
        print(f"✅ RMSE @ {name.replace('_', ' ')} ({days} days): {rmse_val:.6f}")

    return results