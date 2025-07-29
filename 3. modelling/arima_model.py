import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def forecast_and_plot_arima(res, train_df, test_df, steps=712, rate_col='rate', title='ARIMA Forecast vs Actual Rates'):
    """
    Generates and plots ARIMA forecasts against test data.

    Parameters:
    - res: fitted ARIMA model (statsmodels result object)
    - train_df: DataFrame with training data (must include 'rate')
    - test_df: DataFrame with test data (must include 'rate')
    - steps: number of steps to forecast (default: 712 for 2 years)
    - rate_col: name of the column containing exchange rates in both datasets
    - title: custom plot title
    """
    # Forecast range
    last_train_date = train_df.index.max()
    start_fc = last_train_date + pd.Timedelta(days=1)
    fc_index = pd.date_range(start=start_fc, periods=steps, freq='D')

    # Forecast in log scale
    fc = res.get_forecast(steps=steps)
    fc_log = pd.Series(fc.predicted_mean.values, index=fc_index, name='fc_log_rate')
    
    ci = fc.conf_int()
    ci.index = fc_index
    ci.columns = ['lower_log_rate', 'upper_log_rate']

    # Back-transform to rate
    fc_rate = np.exp(fc_log)
    ci_lower = np.exp(ci['lower_log_rate'])
    ci_upper = np.exp(ci['upper_log_rate'])

    # Align actual test data
    test_rate = test_df[rate_col].loc[start_fc:fc_index[-1]]

    # Plot
    plt.figure(figsize=(14,6))
    plt.plot(train_df[rate_col], label='Train (rate)', color='blue')
    plt.plot(fc_rate, label='Forecast', color='red')
    plt.fill_between(fc_rate.index, ci_lower, ci_upper, color='gray', alpha=0.3)
    plt.plot(test_rate, label='Test (actual)', color='orange')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return fc_rate, test_rate

from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_forecast_metrics(fc_rate, test_df, rate_col='rate', verbose=True):
    """
    Aligns forecast and test data indices and computes RMSE and MAE.

    Parameters:
    - fc_rate: pandas Series with forecasted rates (index = datetime)
    - test_df: DataFrame containing actual rates (must include `rate_col`)
    - rate_col: name of the column containing the actual exchange rates
    - verbose: if True, prints metrics

    Returns:
    - metrics: dict with 'rmse' and 'mae'
    """
    # Drop missing actual values
    test_rate = test_df[[rate_col]].dropna().copy()

    # Align forecast and test data on common dates
    common_index = test_rate.index.intersection(fc_rate.index)
    y_true = test_rate.loc[common_index, rate_col]
    y_pred = fc_rate.loc[common_index]

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    if verbose:
        print(f"Aligned y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        print(f"✅ RMSE: {rmse:.6f}")
        print(f"✅ MAE: {mae:.6f}")

    return {'rmse': rmse, 'mae': mae}
