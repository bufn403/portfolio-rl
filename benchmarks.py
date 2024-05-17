import numpy as np

def sharpe_ratio(returns, rf=None, annualized=False) -> np.float64:
    """
    Computes the Sharpe Ratio considering risk free rate.
    """
    scale_factor = np.sqrt(252) if annualized else 1
    np_returns = np.array(returns)
    try:
        if rf is not None:
            return scale_factor * ((np.nanmean(np_returns - rf)) / np.nanstd(np_returns))
        else:
            return scale_factor * (np.nanmean(np_returns) / np.nanstd(np_returns))
    except ZeroDivisionError:
        return np.nan

def sortino_ratio(returns, target_returns, rf=None, annualized=False) -> np.float64:
    """
    Computes the Sortino Ratio
    """
    scale_factor = np.sqrt(252) if annualized else 1
    np_returns = np.array(returns)
    if rf is not None:
        mean = np.nanmean(np_returns - rf)
    else:
        mean = np.nanmean(np_returns)
    mask = (np_returns > 0)
    if not target_returns:
        target_returns = 0
    if mask.all():
        return np.nan
    np_returns[mask] = np.nan
    try:
        return scale_factor * (mean / np.nanstd(np_returns))
    except ZeroDivisionError:
        return np.nan

def max_drawdown(returns) -> np.float64:
    """
    Evaluates the max drawdown over the entire period of returns.
    """
    arr = np.array(returns)
    #  Handle case where returns length is 0 or 1
    if not arr.shape or arr.shape[0] == 0:
        return 0.0
    elif arr.shape[0] == 1:
        return min(0.0, arr[0])

    cumulative_returns = np.nancumsum(arr)

    # index of trough of the maximum difference (peak to trough)
    drawdown_end = np.nanargmax(np.maximum.accumulate(cumulative_returns) - cumulative_returns)

    # No drawdown
    if drawdown_end == 0:
        return 0.0

    # Bug with numpy where argmax from drawdown_end returns corresponding index
    # as if object was not grouped, but then uses 1-n indexes for slicing which
    # causes drawdowns to be reversed in some cases.
    drawdown_end %= cumulative_returns.shape[0]

    # Index of peak (max value up to trough)
    drawdown_start = np.nanargmax(cumulative_returns[:drawdown_end])
    # Edge case for when drawdown starts at beginning of array
    if drawdown_start != 0 or (drawdown_start == 0 and arr[0] >= 0):
        drawdown_start += 1

    # Computes percentage change as decimal value
    change = np.nanprod(arr[drawdown_start:drawdown_end + 1] + 1) - 1
    change *= -1  # Make the change positive
    return change