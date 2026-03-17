import numpy as np

def remove_nan_values(col):
    """Remove NaN values from a column."""
    return col[~np.isnan(col)]


def calculate_mean(values):
    """Calculate mean from a list of values."""
    if len(values) == 0:
        return np.nan
    return sum(values) / len(values)


def calculate_std(values):
    """Calculate standard deviation from a list of values."""
    if len(values) == 0:
        return np.nan
    m = calculate_mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return variance ** 0.5


def calculate_percentile(sorted_values, percent):
    """Calculate percentile from sorted values."""
    if len(sorted_values) == 0:
        return np.nan
    k = (len(sorted_values) - 1) * percent / 100
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def calculate_column_stats(col):
    """Calculate statistics for a single column."""
    col_no_nan = remove_nan_values(col)
    n = len(col_no_nan)
    total = len(col)
    missing = total - n
    unique = len(set(col_no_nan)) if n > 0 else 0

    stats = {
        'count': n,
        'missing': missing,
        'unique': unique,
        'min': min(col_no_nan) if n > 0 else np.nan,
        'max': max(col_no_nan) if n > 0 else np.nan,
        'mean': calculate_mean(col_no_nan),
        'std': calculate_std(col_no_nan),
    }

    if n > 0:
        sorted_col = sorted(col_no_nan)
        stats['p25'] = calculate_percentile(sorted_col, 25)
        stats['p50'] = calculate_percentile(sorted_col, 50)
        stats['p75'] = calculate_percentile(sorted_col, 75)
    else:
        stats['p25'] = np.nan
        stats['p50'] = np.nan
        stats['p75'] = np.nan

    return stats
