import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 🔥 1. Calculate ODR
def calculate_odr(df: pd.DataFrame):
    """Calculate ODR (Overall Defaulter Rate) for each record."""
    df['ODR'] = df['Bad_Count'] / df['Total_Count']
    return df

# 🔥 2. Calculate Weighted ODR
def calculate_weighted_odr(df: pd.DataFrame, weight_column='Total_Count'):
    """Calculate Weighted ODR for overall comparison."""
    total_tc = df[weight_column].sum()
    df['Weighted_ODR'] = (df['Bad_Count'] / df['Total_Count']) * (df[weight_column] / total_tc)
    weighted_odr = df['Weighted_ODR'].sum()
    return weighted_odr

# 🔥 3. Calculate ODR Trends
def calculate_trends(df: pd.DataFrame, period_column='Period', group_by_columns=None):
    """
    Analyze ODR trends over time.
    - group_by_columns: Optional columns to segment the trends (e.g., Region, Pool)
    """
    if group_by_columns:
        group_by = group_by_columns + [period_column]
    else:
        group_by = [period_column]
        
    trend_df = df.groupby(group_by).agg({'Bad_Count': 'sum', 'Total_Count': 'sum'}).reset_index()
    trend_df['ODR'] = trend_df['Bad_Count'] / trend_df['Total_Count']
    return trend_df

# ---------------------------------------------------------
# 🔥 4. Compare ODR by Regions

def compare_regions(df: pd.DataFrame, group_by_columns=['Period']):
    """
    Compare rank-ordering of ODRs across regions.
    - group_by_columns: Columns to group by (default is Period)
    """
    # Ensure 'Region' is string and handle missing values
    df['Region'] = df['Region'].fillna('Unknown').astype(str)
    
    # Calculate ODR while handling potential NaN issues
    comparison_df = (
        df.groupby(['Region'] + group_by_columns)
          .agg({'Bad_Count': 'sum', 'Total_Count': 'sum'})
          .reset_index()
    )
    
    # Avoid division by zero and calculate ODR
    comparison_df['ODR'] = comparison_df['Bad_Count'] / comparison_df['Total_Count'].replace(0, np.nan)
    
    # Pivot the table for comparison
    pivot_df = comparison_df.pivot(index='Region', columns=group_by_columns[0], values='ODR')
    
    # Fill NaN with 0 or another placeholder
    pivot_df = pivot_df.fillna(0)
    
    return pivot_df


# 🔥 5. Calculate ODR Ranks
def calculate_odr_ranks(df: pd.DataFrame, periods_to_compare=None, rank_by='Region'):
    """
    Compare ODR ranks for specified quarters.
    - periods_to_compare: List of periods to compare.
    """
    unique_periods = sorted(df['Period'].unique())
    if periods_to_compare is None:
        periods_to_compare = unique_periods[-2:]

    filtered_df = df[df['Period'].isin(periods_to_compare)]
    ranked_df = filtered_df.copy()
    ranked_df['ODR'] = ranked_df['Bad_Count'] / ranked_df['Total_Count']
    ranked_df['ODR_Rank'] = ranked_df.groupby('Period')['ODR'].rank(ascending=False, method='min')
    
    rank_pivot = ranked_df.pivot(index=rank_by, columns='Period', values='ODR_Rank')
    rank_pivot['Rank_Change'] = rank_pivot.max(axis=1) - rank_pivot.min(axis=1)
    return rank_pivot

# 🔥 6. Compare Product-Level ODR
def compare_product_level_odr(df: pd.DataFrame, product_column='Product', group_by_columns=['Region', 'Period']):
    """
    Compare product-level ODR trends between regions.
    """
    comparison_df = df.groupby(group_by_columns + [product_column]).agg({'Bad_Count': 'sum', 'Total_Count': 'sum'}).reset_index()
    comparison_df['ODR'] = comparison_df['Bad_Count'] / comparison_df['Total_Count']
    return comparison_df

# ---------------------------------------------------------
# 🔥 7. Top N Regions with Highest ODR
def top_odr_regions(df, top_n=5):
    """Find top N regions with highest ODR."""
    return df.groupby('Region')['ODR'].mean().nlargest(top_n)

# 🔥 8. Bottom N Regions with Lowest ODR
def bottom_odr_regions(df, bottom_n=3):
    """Find bottom N regions with lowest ODR."""
    return df.groupby('Region')['ODR'].mean().nsmallest(bottom_n)

# 🔥 9. Rank Change in Regions
def rank_change_regions(df):
    """Identify regions whose rank-ordering has changed over the last 2 quarters."""
    rank_df = calculate_odr_ranks(df)
    rank_change = rank_df.diff(axis=1).iloc[:, 1:]
    return rank_change[rank_change != 0].dropna(how='all')

# ---------------------------------------------------------
# 🔥 10. Segment Regions
def segment_regions(df, segment_column='ODR', num_segments=10):
    """Refine region segmentation into consolidated groups with common trends."""
    df['Segment'] = pd.qcut(df[segment_column], num_segments, labels=False)
    return df.groupby('Segment').agg({'ODR': 'mean'}).reset_index()

# 🔥 11. Group Pools by Count
def group_pools_by_count(df, count_column='Total_Count', count_threshold=100):
    """Group pools with Total Count below a specified threshold."""
    grouped = df[df[count_column] < count_threshold].groupby('Pool').agg({'Bad_Count': 'sum', 'Total_Count': 'sum'})
    grouped['ODR'] = grouped['Bad_Count'] / grouped['Total_Count']
    return grouped.reset_index()

# ---------------------------------------------------------
# 🔥 12. Sourcing Trends
def sourcing_trend(df, group_by_columns=['Region', 'Period']):
    """Analyze sourcing trends in terms of total count over time."""
    return df.groupby(group_by_columns)['Total_Count'].sum().unstack().fillna(0)

# ---------------------------------------------------------
# 🔥 13. Correlation with Macroeconomic Variables
def correlate_with_macroeconomic(df, macro_df, macro_columns, target_column='ODR'):
    """
    Check correlation between ODR trends and macroeconomic variables.
    - macro_columns: List of macroeconomic variables to correlate.
    """
    combined_df = pd.merge(df, macro_df, on='Period', how='inner')
    corr_df = combined_df[[target_column] + macro_columns].corr()
    return corr_df[target_column].drop(target_column)

# 🔥 14. Predict Future ODR
def predict_future_odr(df, trend_column='ODR', period_column='Period'):
    """Predict ODR trends for the next quarters using rolling average."""
    df[period_column] = pd.to_datetime(df[period_column], format="%b %y")
    df.set_index(period_column, inplace=True)
    return df[trend_column].rolling(window=2).mean().shift(-1)

# ---------------------------------------------------------
# 🔥 15. Regions with No Defaults
def region_with_no_defaults(df, group_by='Region', period_column='Period', lookback_periods=5):
    """Identify regions with zero defaults in the last N periods."""
    zero_default_df = df[df['Bad_Count'] == 0].groupby(group_by)[period_column].nunique()
    return zero_default_df[zero_default_df >= lookback_periods].index.tolist()

# 🔥 16. Highest Contributor Analysis
def highest_contributor(df, entity='Region', metric='Bad_Count', group_by_columns=['Pool', 'Period']):
    """Find the highest contributor for a given metric."""
    agg_df = df.groupby(group_by_columns + [entity])[metric].sum().reset_index()
    return agg_df.loc[agg_df[metric].idxmax()]
