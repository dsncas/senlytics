import pandas as pd
from datetime import datetime, timedelta, timezone
import requests
import concurrent.futures
import time
import numpy as np
from scipy import stats
import statsmodels.api as sm
import hvplot.pandas
import holoviews as hv
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
# Enable Holoviews extension
hv.extension('bokeh')


def fetch_data_worker(location_id, start_time, end_time, mode="past", token="8190ffce-1690-40fe-800b-dbab7dd4deb0"):
    """
    Fetches air quality data from the AirGradient API and returns it as a Pandas DataFrame.

    Parameters:
        location_id (str): The location ID for which data is to be fetched.
        start_time (str): Start time in ISO 8601 format (e.g., "20241126T090000Z").
        end_time (str): End time in ISO 8601 format (e.g., "20241128T090000Z").
        mode (str): Either 'past' or 'raw' to choose the API endpoint.
        token (str): API authentication token.

    Returns:
        pd.DataFrame or None
    """
    if mode not in ["past", "raw"]:
        raise ValueError("mode must be either 'past' or 'raw'")

    url = f"https://api.airgradient.com/public/api/v1/locations/{location_id}/measures/{mode}"

    params = {
        "token": token,
        "from": start_time,
        "to": end_time
    }

    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers, params=params)

    if response.ok:
        return pd.DataFrame(response.json())
    else:
        return None


def fetch_data(location_id, start_time, end_time, mode="past", token="8190ffce-1690-40fe-800b-dbab7dd4deb0"):
    """
    Fetches air quality data from AirGradient API across longer time ranges via batching.

    Parameters:
        location_id (str)
        start_time (str): Format "YYYYMMDDTHHMMSSZ"
        end_time (str): Format "YYYYMMDDTHHMMSSZ"
        mode (str): 'past' (10-day interval) or 'raw' (2-day interval)
        token (str)

    Returns:
        pd.DataFrame
    """
    start = datetime.strptime(start_time, "%Y%m%dT%H%M%SZ")
    end = datetime.strptime(end_time, "%Y%m%dT%H%M%SZ")

    # Interval depends on mode
    interval = timedelta(days=10 if mode == "past" else 2)

    all_chunks = []
    current_start = start
    while start < end:
        chunk_end = min(start + interval, end)
        df = fetch_data_worker(location_id, start.strftime("%Y%m%dT%H%M%SZ"), chunk_end.strftime("%Y%m%dT%H%M%SZ"), mode, token)
        if df is not None and not df.empty:
            df = df.dropna(axis=1, how='all')
            all_chunks.append(df)
        start = chunk_end

    if not all_chunks:
        return pd.DataFrame()

    common_cols = set.intersection(*[set(df.columns) for df in all_chunks])
    aligned_chunks = [df[list(common_cols)] for df in all_chunks]
    return pd.concat(aligned_chunks, ignore_index=True)


def fetch_data_parallel(location_ids, start_time, end_time, mode="past", token="8190ffce-1690-40fe-800b-dbab7dd4deb0"):
    """
    Fetches AirGradient data for multiple location IDs in parallel, with progress bar.

    Parameters:
        location_ids (list): List of location IDs
        start_time (str): "YYYYMMDDTHHMMSSZ"
        end_time (str): "YYYYMMDDTHHMMSSZ"
        mode (str): 'past' or 'raw'
        token (str): API key

    Returns:
        pd.DataFrame: Combined and formatted
    """
    def fetch_for_one(location_id):
        return fetch_data(location_id, start_time, end_time, mode, token)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        all_results = list(tqdm(executor.map(fetch_for_one, location_ids), total=len(location_ids), desc="Fetching"))

    valid_results = [df.dropna(axis=1, how='all') for df in all_results if df is not None and not df.empty]

    if not valid_results:
        print("No data fetched from any location.")
        return pd.DataFrame()

    common_cols = set.intersection(*[set(df.columns) for df in valid_results])
    merged_df = pd.concat([df[list(common_cols)] for df in valid_results], ignore_index=True)

    # Standard formatting
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    merged_df['Local Date/Time'] = merged_df['timestamp'].dt.tz_convert('Europe/London').dt.strftime('%Y-%m-%d %H:%M:%S')
    merged_df['UTC Date/Time'] = merged_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    # Rename columns for clarity
    merged_df.rename(columns={
        'locationId': 'Location ID',
        'locationName': 'Location Name',
        'pm01': 'PM1 (μg/m³)',
        'pm02': 'PM2.5 (μg/m³)',
        'pm10': 'PM10 (μg/m³)',
        'pm003Count': '0.3um particle count',
        'atmp': 'Temperature (°C)',
        'rhum': 'Humidity (%)',
        'rco2': 'CO2 (ppm)',
        'tvoc': 'TVOC avg (ppb)',
        'tvocIndex': 'TVOC index',
        'noxIndex': 'NOX index'
    }, inplace=True)

    merged_df.drop(columns=["timestamp"], inplace=True, errors='ignore')

    return merged_df


def update_data(
    combined_data='alldata_combined.csv',
    combined_data_5min='alldata_combined_5min.csv',
    location_id='LocationID.csv',
    mode='past',
    token="8190ffce-1690-40fe-800b-dbab7dd4deb0"
):
    """
    Updates data files with new measurements and processes it into summarized forms.

    Parameters:
        combined_data (str): Path to the main combined dataset file (default: 'alldata_combined.csv').
        combined_data_5min (str): Path to the 5-minute aggregated dataset file (default: 'alldata_combined_5min.csv').
        location_id (str): Path to the file containing location IDs (default: 'LocationID.csv').
        mode (str): 'past' (10-day interval) or 'raw' (2-day interval)
        token (str): API token used for fetching new data.

    Returns:
        tuple: Updated main combined dataset, the processed 5-minute summary, and paths to saved files.
    """
    # Load location ID file and extract unique location IDs and names
    id_metadata = pd.read_csv(location_id, low_memory=False)
    id_lookup = id_metadata[['Location ID', 'Location']].drop_duplicates()
    location_ids = list(map(str, id_lookup['Location ID']))

    # Load existing main dataset
    df = pd.read_csv(combined_data, parse_dates=['Local Date/Time'], low_memory=False)

    # Determine start time using the latest UTC timestamp in the dataset
    start_time = datetime.strptime(df['UTC Date/Time'].max(), "%Y-%m-%dT%H:%M:%S.000Z").strftime("%Y%m%dT%H%M%SZ")

    # Set end time as the current UTC time
    now = datetime.now(timezone.utc)
    end_time = now.strftime("%Y%m%dT%H%M%SZ")

    # Fetch new data from API
    big_df = fetch_data_parallel(location_ids, start_time, end_time, mode=mode, token=token)

    # If no new data, skip update and return existing datasets
    if big_df.empty:
        print("No new data fetched. Skipping update.")
        df['Local Date/Time'] = pd.to_datetime(df['Local Date/Time'])
        result = pd.read_csv(combined_data_5min, parse_dates=['Local Date/Time'], low_memory=False)

        summary_pivot = result.pivot_table(index='Local Date/Time', columns='level_1', aggfunc='mean')
        summary_pivot.columns = summary_pivot.columns.swaplevel(0, 1)
        summary_pivot = summary_pivot.sort_index(axis=1)
        summary_pivot = summary_pivot.rename_axis(columns={'level_1': 'Params'})

        return df, summary_pivot, combined_data, combined_data_5min

    # Preprocess new data
    big_df['Local Date/Time'] = pd.to_datetime(big_df['Local Date/Time'])

    big_df = correction(big_df)

    # Merge existing and new data, remove duplicates
    full_data_updated = pd.concat([df, big_df], axis=0).drop_duplicates(subset=['Local Date/Time', 'Location Name'])

    # Recalculate PM1_2.5 and PM2.5_10 for the merged dataset
    full_data_updated['PM1_2.5'] = full_data_updated['PM2.5 (μg/m³)'] - full_data_updated['PM1 (μg/m³)']
    full_data_updated['PM2.5_10'] = full_data_updated['PM10 (μg/m³)'] - full_data_updated['PM2.5 (μg/m³)']

    # Save the updated main dataset
    full_data_updated.to_csv(combined_data, encoding='utf-8', index=False)

    # Load existing 5-minute dataset
    result = pd.read_csv(combined_data_5min, parse_dates=['Local Date/Time'], low_memory=False)

    # Extract numeric fields and relevant metadata from new data
    numeric_data = big_df.select_dtypes(include=['number']).copy()
    numeric_data['Local Date/Time'] = big_df['Local Date/Time']
    numeric_data['Location Name'] = big_df['Location Name']

    # Create pivot table grouped by time and location
    resampled_new_data = numeric_data.pivot_table(
        index='Local Date/Time',
        columns='Location Name',
        aggfunc='mean'
    ).sort_index()

    # Resample to 5-minute intervals and reshape the table
    resampled_new_data = resampled_new_data.resample('5min').mean()
    resampled_new_data = resampled_new_data.stack(level=0, future_stack=True).reset_index()

    # Merge with existing 5-minute dataset and remove duplicates
    result = pd.concat([result, resampled_new_data]).drop_duplicates()

    # Save the updated 5-minute dataset
    result.to_csv(combined_data_5min, encoding='utf-8', index=False)

    # Build final pivot table for structured analysis
    summary_pivot = result.pivot_table(index='Local Date/Time', columns='level_1', aggfunc='mean')
    summary_pivot.columns = summary_pivot.columns.swaplevel(0, 1)
    summary_pivot = summary_pivot.sort_index(axis=1)
    summary_pivot = summary_pivot.rename_axis(columns={'level_1': 'Params'})

    return full_data_updated, summary_pivot


## number of valid readings
def n(x, mod, obs):
    """
    Calculates the number of valid readings.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        int: Number of valid readings.
    """
    x = x[[mod, obs]].dropna()
    res = x.shape[0]
    return res


## fraction within a factor of two
def FAC2(x, mod, obs):
    """
    Calculates the fraction of values within a factor of two.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Fraction of values within a factor of two.
    """
    x = x[[mod, obs]].dropna()
    ratio = x[mod] / x[obs]
    ratio = ratio.dropna()
    len = ratio.shape[0]
    if len > 0:
        res = ratio[(ratio >= 0.5) & (ratio <= 2)].shape[0] / len
    else:
        res = np.nan
    return res


## mean bias
def MB(x, mod, obs):
    """
    Calculates the mean bias.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Mean bias.
    """
    x = x[[mod, obs]].dropna()
    res = np.mean(x[mod] - x[obs])
    return res


## mean gross error
def MGE(x, mod, obs):
    """
    Calculates the mean gross error.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Mean gross error.
    """
    x = x[[mod, obs]].dropna()
    res = np.mean(np.abs(x[mod] - x[obs]))
    return res


## normalised mean bias
def NMB(x, mod, obs):
    """
    Calculates the normalised mean bias.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Normalised mean bias.
    """
    x = x[[mod, obs]].dropna()
    res = np.sum(x[mod] - x[obs]) / np.sum(x[obs])
    return res


## normalised mean gross error
def NMGE(x, mod, obs):
    """
    Calculates the normalised mean gross error.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Normalised mean gross error.
    """
    x = x[[mod, obs]].dropna()
    res = np.sum(np.abs(x[mod] - x[obs])) / np.sum(x[obs])
    return res


## root mean square error
def RMSE(x, mod, obs):
    """
    Calculates the root mean square error.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Root mean square error.
    """
    x = x[[mod, obs]].dropna()
    res = np.sqrt(np.mean((x[mod] - x[obs]) ** 2))
    return res


## correlation coefficient
def r(x, mod, obs):
    """
    Calculates the correlation coefficient.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        tuple: Correlation coefficient and its p-value.
    """
    x = x[[mod, obs]].dropna()
    x_mod = x[mod].squeeze()
    x_obs = x[obs].squeeze()
    #res = stats.pearsonr(x[mod], x[obs])
    res = stats.pearsonr(x_mod, x_obs)
    return res


## Coefficient of Efficiency
def COE(x, mod, obs):
    """
    Calculates the Coefficient of Efficiency.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Coefficient of Efficiency.
    """
    x = x[[mod, obs]].dropna()
    res = 1 - np.sum(np.abs(x[mod] - x[obs])) / np.sum(np.abs(x[obs] - np.mean(x[obs])))
    return res


## Index of Agreement
def IOA(x, mod, obs):
    """
    Calculates the Index of Agreement.

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Index of Agreement.
    """
    x = x[[mod, obs]].dropna()
    LHS = np.sum(np.abs(x[mod] - x[obs]))
    RHS = 2 * np.sum(np.abs(x[obs] - np.mean(x[obs])))
    if LHS <= RHS:
        res = 1 - LHS / RHS
    else:
        res = RHS / LHS - 1
    return res


#determination of coefficient
def R2(x, mod, obs):
    """
    Calculates the determination coefficient (R-squared).

    Parameters:
        x (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.

    Returns:
        float: Determination coefficient (R-squared).
    """
    x = x[[mod, obs]].dropna()
    X = sm.add_constant(x[obs])
    y=x[mod]
    model = sm.OLS(y, X).fit()
    res = model.rsquared
    return res

def Stats(df, mod, obs,
             statistic = None):
    """
    Calculates specified statistics based on provided data.

    Parameters:
        df (pandas.DataFrame):: Input DataFrame containing the dataset.
        mod (str): Column name of the model predictions.
        obs (str): Column name of the observed values.
        statistic (list): List of statistics to calculate.

    Returns:
        DataFrame: DataFrame containing calculated statistics.
    """

    if statistic is None:
        statistic = ["n", "FAC2", "MB", "MGE", "NMB", "NMGE", "RMSE", "r", "COE", "IOA","R2"]
    res = {}
    if "n" in statistic:
        res["n"] = n(df, mod, obs)
    if "FAC2" in statistic:
        res["FAC2"] = FAC2(df, mod, obs)
    if "MB" in statistic:
        res["MB"] = MB(df, mod, obs)
    if "MGE" in statistic:
        res["MGE"] = MGE(df, mod, obs)
    if "NMB" in statistic:
        res["NMB"] = NMB(df, mod, obs)
    if "NMGE" in statistic:
        res["NMGE"] = NMGE(df, mod, obs)
    if "RMSE" in statistic:
        res["RMSE"] = RMSE(df, mod, obs)
    if "r" in statistic:
        res["r"] = r(df, mod, obs)[0]
        p_value = r(df, mod, obs)[1]
        if p_value >= 0.1:
            res["p_level"] = ""
        elif p_value < 0.1 and p_value >= 0.05:
            res["p_level"] = "+"
        elif p_value < 0.05 and p_value >= 0.01:
            res["p_level"] = "*"
        elif p_value < 0.01 and p_value >= 0.001:
            res["p_level"] = "**"
        else:
            res["p_level"] = "***"
    if "COE" in statistic:
        res["COE"] = COE(df, mod, obs)
    if "IOA" in statistic:
        res["IOA"] = IOA(df, mod, obs)
    if "R2" in statistic:
        res["R2"] = R2(df, mod, obs)

    results = {'n':res['n'], 'FAC2':res['FAC2'], 'MB':res['MB'], 'MGE':res['MGE'], 'NMB':res['NMB'],
               'NMGE':res['NMGE'],'RMSE':res['RMSE'], 'r':res['r'],'p_level':res['p_level'],
               'COE':res['COE'], 'IOA':res['IOA'], 'R2':res['R2']}

    results = pd.DataFrame([results])

    return results


def ts_plots(data, param, time_resolution='1h', plot_kws=None):
    """
    Create vertically stacked line plots for all columns in the specified parameter.

    Parameters:
        data (pd.DataFrame): Input data.
        param (str): The key to access the data columns (e.g., 'pm02').
        time_resolution (str): Resampling time resolution (e.g., '1h', '1d').
        plot_kws (dict): Plot configuration parameters (e.g., width, height, title font size).

    Returns:
        hv.Layout: A vertically stacked layout of plots.
    """
    # Define default plot parameters
    default_plot_kws = {
        'width': 1000,
        'height': 80,
        'title_fontsize': '6pt'
    }

    # Update default parameters with user-specified values
    if plot_kws:
        default_plot_kws.update(plot_kws)

    # Extract plot parameters
    width = default_plot_kws['width']
    height = default_plot_kws['height']
    title_fontsize = default_plot_kws['title_fontsize']

    # Filter columns by name
    cols = data[param].columns

    # Resample data
    data = data.resample(time_resolution).mean()

    # Function to generate a single plot
    def generate_plot(column):
        return data[param][column].hvplot.line(
            title=column,  # Use column name as title
            width=width,
            height=height,
        ).opts(
            yticks=2,                      # Only show one major tick on y-axis
            yaxis='bare',                  # Disable minor ticks
            axiswise=True,                 # Ensure independent y-axes per plot
            xlabel='',                     # Remove x-axis label
            fontsize={'title': title_fontsize}  # Set title font size
        )

    # Generate plots for all columns and stack them vertically
    plots = [generate_plot(col) for col in cols]

    # Combine all plots into a single layout
    grid_plot = hv.Layout(plots).cols(1)  # Stack all plots vertically

    # Display the grid plot
    return grid_plot.opts(
        width=width,             # Overall width of the layout
        height=len(cols) * height  # Overall height of the layout
    )


def ts_plots_matplotlib(data, param, time_resolution='1h', plot_kws=None, save_path=None, dpi=600):
    """
    Create vertically stacked line plots for all columns in the specified parameter using matplotlib.

    Parameters:
        data (pd.DataFrame): Input data.
        param (str): The key to access the data columns (e.g., 'pm02').
        time_resolution (str): Resampling time resolution (e.g., '1h', '1d').
        plot_kws (dict): Plot configuration parameters (e.g., figsize, title font size).
        save_path (str): File path to save the plot. If None, the plot is not saved.
        dpi (int): DPI resolution for saving the image.

    Returns:
        Saves the generated plot and displays it.
    """
    # Default plot settings
    default_plot_kws = {
        'figsize': (7, 0.5),  # Figure size per subplot
        'title_fontsize': 6,  # Title font size for subplots
        'xlabel_fontsize': 7,  # X-axis label font size
        'ylabel_fontsize': 7   # Y-axis label font size
    }
    plot_kws = plot_kws or {}
    plot_kws = {**default_plot_kws, **plot_kws}  # Merge default and user settings

    # Extract settings
    figsize = plot_kws['figsize']
    title_fontsize = plot_kws['title_fontsize']
    xlabel_fontsize = plot_kws['xlabel_fontsize']
    ylabel_fontsize = plot_kws['ylabel_fontsize']

    # Ensure parameter exists in data
    if param not in data:
        raise ValueError(f"Parameter '{param}' not found in the data.")

    # Columns to process
    cols = data[param].columns if hasattr(data[param], 'columns') else [data[param].name]

    # Resample data
    data = data.resample(time_resolution).mean()

    # Create figure
    num_plots = len(cols)
    fig, axes = plt.subplots(num_plots, 1, figsize=(figsize[0], figsize[1] * num_plots), sharex=True)

    # Ensure axes is iterable for a single subplot case
    if num_plots == 1:
        axes = [axes]

    # Generate plots
    for ax, col in zip(axes, cols):
        ax.plot(data.index, data[param][col], label=col, color='blue', linewidth=0.8)
        ax.set_title(col, fontsize=title_fontsize)
        ax.set_ylabel(param, fontsize=ylabel_fontsize)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Configure shared x-axis
    axes[-1].set_xlabel('Time', fontsize=xlabel_fontsize)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=8)

    # Adjust layout
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path} with DPI={dpi}.")

    # Display the plot
    plt.show()


def diurnal_week_plot(df, location_name, param_list=None, std=True, save_path='./', dpi=600, figsize=None):
    """
    Generates and saves a diurnal weekly cycle plot for specified air quality and environmental parameters.

    Parameters:
        df (DataFrame): A Pandas DataFrame with multi-indexed columns, including 'Params' as a level.
        location_name (str or list): Single location name or a list of locations to be plotted.
        param_list (list, optional): List of parameters to be plotted. If None, defaults to common parameters.
        std (bool, optional): Whether to display standard deviation shaded regions. Default is True.
        save_path (str, optional): Directory path to save the plot (default is './').
        dpi (int, optional): Dots per inch (DPI) for the saved figure (default is 600).
        figsize (tuple, optional): Figure size in inches (width, height). Defaults to (7, num_params-1).

    Returns:
        Saves the generated plot and displays it.
    """

    # Convert single location to a list
    if isinstance(location_name, str):
        location_name = [location_name]

    # Convert multi-indexed DataFrame to long format
    dfa = df.stack(level='Params', future_stack=True).reset_index()
    dfa['Hour'] = dfa['Local Date/Time'].dt.hour
    dfa['Day_of_week'] = dfa['Local Date/Time'].dt.day_name()

    # Default parameter list
    if param_list is None:
        param_list = ['PM1 (μg/m³)', 'PM2.5 (μg/m³)', 'PM10 (μg/m³)', 'CO2 (ppm)',
                      'TVOC avg (ppb)', 'NOX index', 'Temperature (°C)', 'Humidity (%)']

    # Pivot table with mean and standard deviation
    dfb = dfa.pivot_table(index='Hour', columns=['Params', 'Day_of_week'], aggfunc=["mean", "std"])
    dfb.columns.names = ['Stat', 'Location', 'Params', 'Day_of_week']
    dfb_swapped = dfb.swaplevel('Location', 'Stat', axis=1)

    # Mapping parameter names to formatted y-axis labels
    param_labels = {
        'PM1 (μg/m³)': 'PM$_1$\n($\mu$g m$^{-3}$)',
        'PM2.5 (μg/m³)': 'PM$_{2.5}$\n($\mu$g m$^{-3}$)',
        'PM10 (μg/m³)': 'PM$_{10}$\n($\mu$g m$^{-3}$)',
        'PM2.5_10': 'PM$_{2.5-10}$\n($\mu$g m$^{-3}$)',
        'CO2 (ppm)': 'CO$_2$\n(ppm)',
        'TVOC avg (ppb)': 'TVOC\n(ppb)',
        'NOX index': 'NO$_x$',
        'Temperature (°C)': 'T\n(°C)',
        'Humidity (%)': 'RH\n(%)'
    }

    # Define figure size dynamically if not provided
    if figsize is None:
        figsize = (7, max(len(param_list), 1))

    # Create subplots: rows = parameters, columns = days of the week
    fig, axes = plt.subplots(nrows=len(param_list), ncols=7, figsize=figsize, sharex=True, sharey='row')

    # Ensure axes is always a 2D array for consistent indexing
    if len(param_list) == 1:
        axes = np.expand_dims(axes, axis=0)

    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    # Define day order
    dayofweek_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Get colors for different locations
    colors = plt.cm.get_cmap('tab10', len(location_name))

    # Store legend handles for later use
    legend_handles = []

    # Loop through each parameter and each day
    for i, poll in enumerate(param_list):
        for j, day in enumerate(dayofweek_list):
            for k, location in enumerate(location_name):
                if (location, 'mean', poll, day) in dfb_swapped.columns:
                    # Plot the mean line
                    line, = axes[i, j].plot(dfb_swapped.index, dfb_swapped[location]['mean'][poll][day],
                                            linestyle='-', color=colors(k), label=location)
                    if i == 0 and j == 3:
                        legend_handles.append(Line2D([0], [0], color=colors(k), linestyle='-', label=location))

                    # Add shaded region for standard deviation if enabled
                    if std and (location, 'std', poll, day) in dfb_swapped.columns:
                        axes[i, j].fill_between(
                            dfb_swapped.index,
                            dfb_swapped[location]['mean'][poll][day] - dfb_swapped[location]['std'][poll][day],
                            dfb_swapped[location]['mean'][poll][day] + dfb_swapped[location]['std'][poll][day],
                            alpha=0.2,
                            color=colors(k)
                        )

            axes[i, j].set_xlim(0, 23)
            axes[i, j].set_xticks([0, 6, 12, 18])
            axes[i, j].set_xticklabels(['00:00', '06:00', '12:00', '18:00'], rotation=90)
            if j == 0:
                axes[i, 0].set_ylabel(param_labels.get(poll, poll), fontsize=7)
            axes[i, j].tick_params(axis='both', which='both', labelsize=7)
            if i == len(param_list) - 1:
                axes[i, j].set_xlabel(day, fontsize=7)
            if i == 0:
                axes[0, j].set_title(day, fontsize=7)

    # Legend placement
    max_cols = 3
    ncol = min(len(location_name), max_cols)
    nrows = np.ceil(len(location_name) / max_cols)
    y_anchor = 1.3
    height = 0.4 + (nrows - 1) * 0.25

    axes[0, 0].legend(handles=legend_handles,
                      bbox_to_anchor=(0.02, y_anchor, 7.6, height),
                      ncol=ncol,
                      fontsize=7,
                      frameon=False)

    # Save plot
    if save_path:
        suffix = '_nostd' if not std else ''
        save_filename = save_path + '_'.join(location_name) + suffix + '.png'
        plt.savefig(save_filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_filename} with DPI={dpi}.")

    # Show plot
    plt.show()


def diurnal_weekend_plot(df, location_name, param_list=None, std=True, save_path='./', dpi=600, figsize=None):
    """
    Generates and saves a diurnal cycle plot comparing weekdays and weekends for specified air quality parameters.

    Parameters:
        df (DataFrame): A Pandas DataFrame with multi-indexed columns, including 'Params' as a level.
        location_name (str or list): Single location name or a list of locations to be plotted.
        param_list (list, optional): List of parameters to be plotted. If None, defaults to common parameters.
        std (bool, optional): Whether to display standard deviation shaded regions. Default is True.
        save_path (str, optional): Directory path to save the plot (default is './').
        dpi (int, optional): Dots per inch (DPI) for the saved figure (default is 600).
        figsize (tuple, optional): Figure size in inches (width, height). Defaults to (6, num_params).

    Returns:
        Saves the generated plot and displays it.
    """

    if isinstance(location_name, str):
        location_name = [location_name]

    dfa = df.stack(level='Params', future_stack=True).reset_index()
    dfa['Hour'] = dfa['Local Date/Time'].dt.hour
    dfa['Day_type'] = np.where(dfa['Local Date/Time'].dt.weekday < 5, 'Weekday', 'Weekend')

    if param_list is None:
        param_list = ['PM1 (μg/m³)', 'PM2.5 (μg/m³)', 'PM10 (μg/m³)', 'CO2 (ppm)',
                      'TVOC avg (ppb)', 'NOX index', 'Temperature (°C)', 'Humidity (%)']

    dfb = dfa.pivot_table(index='Hour', columns=['Params', 'Day_type'], aggfunc=["mean", "std"])
    dfb.columns.names = ['Stat', 'Location', 'Params', 'Day_type']
    dfb_swapped = dfb.swaplevel('Location', 'Stat', axis=1)

    param_labels = {
        'PM1 (μg/m³)': 'PM$_1$\n($\mu$g m$^{-3}$)',
        'PM2.5 (μg/m³)': 'PM$_{2.5}$\n($\mu$g m$^{-3}$)',
        'PM10 (μg/m³)': 'PM$_{10}$\n($\mu$g m$^{-3}$)',
        'PM2.5_10': 'PM$_{2.5-10}$\n($\mu$g m$^{-3}$)',
        'CO2 (ppm)': 'CO$_2$\n(ppm)',
        'TVOC avg (ppb)': 'TVOC\n(ppb)',
        'NOX index': 'NO$_x$',
        'Temperature (°C)': 'T\n(°C)',
        'Humidity (%)': 'RH\n(%)'
    }

    if figsize is None:
        figsize = (6, max(len(param_list), 1))

    fig, axes = plt.subplots(nrows=len(param_list), ncols=2, figsize=figsize, sharex=True, sharey='row')

    if len(param_list) == 1:
        axes = np.expand_dims(axes, axis=0)

    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    day_type_list = ['Weekday', 'Weekend']
    colors = plt.cm.get_cmap('tab10', len(location_name))
    legend_handles = []

    for i, poll in enumerate(param_list):
        for j, day_type in enumerate(day_type_list):
            for k, location in enumerate(location_name):
                if (location, 'mean', poll, day_type) in dfb_swapped.columns:
                    axes[i, j].plot(dfb_swapped.index, dfb_swapped[location]['mean'][poll][day_type],
                                    linestyle='-', color=colors(k), label=location)
                    if i == 0 and j == 1:
                        legend_handles.append(Line2D([0], [0], color=colors(k), linestyle='-', label=location))

                    # Add standard deviation shaded area if enabled
                    if std and (location, 'std', poll, day_type) in dfb_swapped.columns:
                        axes[i, j].fill_between(
                            dfb_swapped.index,
                            dfb_swapped[location]['mean'][poll][day_type] - dfb_swapped[location]['std'][poll][day_type],
                            dfb_swapped[location]['mean'][poll][day_type] + dfb_swapped[location]['std'][poll][day_type],
                            alpha=0.2,
                            color=colors(k)
                        )

            axes[i, j].set_xlim(0, 23)
            axes[i, j].set_xticks([0, 6, 12, 18])
            axes[i, j].set_xticklabels(['00:00', '06:00', '12:00', '18:00'], rotation=90)

            if j == 0:
                axes[i, 0].set_ylabel(param_labels.get(poll, poll), fontsize=7)

            axes[i, j].tick_params(axis='both', which='both', labelsize=7)

            if i == 0:
                axes[0, j].set_title(day_type, fontsize=7)

    for j in range(2):
        axes[-1, j].set_xlabel('Local time (hh:mm)', fontsize=7)

    max_cols = 3
    ncol = min(len(location_name), max_cols)
    nrows = np.ceil(len(location_name) / max_cols)
    y_anchor = 1.3
    height = 0.4 + (nrows - 1) * 0.25

    axes[0, 1].legend(handles=legend_handles,
                      bbox_to_anchor=(0.02, y_anchor, 1., height),
                      ncol=ncol,
                      fontsize=7,
                      frameon=False)

    if save_path:
        suffix = '_nostd' if not std else ''
        save_filename = save_path + '_'.join(location_name) + '_weekday_weekend' + suffix + '.png'
        plt.savefig(save_filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_filename} with DPI={dpi}.")

    plt.show()


def plot_air_quality(df, location_name, param_list, colors=None, start_date='2025-01-07', end_date=None, save_path='air_quality_plot.png'):
    """
    Plots multiple air quality parameters over time with subplots and saves the result.

    Parameters:
        df (DataFrame): Pandas DataFrame containing air quality data with timestamps as index.
        location_name (list): List of station names to compare.
        param_list (list): List of air quality parameters to be plotted.
        colors (list, optional): List of colors for different stations. If None, uses the default 'tab10' colormap.
        start_date (str, optional): Start date for the x-axis. Default is '2025-01-07'.
        end_date (str, optional): End date for the x-axis. If None, it uses the dataset's maximum timestamp.
        save_path (str, optional): Path to save the plot image. Default is 'air_quality_plot.png'.

    Returns:
        Saves the generated plot and displays it.
    """

    # Assign colors if not provided, using the 'tab10' colormap for distinct station colors
    if colors is None:
        colors = plt.cm.get_cmap('tab10', len(location_name)).colors

    # Set end_date to the dataset's maximum timestamp if not provided
    if end_date is None:
        end_date = df.index.max()

    # Create subplots (one for each air quality parameter)
    fig, axes = plt.subplots(nrows=len(param_list), ncols=1, figsize=(7, 7), sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.7, hspace=0.1)

    # Mapping pollutant names to formatted y-axis labels
    param_labels = {
        'PM1 (μg/m³)': 'PM$_1$\n($\mu$g m$^{-3}$)',
        'PM2.5 (μg/m³)': 'PM$_{2.5}$\n($\mu$g m$^{-3}$)',
        'PM10 (μg/m³)': 'PM$_{10}$\n($\mu$g m$^{-3}$)',
        'PM2.5_10': 'PM$_{2.5-10}$\n($\mu$g m$^{-3}$)',
        'CO2 (ppm)': 'CO$_2$\n(ppm)',
        'TVOC avg (ppb)': 'TVOC\n(ppb)',
        'NOX index': 'NO$_x$',
        'Temperature (°C)': 'T\n(°C)',
        'Humidity (%)': 'RH\n(%)'
    }

    # Loop through each pollutant to plot the time-series data
    for i, pollutant in enumerate(param_list):
        df[pollutant][location_name].plot(color=colors, ax=axes[i])  # Plot data for each station
        axes[i].set_xlim(start_date, end_date)  # Set x-axis range
        axes[i].set_ylabel(param_labels.get(pollutant, pollutant), fontsize=7)  # Set y-axis label
        axes[i].legend().set_visible(False)  # Hide legend in individual subplots
        axes[i].yaxis.set_major_locator(MaxNLocator(2))  # Limit y-axis ticks for clarity
        axes[i].tick_params(axis='both', which='both', labelsize=7)  # Adjust tick size
        axes[i].tick_params(axis='x', which='minor', bottom=False, top=False)

    # **Dynamically adjust the legend placement based on the number of locations**
    max_cols = 3  # Maximum number of columns in the legend
    ncol = min(len(location_name), max_cols)  # Set number of legend columns
    nrows = -(-len(location_name) // max_cols)  # Compute number of rows (ceil division)

    # Adjust legend placement dynamically
    y_anchor = 1.3  # Shift legend higher if more rows are needed
    height = 0.4 + (nrows - 1) * 0.25  # Increase legend height based on row count

    # Add the legend to the first subplot
    axes[0].legend(
        location_name,
        bbox_to_anchor=(0.02, y_anchor, 1, height),  # Adjust position dynamically
        ncol=ncol,
        fontsize=7,
        frameon=False
    )

    # Label the x-axis in the last subplot
    axes[-1].set_xlabel('Local Date/Time', fontsize=8)

    # Save and display the plot
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved as {save_path}")
    plt.show()


def diurnal_plot(df, param='PM2.5 (μg/m³)', location_name=None, time_resolution='1h', weekday='weekday', agg_func='mean', plot_kws=None):
    """
    Create and display a single plot with multiple columns overlaid.

    Parameters:
        df (pd.DataFrame): Input data.
        param (str): Parameter type ('pm25', 'nox', or 'co2').
        location_name (list): List of station names to compare.
        weekday (str): Filter for 'weekday', 'weekend', or 'all' data.
        agg_func (str or float): Aggregation function to apply ('mean', 'median', or a quantile value like 0.25).
        plot_kws (dict): Plot configuration parameters (e.g., width, height, title prefix, title font size).

    Returns:
        hvplot object: Interactive line plot of aggregated data.
    """
    # Define default plot parameters
    default_plot_kws = {
        'width': 1000,
        'height': 400,
        'title_fontsize': '12pt',
        'title_prefix': '',  # Default prefix for the plot title
        'palette': 'Paired'
    }

    # Update default parameters with user-specified values
    if plot_kws:
        default_plot_kws.update(plot_kws)

    # Extract plot parameters
    width = default_plot_kws['width']
    height = default_plot_kws['height']
    title_fontsize = default_plot_kws['title_fontsize']
    title_prefix = default_plot_kws['title_prefix']

    # Filter df for the specified parameter
    if location_name is None:
        location_name = list(df[param].columns)

    df = df[param][location_name].resample(time_resolution).mean()

    # Filter data based on weekday
    if weekday == 'weekday':
        df = df[df.index.weekday < 5]  # Monday to Friday
    elif weekday == 'weekend':
        df = df[df.index.weekday >= 5]  # Saturday and Sunday
    elif weekday != 'all':
        raise ValueError("Invalid weekday. Choose 'weekday', 'weekend', or 'all'.")

    # Compute hourly aggregation for the specified columns
    if agg_func == 'mean':
        hourly_agg = df.groupby(df.index.hour).mean()
        agg_label = "Mean"
    elif agg_func == 'median':
        hourly_agg = df.groupby(df.index.hour).median()
        agg_label = "Median"
    elif isinstance(agg_func, float) and 0 <= agg_func <= 1:
        hourly_agg = df.groupby(df.index.hour).quantile(agg_func)
        agg_label = f"{int(agg_func * 100)}th Percentile"
    else:
        raise ValueError("Invalid agg_func. Use 'mean', 'median', or a float between 0 and 1 (e.g., 0.25 for 25th percentile).")

    # Generate the combined plot
    plot = hourly_agg.hvplot.line(
        xlabel="Hour of the Day",
        ylabel=param,
        width=width,
        height=height,
        title=f"{title_prefix} Diurnal variation ({agg_label} - {weekday.capitalize()})",
        legend=True
    ).opts(
        fontsize={'title': title_fontsize},  # Set title font size
    )

    return plot


def create_boxplot(df, param='PM2.5 (μg/m³)', location_name=None,
                   time_resolution='1h', weekday='all', plot_kws=None):
    """
    Create and display a box plot with optional mean points for a specific parameter.

    Parameters:
        df (pd.DataFrame): Input data.
        param (str): Parameter type ('pm25', 'nox', 'co2', etc.).
        location_name (list): List of station names to compare.
        weekday (str): Filter for 'weekday', 'weekend', or 'all' data.
        plot_kws (dict): Plot configuration parameters (e.g., width, height, title prefix, title font size).

    Returns:
        hvplot object: Interactive box plot with optional mean points.
    """
    # Define default plot parameters
    default_plot_kws = {
        'width': 1000,
        'height': 400,
        'title_fontsize': '12pt',
        'title_prefix': ' ',  # Default prefix for the plot title
        'show_mean': True,    # Whether to show mean points on the box plot
    }

    # Update default parameters with user-specified values
    if plot_kws:
        default_plot_kws.update(plot_kws)

    # Extract plot parameters
    width = default_plot_kws['width']
    height = default_plot_kws['height']
    title_fontsize = default_plot_kws['title_fontsize']
    title_prefix = default_plot_kws['title_prefix']
    show_mean = default_plot_kws['show_mean']

    # Filter df for the specified parameter
    if location_name is None:
        location_name = list(df[param].columns)

    df = df[param][location_name].resample(time_resolution).mean()

    # Filter data based on weekday
    if weekday == 'weekday':
        df = df[df.index.weekday < 5]  # Monday to Friday
    elif weekday == 'weekend':
        df = df[df.index.weekday >= 5]  # Saturday and Sunday
    elif weekday != 'all':
        raise ValueError("Invalid weekday. Choose 'weekday', 'weekend', or 'all'.")

    # Identify relevant columns
    cols = df.columns
    if not cols.any():
        raise ValueError(f"No columns found for parameter: {param}")

    # Calculate means (optional)
    means = df.mean()

    # Create the box plot
    boxplot = df.hvplot.box(
        title=f"Box Plot of {param} ({weekday.capitalize()})",
        ylabel=param,
        width=width,
        height=height,
        legend=False,
    ).opts(
        xrotation=90,  # Rotate x-axis tick labels
        fontsize={'title': title_fontsize},  # Set title font size
        xlabel=''  # Remove x-axis label
    )

    # Add mean points if enabled
    if show_mean:
        mean_dots = hv.Scatter((cols, means), "Column", "Mean").opts(
            color="red",
            marker="o",
            size=6,  # Adjust mean point size
            show_legend=True
        )
        combined_plot = (boxplot * mean_dots).opts(
            legend_labels={1: "Mean"},  # Add legend label for mean points
            show_legend=True
        )
        return combined_plot
    else:
        return boxplot


def correction(df):
    """
    Apply corrections to PM2.5, Temperature, and Relative Humidity values based on specified formulas.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'PM2.5 (μg/m³)', 'Humidity (%)', and 'Temperature (°C)' columns.

    Returns:
        pd.DataFrame: DataFrame with new corrected columns for PM2.5, Temperature, and Relative Humidity.
    """
    # Create corrected columns
    df['PM2.5 (μg/m³)_r'] = np.nan
    df['Temperature (°C)_r'] = np.nan
    df['Humidity (%)_r'] = np.nan

    # Extract raw columns
    AGraw = df['PM2.5 (μg/m³)']
    RHraw = df['Humidity (%)']
    Traw = df['Temperature (°C)']

    # PM2.5 correction
    # Case 1: AGraw < 30
    mask1 = AGraw < 30
    df.loc[mask1, 'PM2.5 (μg/m³)_r'] = (
        AGraw[mask1] * 0.54 -
        RHraw[mask1] * 0.0862 +
        5.75
    )

    # Case 2: 30 ≤ AGraw < 50
    mask2 = (AGraw >= 30) & (AGraw < 50)
    term2 = AGraw[mask2] / 20 - 3 / 2
    df.loc[mask2, 'PM2.5 (μg/m³)_r'] = (
        term2 * 0.786 +
        0.524 * (1 - term2) * AGraw[mask2] -
        0.0862 * RHraw[mask2] +
        5.75
    )

    # Case 3: 50 ≤ AGraw < 210
    mask3 = (AGraw >= 50) & (AGraw < 210)
    df.loc[mask3, 'PM2.5 (μg/m³)_r'] = (
        AGraw[mask3] * 0.786 -
        0.0862 * RHraw[mask3] +
        5.75
    )

    # Case 4: 210 ≤ AGraw < 260
    mask4 = (AGraw >= 210) & (AGraw < 260)
    term4 = AGraw[mask4] / 50 - 21 / 5
    df.loc[mask4, 'PM2.5 (μg/m³)_r'] = (
        (0.69 * term4 + 0.786 * (1 - term4)) * AGraw[mask4] -
        0.0862 * RHraw[mask4] * (1 - term4) +
        2.966 * term4 +
        5.75 * (1 - term4) +
        8.84e-4 * AGraw[mask4]**2 * term4
    )

    # Case 5: AGraw ≥ 260
    mask5 = AGraw >= 260
    df.loc[mask5, 'PM2.5 (μg/m³)_r'] = (
        2.966 +
        0.69 * AGraw[mask5] +
        8.84e-4 * AGraw[mask5]**2
    )

    # Replace negative PM2.5 values with 0
    df['PM2.5 (μg/m³)_r'] = df['PM2.5 (μg/m³)_r'].clip(lower=0)

    # Temperature calibration
    df['Temperature (°C)_r'] = np.where(
        Traw < 10,
        Traw * 1.327 - 6.738,
        Traw * 1.181 - 5.113
    )

    # Relative Humidity calibration
    df['Humidity (%)_r'] = (RHraw * 1.259 + 7.34).clip(upper=100)

    return df


def compare_sensor_to_reference(mod_file, ref_file, ref_sensors, time_resolution='1H'):
    """
    Compare MOD sensor data to a list of reference sensors and calculate comprehensive evaluation statistics.

    Parameters:
        mod_file (str): Path to MOD sensor CSV file.
        ref_file (str): Path to reference sensor CSV file.
        ref_sensors (list): List of reference sensor names to include.
        time_resolution (str): Resampling frequency (default is '1H' for hourly).

    Returns:
        pd.DataFrame: Evaluation metrics for each pollutant and reference sensor.
    """
    # Load data
    mod_df = pd.read_csv(mod_file)
    ref_df = pd.read_csv(ref_file)

    # Preprocess reference data
    ref_subset = ref_df[ref_df['Location Name'].isin(ref_sensors)].copy()
    ref_subset['Local Date/Time'] = pd.to_datetime(ref_subset['Local Date/Time'])
    ref_subset.set_index('Local Date/Time', inplace=True)

    # Preprocess MOD data
    mod_df['timestamp_local'] = pd.to_datetime(mod_df['timestamp_local'])
    mod_df.set_index('timestamp_local', inplace=True)
    mod_df.index = mod_df.index.tz_localize(None)

    # Resample numeric data only
    mod_numeric = mod_df.select_dtypes(include=['number'])
    mod_resampled = mod_numeric.resample(time_resolution).mean()

    numeric_cols = ref_subset.select_dtypes(include=['number']).columns
    ref_numeric = ref_subset[numeric_cols.union(['Location Name'], sort=False)]
    ref_resampled = ref_numeric.groupby('Location Name').resample(time_resolution).mean().reset_index()

    # Define pollutant mapping
    pollutants = {
        'PM1 (μg/m³)': 'pm1',
        'PM2.5 (μg/m³) raw': 'pm25',
        'PM10 (μg/m³)': 'pm10',
        'CO2 (ppm) raw': 'co2',
        'Temperature (°C) raw': 'temp',
        'Humidity (%) raw': 'rh'
    }

    all_results = []

    for sensor in ref_sensors:
        ref_sensor_data = ref_resampled[ref_resampled['Location Name'] == sensor].copy()
        ref_sensor_data.set_index('Local Date/Time', inplace=True)

        # Join with MOD data
        merged = ref_sensor_data.join(mod_resampled, how='inner', lsuffix='_ref', rsuffix='_mod')

        for ref_col, mod_col in pollutants.items():
            if ref_col in merged.columns and mod_col in merged.columns:
                df_valid = merged[[ref_col, mod_col]].dropna()
                if df_valid.shape[0] > 10:
                    stats_result = Stats(df_valid, mod=mod_col, obs=ref_col)
                    stats_result.insert(0, 'Pollutant', ref_col)
                    stats_result.insert(0, 'Sensor', sensor)
                    all_results.append(stats_result)

    # Concatenate all stats results
    return pd.concat(all_results, ignore_index=True)
