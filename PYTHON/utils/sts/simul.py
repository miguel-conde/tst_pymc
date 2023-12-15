import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def stochastic_trend(
        rng, 
        N = 100, 
        sigma_local_trend = 1, 
        local_trend_0 = 1, 
        sigma_local_level = 1, 
        local_level_0 = 100
        ):
    """Simulate a stochastic trend

    Args:
        rng (_type_): _description_
        N (int, optional): _description_. Defaults to 100.
        sigma_local_trend (int, optional): _description_. Defaults to 1.
        sigma_local_level (int, optional): _description_. Defaults to 1.
        local_level_0 (int, optional): _description_. Defaults to 100.
        local_trend_0 (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # local trend is a random walk
    diff_local_trend = rng.normal(0, sigma_local_trend, N)
    local_trend      = np.cumsum(diff_local_trend) + local_trend_0

    # local level is the local trend plus noise
    diff_local_level = rng.normal(local_trend, sigma_local_level, N)
    local_level      = np.cumsum(diff_local_level) + local_level_0

    # dataframe with t = 1:N, local_trend, local_level
    df = pd.DataFrame(
        {
            "t": np.arange(1, N + 1),
            "local_trend": local_trend,
            "local_level": local_level,
        }
    )
    return df

def plot_stochastic_trend(df):
    """plot local_trend and local_level vs t, each with a different y-axis

    Args:
        df (_type_): _description_
    """
    # plot local_trend and local_level vs t, each with a different y-axis
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    df.plot(x="t", y=["local_trend"], ax=ax1, alpha=0.5)
    df.plot(x="t", y=["local_level"], ax=ax2, alpha=0.5, color="C1")

    # Add y-axis labels
    ax1.set_ylabel("local_trend")
    ax2.set_ylabel("local_level")

    # Add a legend
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Add a title
    plt.title("Local Trend and Local Level")

    # Add horizontal dotted line at local_trend = 0
    ax1.axhline(0, linestyle="--", color="black", alpha=0.5)

    fig.tight_layout()
    plt.show()

def local_level_trend(
        rng, 
        N = 100, 
        sigma = 1, 
        level_0 = 100
        ):
    """simulate a local level trend

    Args:
        rng (_type_): _description_
        N (int, optional): _description_. Defaults to 100.
        sigma_local_trend (int, optional): _description_. Defaults to 1.
        sigma_local_level (int, optional): _description_. Defaults to 1.
        local_level_0 (int, optional): _description_. Defaults to 100.
        local_trend_0 (int, optional): _description_. Defaults to 1.
    """

    df = stochastic_trend(
        rng, 
        N = N, 
        sigma_local_trend = 0, 
        local_trend_0 = 0, 
        sigma_local_level = sigma, 
        local_level_0 = level_0
    )

    return df

def deterministic_trend(
        rng,
        N = 100,
        slope = 1,
        intercept = 100
):
    """Generate a deterministic trend.

    This function generates a deterministic trend using the specified parameters.

    Args:
        rng (numpy.random.Generator): The random number generator.
        N (int, optional): The number of data points to generate. Defaults to 100.
        slope (float, optional): The slope of the trend line. Defaults to 1.
        intercept (float, optional): The intercept of the trend line. Defaults to 100.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated trend data.
    """

    df = stochastic_trend(
        rng, 
        N = N, 
        sigma_local_trend = 0, 
        local_trend_0 = slope, 
        sigma_local_level = 0, 
        local_level_0 = intercept
    )

    return df

def make_fourier_features(df, date_var_name, n_order=10, period=365.25):
    """
    Generate Fourier features based on a given date variable in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the date variable.
        date_var_name (str): The name of the date variable column in the DataFrame.
        n_order (int, optional): The number of Fourier orders to generate. Defaults to 10.
        period (float, optional): The period of the Fourier series. Defaults to 365.25.

    Returns:
        pandas.DataFrame: A DataFrame containing the Fourier features.
    """
    # Calculate the periods based on the date variable
    periods = (df[date_var_name] - pd.Timestamp("1900-01-01")).dt.days / period

    # Generate the Fourier features using sine and cosine functions
    fourier_features = pd.DataFrame(
        {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )

    return fourier_features


##################################################################################################
#
# https://discourse.pymc.io/t/pymc-experimental-now-includes-state-spaces-models/12773
# https://github.com/pymc-devs/pymc-experimental/blob/main/notebooks/Structural%20Timeseries%20Modeling.ipynb
#
##################################################################################################

from pymc_experimental.statespace import structural as st
from pymc_experimental.statespace.utils.constants import SHORT_NAME_TO_LONG, MATRIX_NAMES
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pytensor
import pytensor.tensor as pt
import numpy as np
import pandas as pd
from patsy import dmatrix

def unpack_statespace(ssm):
    """
    Unpacks the symbolic matrices from the statespace model.

    Args:
        ssm (dict): Dictionary containing the symbolic matrices.

    Returns:
        list: List of unpacked symbolic matrices.
    """
    return [ssm[SHORT_NAME_TO_LONG[x]] for x in MATRIX_NAMES]


def unpack_symbolic_matrices_with_params(mod, param_dict):
    """
    Unpacks the symbolic matrices from the statespace model with parameter values.

    Args:
        mod (type): The statespace model.
        param_dict (dict): Dictionary containing the parameter values.

    Returns:
        tuple: Tuple containing the unpacked symbolic matrices.
    """
    f_matrices = pytensor.function(
        list(mod._name_to_variable.values()), unpack_statespace(mod.ssm), on_unused_input="ignore"
    )
    x0, P0, c, d, T, Z, R, H, Q = f_matrices(**param_dict)
    return x0, P0, c, d, T, Z, R, H, Q


def simulate_from_numpy_model(mod, rng, param_dict, steps=100):
    """
    Helper function to simulate the components of a structural timeseries model outside of a PyMC model context.

    Args:
        mod (type): The structural timeseries model.
        rng (numpy.random.Generator): Random number generator.
        param_dict (dict): Dictionary containing the parameter values.
        steps (int, optional): Number of steps to simulate. Defaults to 100.

    Returns:
        tuple: Tuple containing the simulated components.
    """
    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, param_dict)
    Z_time_varies = Z.ndim == 3

    k_states = mod.k_states
    k_posdef = mod.k_posdef

    x = np.zeros((steps, k_states))
    y = np.zeros(steps)

    x[0] = x0
    if Z_time_varies:
        y[0] = Z[0] @ x0
    else:
        y[0] = Z @ x0

    if not np.allclose(H, 0):
        y[0] += rng.multivariate_normal(mean=np.zeros(1), cov=H)

    for t in range(1, steps):
        if k_posdef > 0:
            shock = rng.multivariate_normal(mean=np.zeros(k_posdef), cov=Q)
            innov = R @ shock
        else:
            innov = 0

        if not np.allclose(H, 0):
            error = rng.multivariate_normal(mean=np.zeros(1), cov=H)
        else:
            error = 0

        x[t] = c + T @ x[t - 1] + innov

        if Z_time_varies:
            y[t] = d + Z[t] @ x[t] + error
        else:
            y[t] = d + Z @ x[t] + error

    return x, y


def simulate_many_trajectories(mod, rng, param_dict, n_simulations, steps=100):
    """
    Simulates multiple trajectories of a structural timeseries model.

    Args:
        mod (type): The structural timeseries model.
        rng (numpy.random.Generator): Random number generator.
        param_dict (dict): Dictionary containing the parameter values.
        n_simulations (int): Number of simulations to perform.
        steps (int, optional): Number of steps to simulate. Defaults to 100.

    Returns:
        tuple: Tuple containing the simulated trajectories.
    """
    k_states = mod.k_states
    k_posdef = mod.k_posdef

    xs = np.zeros((n_simulations, steps, k_states))
    ys = np.zeros((n_simulations, steps))

    for i in range(n_simulations):
        x, y = simulate_from_numpy_model(mod, rng, param_dict, steps)
        xs[i] = x
        ys[i] = y
    return xs, ys


def simulate_regression_component(rng, k_exog = 3, n_obs = 100):
    """
    Simulate a regression component using the provided parameters.

    Args:
        rng (numpy.random.Generator): Random number generator.
        param_dict (dict): Dictionary of parameters for the regression component.

    Returns:
        numpy.ndarray: Simulated regression coefficients.
        numpy.ndarray: Simulated observed state.
    """
    exog_data = rng.normal(size=(n_obs, k_exog))
    true_betas = rng.normal(size=(k_exog,))
    param_dict = {"beta_exog": true_betas, "data_exog": exog_data}

    reg = st.RegressionComponent(name="exog", k_exog=k_exog, innovations=False)
    x, y = simulate_from_numpy_model(reg, rng, param_dict)

    return x, y


def plot_exog_regression(x, y, data):
    """
    Plot the observed state, hidden states (data), and regression coefficients.

    Args:
        x (array-like): The regression coefficients.
        y (array-like): The observed state.
        data (array-like): The hidden states (data).
    """
    fig = plt.figure(figsize=(14, 6))
    gs = plt.GridSpec(nrows=2, ncols=2, figure=fig)

    y_axis = fig.add_subplot(gs[0, :])
    data_axis = fig.add_subplot(gs[1, 0])
    param_axis = fig.add_subplot(gs[1, 1])
    axes = [y_axis, data_axis, param_axis]
    datas = [y, data, x]
    titles = ["Observed State", "Hidden States (Data)", "Regression Coefficients"]
    for axis, data, title in zip(axes, datas, titles):
        axis.plot(data)
        axis.set_title(title)
    plt.show()