import pandas as pd
import numpy as np
import xarray as xr
import arviz as az

from typing import Optional, List
from pandas.core.frame import DataFrame

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

def make_fourier_features(df, date_var_name, n_order=10, period=365.25):
    """
    Generate Fourier features based on a given date variable in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the date variable.
        date_var_name (str): The name of the date variable column in the DataFrame.
            The variable must be of type pandas.Timestamp.
        n_order (int, optional): The number of Fourier orders to generate. Defaults to 10.
        period (float, optional): The period of the Fourier series. Defaults to 365.25.
            Always use days as the unit of the period.

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


######## CLASE CustomGroupScaler
class CustomGroupScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, group_columns, method='mean'):
        """
        Inicializa el transformer de escalado por grupo.

        :param columns: Lista de nombres de columnas a escalar.
        :param group_columns: Lista de nombres de columnas que definen los grupos.
        :param method: Método de escalado ('max', 'min', 'mean', 'median').
        
        :example:
        
        scaler = CustomGroupScalerModified(columns=['x1', 'x2', 'x3'], group_columns=['grupo'], method='mean')
        
        scaler.fit(df_onehot)
        
        df_scaled = scaler.transform(df_onehot)
        
        df_reversed_specific = scaler.inverse_transform(df_scaled, columns=['x1', 'x2'])
        
        df_reversed_all = scaler.inverse_transform(df_scaled)
        
        """
        self.columns = columns
        self.group_columns = group_columns
        self.method = method
        self.scaling_factors_ = {}
        self.__fitted__ = False

    def fit(self, X, y=None):
        """
        Calcula los factores de escalado para cada grupo y cada columna.

        :param X: DataFrame de entrada.
        :param y: No utilizado, existe por compatibilidad.
        :return: self.
        """
        for col in self.group_columns:
            if col not in X.columns:
                raise ValueError(f"La columna '{col}' no está presente en el DataFrame.")

        for col in self.columns:
            self.scaling_factors_[col] = X.groupby(self.group_columns)[col].agg(self.method)
            
        self.__fitted__ = True
        
        return self

    def transform(self, X):
        """
        Aplica la transformación de escalado al DataFrame.

        :param X: DataFrame de entrada.
        :return: DataFrame transformado.
        """
    
        check_is_fitted(self, ['__fitted__'])
        
        X_transformed = X.copy()
        for col in self.columns:
            for group, factor in self.scaling_factors_[col].items():
                group_condition = (X[self.group_columns] == pd.Series(group, index=self.group_columns)).all(axis=1)
                X_transformed.loc[group_condition, col] /= factor
        return X_transformed

    def inverse_transform(self, X, columns=None):
        """
        Invierte la transformación de escalado aplicada al DataFrame.

        :param X: DataFrame transformado.
        :param columns: Lista de columnas a desescalar. Si es None, desescala todas las posibles.
        :return: DataFrame original.
        """
        
        check_is_fitted(self, ['__fitted__'])
        
        X_inversed = X.copy()
        columns_to_scale = self.columns if columns is None else columns

        for col in columns_to_scale:
            if col in self.scaling_factors_:
                for group, factor in self.scaling_factors_[col].items():
                    group_condition = (X[self.group_columns] == pd.Series(group, index=self.group_columns)).all(axis=1)
                    X_inversed.loc[group_condition, col] *= factor

        return X_inversed

# Ejemplo de cómo se usaría el transformer con la opción de desescalar columnas específicas
# scaler = CustomGroupScalerModified(columns=['x1', 'x2', 'x3'], group_columns=['grupo'], method='mean')
# scaler.fit(df_onehot)
# df_scaled = scaler.transform(df_onehot)
# df_reversed_specific = scaler.inverse_transform(df_scaled, columns=['x1', 'x2'])
# df_reversed_all = scaler.inverse_transform(df_scaled)


######## CLASE GeomAdStock
class GeomAdStock(BaseEstimator, TransformerMixin):
    
    def __init__(self, ads, valor_inicial = 0.0):
        """Esta clase implementa como transformer sklearn el modelo de Adstock Geométrico para una variable.

        Args:
            ads (float): decay rate del adstock. Puede ir entre 0.0 y 1.0 o entre 0.0 y 100.0
            valor_inicial (float, optional): valor inicial de la serie adstock. Defaults to 0.
        """
        
        if (ads > 1):
            self.ads = ads / 100
        else:
            self.ads = ads
        self.valor_inicial = valor_inicial
        self.__fitted__ = False
    
    def fit(self, X = None, y = None):
        
        self.n_cols = X.shape[1]
        self.n_rows = X.shape[0]
        
        self.alpha = np.zeros((self.n_rows, self.n_rows))
        
        for j in np.arange(self.n_rows):
            for i in np.arange(j, self.n_rows):
                self.alpha[i, j] = self.ads**(i - j)
                
        self.alpha_0 = self.alpha[:, 0].reshape(-1,1) * self.ads
        self.valor_inicial = np.full(shape = (1, self.n_cols), fill_value = self.valor_inicial)
        
        self.__fitted__ = True
        
        return self
    
    def transform(self, X, y = None):
        
        check_is_fitted(self, ['__fitted__'])
        
        if (self.n_rows != X.shape[0]):
            return X

        return np.matmul(self.alpha, X) + np.matmul(self.alpha_0, self.valor_inicial)
    
    


def reshape_x_groups(X: DataFrame, group_var_name: str = "group", to_drop_var_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Reshape X to a 3D array.

    Args:
        X (DataFrame): The input DataFrame.
        group_var_name (str, optional): The name of the column that represents the groups. Defaults to "group".
        to_drop_var_names (list, optional): A list of column names to drop from the reshaped array. Defaults to None.

    Returns:
        ndarray: The reshaped 3D array.
    """
    # Step 1: Identify the number of unique groups
    num_grupos = X[group_var_name].nunique()
    
    # Step 2: Create an empty 3D array
    X_g = np.zeros((int(X.shape[0] / num_grupos), X.shape[1] - len(to_drop_var_names), num_grupos))
    
    # Step 3: Fill the 3D array
    for i, grupo in enumerate(X[group_var_name].unique()):
        datos_grupo = X[X[group_var_name] == grupo]
        X_g[:, :, i] = datos_grupo.drop(to_drop_var_names, axis=1).to_numpy()
        
    return X_g


def make_xarray_x_group(X: pd.DataFrame, group_var_name: str = "group", to_drop_var_names: list = None) -> xr.DataArray:
    """
    Reshape X to a DataArray with group dimensions.

    Args:
        X (DataFrame): The input DataFrame.
        group_var_name (str, optional): The name of the column that represents the groups. Defaults to "group".
        to_drop_var_names (list, optional): A list of column names to drop from the reshaped array. Defaults to None.

    Returns:
        DataArray: The reshaped DataArray with group dimensions.
    """
    grupos = X[group_var_name].unique()
    num_grupos = len(grupos)
    num_filas = X.shape[0] // num_grupos
    
    reshaped_array = reshape_x_groups(X, group_var_name=group_var_name, to_drop_var_names=to_drop_var_names)
    
    xr_array = xr.DataArray(
        reshaped_array,
        dims=["index", "variables", "grupos"], 
        coords={"index": np.arange(num_filas), "variables": X.columns.drop(["group", "date_week"]), "grupos": grupos})
    
    return xr_array

def compute_metrics(idata: az.InferenceData, Y: pd.core.frame.DataFrame, y_hat: str = "mu", y: str = "y", group: str = 'group') -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compute various metrics for evaluating model performance.

    Args:
        idata (az.InferenceData): The input ArviZ InferenceData object.
        Y (pd.core.frame.DataFrame): The target variable.
        y_hat (str, optional): The predicted variable. Defaults to "mu".
        y (str, optional): The actual variable. Defaults to "y".
        group (str, optional): The grouping variable. Defaults to 'group'.

    Returns:
        Tuple[xr.DataArray]: A tuple containing the computed samples metrics for each group:
            - rmse: Root Mean Squared Error for each group
            - mse: Mean Squared Error for each group
            - mae: Mean Absolute Error for each group
            - mape: Mean Absolute Percentage Error for each group
            - wmape: Weighted Mean Absolute Percentage Error for each group
            - r2: R-squared for each group
            
    Example:
    ```
        rmse, mse, mae, mape, wmape, r2 = compute_metrics(idata_2, Y_scaled)
        
        mape.mean('sample')
        
        import seaborn as sns
        sns.distplot(100*mape.loc[:,'group_1'].values, hist=True, kde=True,);
    ```
    """
    
    # Extract predicted values from idata
    y_pred = az.extract(idata, group="posterior")[y_hat]
    
    # Compute error metrics
    error = y_pred - Y[y].values.reshape(-1,1)
    abs_error = np.abs(error)
    se = np.square(error)
    
    # Compute metrics for each group
    groups = Y[group].unique() - 1
    rmse_dict = {}
    mae_dict = {}
    mse_dict = {}
    mape_dict = {}
    wmape_dict = {}
    r2_dict = {}
    for g in groups:
        grp_str = 'group_' + str(g+1)
        g_mask = (Y[group] == g+1) 
        
        weights_g = Y[y][g_mask]/Y[y][g_mask].sum()
        weights_g = weights_g / weights_g.sum()
        
        mae_dict[grp_str] = abs_error[g_mask,:].mean('obs')
        mse_dict[grp_str] = se[g_mask,:].mean('obs')
        rmse_dict[grp_str] = np.sqrt(mse_dict[grp_str])
        mape_dict[grp_str] = (abs_error[g_mask,:] / Y[y].values[g_mask].reshape(-1,1)).mean('obs')
        wmape_dict[grp_str] = (weights_g.values.reshape(-1,1) * abs_error[g_mask, :]).sum('obs')
        r2_dict[grp_str] = 1 - (se[g_mask,:].sum('obs') / np.square(Y[y].values[g_mask].reshape(-1,1)).sum(axis = 0))
        
    # Convert metrics to xarray DataArrays
    rmse = xr.DataArray(pd.DataFrame(rmse_dict).values, dims = ['sample', 'group'], coords = {'group': list(rmse_dict.keys())})
    mse  = xr.DataArray(pd.DataFrame(mse_dict).values,  dims = ['sample', 'group'], coords = {'group': list(mse_dict.keys())})
    mae  = xr.DataArray(pd.DataFrame(mae_dict).values,  dims = ['sample', 'group'], coords = {'group': list(mae_dict.keys())})
    mape = xr.DataArray(pd.DataFrame(mape_dict).values, dims = ['sample', 'group'], coords = {'group': list(mape_dict.keys())})
    wmape = xr.DataArray(pd.DataFrame(wmape_dict).values, dims = ['sample', 'group'], coords = {'group': list(wmape_dict.keys())})
    r2   = xr.DataArray(pd.DataFrame(r2_dict).values,   dims = ['sample', 'group'], coords = {'group': list(r2_dict.keys())})
    
    return rmse, mse, mae, mape, wmape, r2
