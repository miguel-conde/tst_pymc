import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from typing import Union, Dict
from pymc_experimental.model_builder import ModelBuilder
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation

from utils.tools import CustomGroupScaler, make_fourier_features



class Model2(ModelBuilder):
    # Give the model a name
    _model_type = "Trend Season GeomAS Logistic Saturation"

    # And a version
    version = "0.1"
    
    def check_X_y(self, X, y):
        """

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        # Check X is a dataframe
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        # Check X contains self.name_group_var, self.name_date_var, self.media_vars and self.control_vars
        if not all([var in X.columns for var in [self.name_group_var, self.name_date_var, *self.media_vars, *self.control_vars]]):
            raise ValueError(f"X must contain {self.name_group_var}, {self.name_date_var}, {self.media_vars} and {self.control_vars}")
        # Check y is a numpy array
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy array")
        # Check y contains self.name_tgt_var and self.name_group_var
        # if not all([var in y.columns for var in [self.name_tgt_var, self.name_group_var]]):
        #     raise ValueError(f"y must contain {self.name_tgt_var} and {self.name_group_var}")
        
    def def_model(self, coords, X_scaled, Y_scaled, groups, groups_idx, fourier_features, media_vars, control_vars, L_MAX):
    
        with pm.Model(coords = coords) as model_2:
        ### DATA
            X_t       = pm.MutableData('X_t',       X_scaled['t'],          dims=('obs'))
            X_fourier = pm.MutableData('X_fourier', fourier_features,       dims=('obs', 'fourier_features'))
            X_media   = pm.MutableData('X_media',   X_scaled[media_vars],   dims=('obs', 'media_vars'))
            X_control = pm.MutableData('X_control', X_scaled[control_vars], dims=('obs', 'control_vars'))
            Y         = pm.MutableData('Y',         Y_scaled,               dims='obs')
    
        ## HYPERPRIORS
        
        # Intercept positiva
            alpha_trend       = pm.HalfNormal("alpha", sigma=self.model_config.get("sigma_alpha_trend", 1))
            sigma_alpha_trend = pm.Exponential("sigma_alpha_trend", lam=self.model_config.get("lam_sigma_alpha_trend", 1))
            beta_trend        = pm.Normal('beta_trend', mu = 0, sigma = self.model_config.get("sigma_beta_trend", 1))
            sigma_beta_trend  = pm.Exponential("sigma_beta_trend", lam=self.model_config.get("lam_sigma_beta_trend", 1))
            k                 = pm.Uniform("k", lower=self.model_config.get("lower_k", 0.1), upper=self.model_config.get("upper_k", 4))
        # Media vars positivas
            mu_media_var      = pm.Exponential('mu_media_var',    lam = self.model_config.get("lam_mu_media_var", 1))
            sigma_media_var   = pm.Exponential('sigma_media_var', lam = self.model_config.get("lam_sigma_media_var", 1))
        # Adstock - para la distribucion beta del recuerdo
            alpha_as_media_var = pm.Uniform(
                f'alpha_as_media_var', 
                lower=self.model_config.get("lower_alpha_as_media_var", 1), 
                upper=self.model_config.get("upper_alpha_as_media_var", 4)
                )
            beta_as_media_var  = pm.Uniform(
                f'beta_as_media_var', 
                lower=self.model_config.get("lower_beta_as_media_var", 1), 
                upper=self.model_config.get("upper_beta_as_media_var", 4)
                )
        # Saturations - para la distribución exponencial de la saturación
            lam_media_var      = pm.Exponential(f'lam_media_var', lam = self.model_config.get("lam_lam_media_var", 1))
        # Control vars libres
            mu_control_var    = pm.Normal(
                'mu_control_var', 
                mu = self.model_config.get("mu_mu_control_var", 0), 
                sigma = self.model_config.get("sigma_mu_control_var", 1)
                )
            sigma_control_var = pm.Exponential('sigma_control_var', lam = self.model_config.get("lam_sigma_control_var", 1))
        
        # Noise
            sigma_noise       = pm.Exponential('noise', lam = self.model_config.get("lam_sigma_noise", 1), dims='group')
        
        ## COMPONENTES
        
        # Trend
            alphas_trend     = pm. Normal('alphas_trend', mu = alpha_trend, sigma = sigma_alpha_trend, dims='group')
            betas_trend      = pm. Normal('betas_trend',  mu = beta_trend,  sigma = sigma_beta_trend,  dims='group')    
            ks               = pm. Normal('ks',           mu = k,           sigma = 0.5,               dims='group')    
        
            trend = pm.Deterministic("trend", alphas_trend[groups_idx] + betas_trend[groups_idx] * X_t**ks[groups_idx], dims="obs")
        
        # Seasonality
            betas_fourier = pm.Normal("beta_fourier", mu=0, sigma=1, dims = ("group", "fourier_features"))
        
            seasonality = pm.Deterministic(
            "seasonality", 
            # pm.math.dot(X_fourier[groups_idx], betas_fourier[groups_idx]),
            (betas_fourier[groups_idx,:] * X_fourier).sum(axis=-1),
            dims = "obs"
        )
            
        # Coeficientes
            betas  = pm.Normal(f'beta',  mu = mu_media_var, sigma=sigma_media_var,     dims = ('group', 'media_vars'))
            gammas = pm.Normal(f'gamma', mu = mu_control_var, sigma=sigma_control_var, dims = ('group', 'control_vars'))
        
        # Pars Adstock
            alphas_as = pm.Beta(f'alpha_as_media_vars', alpha = alpha_as_media_var, beta = beta_as_media_var, dims = ('group', 'media_vars'))
        
        # Pars Saturation
            lams = pm.Exponential(f'lam_media_vars', lam = lam_media_var, dims = ('group', 'media_vars'))
        
        # Adstock 
            geom_as_media_var_list = []
    
        # Itera sobre cada grupo y aplica adstock, luego agrega el resultado a la lista
            for g in groups:
                adstocked = geometric_adstock(x=X_media[g == (groups_idx + 1)], alpha=alphas_as[g-1], l_max=L_MAX, normalize=True, axis=0)
                geom_as_media_var_list.append(adstocked)
    
        # Concatena los resultados para formar el tensor geom_as_media_var
            geom_as_media_var = pm.Deterministic(
            'geom_as_media_var',
            pt.concatenate(geom_as_media_var_list, axis=0), 
            dims=('obs', 'media_vars')
            )
        
        # Saturacion
            satur_geom_as_media_var = pm.Deterministic(
            'satur_geom_as_media_var', 
            logistic_saturation(x=geom_as_media_var, lam=lams[groups_idx]), 
            dims=('obs', 'media_vars')
            )
        
        # Contribuciones
            contribs_media_vars   = pm.Deterministic(f'contrib_media_vars',   betas[groups_idx] * satur_geom_as_media_var , dims=('obs', 'media_vars'))
            contribs_control_vars = pm.Deterministic(f'contrib_control_vars', gammas[groups_idx] * X_control ,              dims=('obs', 'control_vars'))
        
        # Media de ventas
            mu = pm.Deterministic(
            "mu", 
            contribs_media_vars.sum(axis=-1) + contribs_control_vars.sum(axis=-1) + trend + seasonality, 
            dims = 'obs'
            )
        
        ### LIKELIHOOD 
            pm.Normal("y_obs", mu=mu, sigma=sigma_noise[groups_idx], observed=Y)
            
        return model_2


    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """
        
        self.n_order        = kwargs.get('n_order',        self.model_config['n_order'])        # order of the fourier series
        self.L_MAX          = kwargs.get('L_MAX',          self.model_config['L_MAX'])          # order of the adstock
        self.name_date_var  = kwargs.get('name_date_var',  self.model_config['name_date_var'])  # name of the date variable
        self.name_group_var = kwargs.get('name_group_var', self.model_config['name_group_var']) # name of the group variable
        self.period         = kwargs.get('period',         self.model_config['period'])         # period of the seasonality
        self.scaling_method = kwargs.get('scaling_method', self.model_config['scaling_method']) # method for scaling the variables
        self.name_tgt_var   = kwargs.get('name_tgt_var',   self.model_config['name_tgt_var'])   # name of the target variable
        self.media_vars     = kwargs.get('media_vars',     self.model_config['media_vars'])     # media variables
        self.control_vars   = kwargs.get('control_vars',   self.model_config['control_vars'])   # control variables
        
        self.check_X_y(X, y)
        
        self._generate_and_preprocess_model_data(X, y)
        
        self.model = self.def_model(
            self.coords, 
            self.X_scaled, self.y_scaled, 
            self.groups, self.groups_idx, 
            self.fourier_features, 
            self.media_vars, 
            self.control_vars, 
            self.L_MAX
            )

    def _data_setter(
        self, X: pd.DataFrame, y: pd.DataFrame = None
    ):
        
        self.check_X_y(X, y)
        self._generate_and_preprocess_model_data(X, y.values if isinstance(y, pd.Series) else y)

        with self.model: 
            pm.set_data({"X_t":       self.X_scaled['t']})
            pm.set_data({"X_fourier": self.fourier_features})
            pm.set_data({"X_media":   self.X_scaled[self.media_vars]})
            pm.set_data({"X_control": self.X_scaled[self.control_vars]})
            pm.set_data({"Y":         self.y_scaled})
            
    @staticmethod
    def get_default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
       
        model_config: Dict = {
            'n_order': 2,
            'L_MAX': 8,
            'name_date_var': 'date_week',
            'name_group_var': 'group',
            'period': 365.25/7,
            'scaling_method': 'max',
            'name_tgt_var': 'y',
            "sigma_alpha_trend": 1.0,
            "lam_sigma_alpha_trend": 1.0,
            "sigma_beta_trend": 1.0,
            "lam_sigma_beta_trend": 1.0,
            "lower_k": 0.1,
            "upper_k": 4.0,
            "lam_mu_media_var": 1.0,
            "lam_sigma_media_var": 1.0,
            "lower_alpha_as_media_var": 1.0,
            "upper_alpha_as_media_var": 4.0,
            "lower_beta_as_media_var": 1.0,
            "upper_beta_as_media_var": 4.0,
            "lam_lam_media_var": 1.0,
            "mu_mu_control_var": 0.0,
            "sigma_mu_control_var": 1.0,
            "lam_sigma_control_var": 1.0,
            "lam_sigma_noise": 1.0,
        }
        return model_config

    @staticmethod
    def get_default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
              
        sampler_config: Dict = {
            "nuts_sampler": "numpyro",
            "draws": 1_000,
            # "chains": 5,
            # "target_accept": 0.95,
            # "random_seed": 42,
            # "return_inferencedata": True,
            # "idata_kwargs": {"log_likelihood": True},
        }
        return sampler_config

    @property
    def output_var(self):
        return "y_obs"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """
        pass

        pass

    def _generate_and_preprocess_model_data(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        
        self.n_order        = self.model_config['n_order']        # order of the fourier series
        self.L_MAX          = self.model_config['L_MAX']          # order of the adstock
        self.name_date_var  = self.model_config['name_date_var']  # name of the date variable
        self.name_group_var = self.model_config['name_group_var'] # name of the group variable
        self.period         = self.model_config['period']         # period of the seasonality
        self.scaling_method = self.model_config['scaling_method'] # method for scaling the variables
        self.name_tgt_var   = self.model_config['name_tgt_var']   # name of the target variable
        self.media_vars     = self.model_config['media_vars']     # name of the target variable
        self.control_vars   = self.model_config['control_vars']   # name of the target variable
        
        self.X_orig = X
        self.y_orig = y    
        
        self.X = X.copy()
        self.y = y.copy()
        self.groups_idx, self.groups = self.X[self.name_group_var].factorize(sort=True)
        
        # Creamos por grupo una columna para la tendencia
        for g in self.groups:
            size_g = sum(self.X[self.name_group_var] == g)
            self.X.loc[self.X[self.name_group_var] == g, 't'] = np.linspace(1, size_g, num = size_g) 

        # Creamos tembién por grupo las variables de Fourier para la estacionalidad  
        self.fourier_features = pd.DataFrame()
        for g in self.groups:
            fourier_features_g = make_fourier_features(
                self.X[self.X[self.name_group_var] == g], 
                self.name_date_var, 
                n_order=self.n_order, 
                period=self.period)
            self.fourier_features = pd.concat([self.fourier_features, fourier_features_g], axis=0)
        self.X = pd.concat([self.X, self.fourier_features], axis=1)
    
        # Escalado de las variables predictivas
        self.the_X_scaler = CustomGroupScaler(columns=['t'], group_columns=[self.name_group_var], method=self.scaling_method)
        self.X_scaled     = self.the_X_scaler.fit_transform(X)
    
        # Escalado de la variable respuesta
        self.the_y_scaler = CustomGroupScaler(columns=[self.name_tgt_var], group_columns=[self.name_group_var], method=self.scaling_method)
        self.y_scaled     = self.the_y_scaler.fit_transform(pd.DataFrame({self.name_tgt_var: y, self.name_group_var: self.X[self.name_group_var]}))
        self.y_scaled = self.y_scaled[self.name_tgt_var].values

        self.coords = {
            'group':            self.groups,
            'media_vars':       self.media_vars,
            'control_vars':     self.control_vars,
            "fourier_features": np.arange(2 * self.n_order),
            'obs':              self.X_scaled.index.to_numpy()
        }