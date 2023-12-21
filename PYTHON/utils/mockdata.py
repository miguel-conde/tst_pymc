######################################################################################
# https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_example.html
######################################################################################


# import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation

def generate_dates(start_date, end_date):
    
    # check if min_date is minor than max_date  
    if start_date > end_date:
        raise ValueError("start_date must be minor than end_date")  
    
    min_date = pd.to_datetime(start_date)
    max_date = pd.to_datetime(end_date)

    df = pd.DataFrame(
        data={"date_week": pd.date_range(start=min_date, end=max_date, freq="W-MON")}
    ).assign(
        year=lambda x: x["date_week"].dt.year,
        month=lambda x: x["date_week"].dt.month,
        dayofyear=lambda x: x["date_week"].dt.dayofyear,
    )
    
    n = df.shape[0]

    return n, df

def generate_media_costs_raw(
    df, n, rng,
    ):
    ## Generate Media Costs Data
    
    # 1 - Raw Signal
    # By design, x1 and x2 should resemble a typical paid social channel and a offline (e.g. TV) spend time series.
    
    # media data
    x1 = rng.uniform(low=0.0, high=1.0, size=n)
    df["x1"] = np.where(x1 > 0.9, x1, x1 / 2)
    
    x2 = rng.uniform(low=0.0, high=1.0, size=n)
    df["x2"] = np.where(x2 > 0.8, x2, 0)
    
    return df

def generate_media_costs_as_satur(
    df,
    alpha1, alpha2,
    lmax1, lmax2,
    lam1, lam2,
    ):
        
    # 2 - Effect signal
    # apply geometric adstock transformation

    df["x1_adstock"] = (
        geometric_adstock(x=df["x1"].to_numpy(), alpha=alpha1, l_max=lmax1, normalize=True)
        .eval()
        .flatten()
    )

    df["x2_adstock"] = (
        geometric_adstock(x=df["x2"].to_numpy(), alpha=alpha2, l_max=lmax2, normalize=True)
        .eval()
        .flatten()
    )
    
    # apply saturation transformation

    df["x1_adstock_saturated"] = logistic_saturation(
        x=df["x1_adstock"].to_numpy(), lam=lam1
    ).eval()

    df["x2_adstock_saturated"] = logistic_saturation(
        x=df["x2_adstock"].to_numpy(), lam=lam2
    ).eval()
    
    return df

def generate_media_costs(
    df, n, rng,
    alpha1, alpha2,
    lmax1, lmax2,
    lam1, lam2,
    ):
    ## Generate Media Costs Data
    
    # 1 - Raw Signal
    # By design, x1 and x2 should resemble a typical paid social channel and a offline (e.g. TV) spend time series.
    
    # media data
    # x1 = rng.uniform(low=0.0, high=1.0, size=n)
    # df["x1"] = np.where(x1 > 0.9, x1, x1 / 2)
    # 
    # x2 = rng.uniform(low=0.0, high=1.0, size=n)
    # df["x2"] = np.where(x2 > 0.8, x2, 0)
    
    df = generate_media_costs_raw(df, n, rng)
    
    # 2 - Effect signal
    # apply geometric adstock transformation

    # df["x1_adstock"] = (
    #     geometric_adstock(x=df["x1"].to_numpy(), alpha=alpha1, l_max=lmax1, normalize=True)
    #     .eval()
    #     .flatten()
    # )
# 
    # df["x2_adstock"] = (
    #     geometric_adstock(x=df["x2"].to_numpy(), alpha=alpha2, l_max=lmax2, normalize=True)
    #     .eval()
    #     .flatten()
    # )
    # 
    # # apply saturation transformation
# 
    # df["x1_adstock_saturated"] = logistic_saturation(
    #     x=df["x1_adstock"].to_numpy(), lam=lam1
    # ).eval()
# 
    # df["x2_adstock_saturated"] = logistic_saturation(
    #     x=df["x2_adstock"].to_numpy(), lam=lam2
    # ).eval()
    
    df = generate_media_costs_as_satur(
        df,
        alpha1, alpha2,
        lmax1, lmax2,
        lam1, lam2,
        )
    
    return df

def generate_trend_seasonality(df, n):
    ## Trend & Seasonal Components

    df["trend"] = (np.linspace(start=0.0, stop=50, num=n) + 10) ** (1 / 4) - 1

    df["cs"] = -np.sin(2 * 2 * np.pi * df["dayofyear"] / 365.5)
    df["cc"] = np.cos(1 * 2 * np.pi * df["dayofyear"] / 365.5)
    df["seasonality"] = 0.5 * (df["cs"] + df["cc"])
    
    return df

def generate_control_variables(df):
    ## Control variables

    df["event_1"] = (df["date_week"] == "2022-05-09").astype(float)
    df["event_2"] = (df["date_week"] == "2023-09-04").astype(float)
    
    return df

def generate_response(df, n, rng, 
                      intercept,
                      loc, scale,
                      amplitude,
                      beta_1, beta_2,
                      ):
    ## Target Variable

    df["intercept"] = intercept
    df["epsilon"] = rng.normal(loc=loc, scale=scale, size=n)

    df["y"] = amplitude * (
        df["intercept"]
        + df["trend"]
        + df["seasonality"]
        + 1.5 * df["event_1"]
        + 2.5 * df["event_2"]
        + beta_1 * df["x1_adstock_saturated"]
        + beta_2 * df["x2_adstock_saturated"]
        + df["epsilon"]
    )

    return df

def add_groups(df, feeder_market, segmento_cliente, canal, segmento_hotel):
    
    df['feeder_market'] = feeder_market
    df['segmento_cliente']= segmento_cliente
    df['canal'] = canal 
    df['segmento_hotel'] = segmento_hotel   
    
    return df

def generate_mock_data(
    start_date, end_date,       # Para las fechas
    alpha1:float, alpha2:float, # 0 - 1 Para adstock geométrico
    lmax1:int, lmax2:int,       # > 0 - Para el maximum lag effect de adstock geométrico
    lam1:float, lam2:float,     # > 0 - Para saturación logística
    intercept:float,            # Para el intercepto
    loc:float, scale:float,     # Para la distribución normal de los errores
    amplitude:float,            # > 0 # Para la amplitud de la señal
    beta_1:float, beta_2:float, # Para la contribución de cada señal
    feeder_market:str, segmento_cliente:str, canal:str, segmento_hotel:str, # Para los grupos
    seed_str = "mmm"
):
    
    seed: int = sum(map(ord, seed_str))
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    n, df = generate_dates(start_date, end_date)
    
    df = generate_media_costs(df, n, rng, 
                              alpha1=alpha1, alpha2=alpha2, 
                              lmax1=lmax1, lmax2=lmax2,
                              lam1=lam1, lam2=lam2)
    df = generate_trend_seasonality(df, n)
    df = generate_control_variables(df)
    df = generate_response(df, n, rng,
                           intercept=intercept,
                           loc = loc, scale = scale,
                           amplitude=amplitude,
                           beta_1=beta_1, beta_2=beta_2)
    
    df = add_groups(df, feeder_market, segmento_cliente, canal, segmento_hotel)
    
    return df


def generate_mock_data_groups(
    feeder_markets = ["Germany", "Italy", "Spain", "UK"],
    segmentos_cliente = ["B2B", "B2C"],
    canales = ["OTAs", "Web"],
    segmentos_hotel = ["business", "leisure", "luxury", "resort"],
    seed_str = "mmm"
    ):
    
    n_groups = len(feeder_markets) * len(segmentos_cliente) * len(canales) * len(segmentos_hotel)
    
    seed: int = sum(map(ord, seed_str))
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    
    alphas1    = rng.uniform(low=0.0, high=1.0, size=n_groups)
    alphas2    = rng.uniform(low=0.0, high=1.0, size=n_groups)
    lmax1      = rng.integers(low=1, high=4,  size=n_groups)
    lmax2      = rng.integers(low=6, high=16, size=n_groups)
    lam1       = rng.uniform(low=5.0, high=10.0, size=n_groups)
    lam2       = rng.uniform(low=0.0, high=5.0,  size=n_groups)
    intercepts = rng.uniform(low=1.0, high=3.0, size=n_groups)
    scales     = rng.uniform(low=0.2, high=0.3,  size=n_groups)
    betas1     = rng.uniform(low=1.0, high=5.0,  size=n_groups)
    betas2     = rng.uniform(low=1.0, high=5.0,  size=n_groups)
    
    iter = 0
    lst_dfs = []
    for feeder_market in feeder_markets:
        for segmento_cliente in segmentos_cliente:
            for canal in canales:
                for segmento_hotel in segmentos_hotel:
                    # Print progress including the number of the iteration over the number of groups
                    print(f"Generating mock data for {feeder_market}, {segmento_cliente}, {canal}, {segmento_hotel} - {iter+1}/{n_groups}")
                    
                    df = generate_mock_data(
                        start_date="2021-11-01",
                        end_date="2023-11-30",
                        alpha1=alphas1[iter],
                        alpha2=alphas2[iter], 
                        lmax1=lmax1[iter], 
                        lmax2=lmax2[iter],
                        lam1=lam1[iter],
                        lam2=lam2[iter],
                        intercept=intercepts[iter],
                        loc=0.0,
                        scale=scales[iter],
                        amplitude=1,
                        beta_1=betas1[iter],
                        beta_2=betas2[iter],
                        feeder_market=feeder_market,
                        segmento_cliente=segmento_cliente,
                        canal=canal,
                        segmento_hotel=segmento_hotel,
                        seed_str=feeder_market + segmento_cliente + canal + segmento_hotel,
                    )
                    iter += 1
                    df['group'] = iter
                    lst_dfs.append(df)
                    
    # Concatenate all dataframes
    df = pd.concat(lst_dfs)
    
    # Order by group, date_week, feeder_market, segmento_cliente, canal, segmento_hotel
    df = df.sort_values(by=['group', 'date_week', 'feeder_market', 'segmento_cliente', 'canal', 'segmento_hotel'])
    df.reset_index(drop=True, inplace=True)
    print("QUÉ PACHAAAAAAAA")
    
    pars = pd.DataFrame({
        'group': np.arange(1, n_groups+1),
        'alpha1': alphas1,
        'alpha2': alphas2,
        'lmax1': lmax1,
        'lmax2': lmax2,
        'lam1': lam1,
        'lam2': lam2,
        'intercept': intercepts,
        'scale': scales,
        'beta1': betas1,
        'beta2': betas2
    })
    
    return pars, df


def generate_mock_data_groups_2(
    feeder_markets = ["Germany", "Italy", "Spain", "UK"],
    segmentos_cliente = ["B2B", "B2C"],
    canales = ["OTAs", "Web"],
    segmentos_hotel = ["business", "leisure", "luxury", "resort"],
    seed_str = "mmm"
    ):
    """A diferencia de `generate_mock_data_groups()`, esta función hace que todos los grupos tengan las
    mismas variables de fechas, de medios y de control. 
    
    Args:
        feeder_markets (list, optional): _description_. Defaults to ["Germany", "Italy", "Spain", "UK"].
        segmentos_cliente (list, optional): _description_. Defaults to ["B2B", "B2C"].
        canales (list, optional): _description_. Defaults to ["OTAs", "Web"].
        segmentos_hotel (list, optional): _description_. Defaults to ["business", "leisure", "luxury", "resort"].
        seed_str (str, optional): _description_. Defaults to "mmm".

    Returns:
        _type_: _description_
    """
    
    n_groups = len(feeder_markets) * len(segmentos_cliente) * len(canales) * len(segmentos_hotel)
    
    seed: int = sum(map(ord, seed_str))
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    
    alphas1    = rng.uniform(low=0.0, high=1.0, size=n_groups)
    alphas2    = rng.uniform(low=0.0, high=1.0, size=n_groups)
    lmax1      = rng.integers(low=1, high=4,  size=n_groups)
    lmax2      = rng.integers(low=6, high=16, size=n_groups)
    lam1       = rng.uniform(low=5.0, high=10.0, size=n_groups)
    lam2       = rng.uniform(low=0.0, high=5.0,  size=n_groups)
    intercepts = rng.uniform(low=1.0, high=3.0, size=n_groups)
    scales     = rng.uniform(low=0.2, high=0.3,  size=n_groups)
    betas1     = rng.uniform(low=1.0, high=5.0,  size=n_groups)
    betas2     = rng.uniform(low=1.0, high=5.0,  size=n_groups)
    
    iter = 0
    lst_dfs = []
    for feeder_market in feeder_markets:
        # Cada feeder market tiene las mismas variables de fechas, de medios y de control
        n, df_feeder_mkt = generate_dates(start_date = "2021-11-01", end_date = "2023-11-30")
    
        df_feeder_mkt = generate_media_costs_raw(df_feeder_mkt, n, rng,)
        df_feeder_mkt = generate_control_variables(df_feeder_mkt)
        
        for segmento_cliente in segmentos_cliente:
            for canal in canales:
                for segmento_hotel in segmentos_hotel:
                    # Print progress including the number of the iteration over the number of groups
                    print(f"Generating mock data for {feeder_market}, {segmento_cliente}, {canal}, {segmento_hotel} - {iter+1}/{n_groups}")
                    
                    df = df_feeder_mkt.copy()
                    df = generate_media_costs_as_satur(df_feeder_mkt,
                                                       alpha1=alphas1[iter,], alpha2=alphas2[iter,], 
                                                       lmax1=lmax1[iter,], lmax2=lmax2[iter,],
                                                       lam1=lam1[iter,], lam2=lam2[iter,])
                    df = generate_trend_seasonality(df, n)
                    df = generate_response(df, n, rng,
                                           intercept=intercepts[iter,],
                                           loc = 0.0, scale = scales[iter,],
                                           amplitude=1.0,
                                           beta_1=betas1[iter,], beta_2=betas2[iter,])
    
                    df = add_groups(df, feeder_market, segmento_cliente, canal, segmento_hotel)
                    
                    iter += 1
                    df['group'] = iter
                    lst_dfs.append(df.copy())
    
    # Concatenate all dataframes
    df = pd.concat(lst_dfs)
    
    # Order by group, date_week, feeder_market, segmento_cliente, canal, segmento_hotel
    df = df.sort_values(by=['group', 'date_week', 'feeder_market', 'segmento_cliente', 'canal', 'segmento_hotel'])
    df.reset_index(drop=True, inplace=True)
    print("QUÉ PACHAAAAAAAA")
    
    pars = pd.DataFrame({
        'group': np.arange(1, n_groups+1),
        'alpha1': alphas1,
        'alpha2': alphas2,
        'lmax1': lmax1,
        'lmax2': lmax2,
        'lam1': lam1,
        'lam2': lam2,
        'intercept': intercepts,
        'scale': scales,
        'beta1': betas1,
        'beta2': betas2
    })
    
    return pars, df





############################################################################################################
# https://juanitorduz.github.io/multilevel_elasticities_single_sku/
############################################################################################################
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns

from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator, field_validator
from tqdm.notebook import tqdm

class Sku(BaseModel):
    id: int = Field(..., ge=0)
    prices: NDArray[np.float_]
    quantities: NDArray[np.float_]

    class Config:
        arbitrary_types_allowed = True

    @field_validator("prices", "quantities")
    def validate_gt_0(cls, value):
        if (value <= 0).any():
            raise ValueError("prices and quantities must be positive")
        return value

    @field_validator("prices", "quantities")
    def validate_size_gt_0(cls, value):
        if value.size == 0:
            raise ValueError("prices and quantities must have at least one element")
        return value

    @model_validator(mode="before")
    def validate_sizes(cls, values):
        if values["prices"].size != values["quantities"].size:
            raise ValueError("prices and quantities must have the same size")
        return values

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            data={
                "item_id": self.id,
                "price": self.prices,
                "quantities": self.quantities,
                "time_step": np.arange(self.prices.size)[::-1],
            }
        )


class Store(BaseModel):
    id: int = Field(..., ge=0)
    items: list[Sku] = Field(..., min_items=1)

    @field_validator("items")
    def validate_item_ids(cls, value):
        if len({item.id for item in value}) != len(value):
            raise ValueError("items must have unique ids")
        return value

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.concat([item.to_dataframe() for item in self.items], axis=0)
        df["store_id"] = self.id
        df["region_store_id"] = f"r-{self.id}_s-" + df["store_id"].astype(str)
        return df.reset_index(drop=True)


class Region(BaseModel):
    id: int = Field(..., ge=0)
    stores: list[Store] = Field(..., min_items=1)
    median_income: float = Field(..., gt=0)

    @field_validator("stores")
    def validate_store_ids(cls, value):
        if len({store.id for store in value}) != len(value):
            raise ValueError("stores must have unique ids")
        return value

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.concat([store.to_dataframe() for store in self.stores], axis=0)
        df["region_id"] = self.id
        df["median_income"] = self.median_income
        return df.reset_index(drop=True)


class Market(BaseModel):
    regions: list[Region] = Field(..., min_items=1)

    @field_validator("regions")
    def validate_region_ids(cls, value):
        if len({region.id for region in value}) != len(value):
            raise ValueError("regions must have unique ids")
        return value

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.concat([region.to_dataframe() for region in self.regions], axis=0)
        return df.reset_index(drop=True).assign(
            log_price=lambda x: np.log(x["price"]),
            log_quantities=lambda x: np.log(x["quantities"]),
            region_id=lambda x: x["region_id"].astype("category"),
            region_store_id=lambda x: x["region_store_id"].astype("category"),
        )

class LinearRegressionConfig(BaseModel):
    intercept: float
    slope: float
    sigma: float = Field(..., gt=0)
    
class MultiLevelElasticitiesDataGenerator(BaseModel):
    rng: np.random.Generator
    n_regions: int = Field(..., gt=0)
    time_range_mu: float = Field(..., gt=0)
    time_range_sigma: float = Field(..., gt=0)
    n_stores_per_region_mu: float = Field(..., gt=0)
    n_stores_per_region_sigma: float = Field(..., gt=0)
    median_income_per_region_mu: float = Field(..., gt=0)
    median_income_per_region_sigma: float = Field(..., gt=0)
    intercepts_lr_config: LinearRegressionConfig
    slopes_lr_config: LinearRegressionConfig
    price_mu: float = Field(..., gt=0)
    price_sigma: float = Field(..., gt=0)
    epsilon: float = Field(..., gt=0)

    class Config:
        arbitrary_types_allowed = True

    def get_n_stores_per_region_draws(self) -> NDArray:
        n_stores_per_region_dist = pm.NegativeBinomial.dist(
            mu=self.n_stores_per_region_mu, alpha=self.n_stores_per_region_sigma
        )
        n_stores_per_region_draws = pm.draw(
            n_stores_per_region_dist, draws=self.n_regions, random_seed=self.rng
        )
        return n_stores_per_region_draws + 2

    def get_median_income_per_region_draws(self) -> NDArray:
        median_income_per_region_dist = pm.Gamma.dist(
            mu=self.median_income_per_region_mu,
            sigma=self.median_income_per_region_sigma,
        )
        median_income_per_region_draws = pm.draw(
            median_income_per_region_dist, draws=self.n_regions, random_seed=self.rng
        )
        return median_income_per_region_draws + 1

    def get_store_time_range(self) -> int:
        time_range_dist = pm.NegativeBinomial.dist(
            mu=self.time_range_mu, alpha=self.time_range_sigma
        )
        time_range_samples = pm.draw(
            vars=time_range_dist, draws=1, random_seed=self.rng
        ).item()
        return time_range_samples + 2

    def get_alpha_j_samples(
        self, median_income_per_region: float, store_time_range: int
    ) -> NDArray:
        alpha_j_dist = pm.Normal.dist(
            mu=self.intercepts_lr_config.intercept
            + self.intercepts_lr_config.slope * median_income_per_region,
            sigma=self.intercepts_lr_config.sigma,
        )
        return pm.draw(alpha_j_dist, draws=store_time_range, random_seed=self.rng)

    def get_beta_j_samples(
        self, median_income_per_region: float, store_time_range: int
    ) -> NDArray:
        beta_j_dist = pm.Normal.dist(
            mu=self.slopes_lr_config.intercept
            + self.slopes_lr_config.slope * median_income_per_region,
            sigma=self.slopes_lr_config.sigma,
        )
        return pm.draw(beta_j_dist, draws=store_time_range, random_seed=self.rng)

    def get_prices_samples(self, store_time_range: int) -> NDArray:
        price_dist = pm.Gamma.dist(
            mu=self.price_mu,
            sigma=self.price_sigma,
        )
        return pm.draw(price_dist, draws=store_time_range, random_seed=self.rng)

    def get_quantities_samples(
        self, alpha_j_samples, beta_j_samples, prices_samples
    ) -> NDArray:
        log_quantities_dist = pm.Normal.dist(
            mu=alpha_j_samples + beta_j_samples * np.log(prices_samples),
            sigma=self.epsilon,
        )
        log_quantities_samples = pm.draw(
            log_quantities_dist, draws=1, random_seed=self.rng
        )
        return np.exp(log_quantities_samples)

    def create_store(self, id: int, median_income_per_region: float) -> Store:
        store_time_range = self.get_store_time_range()
        alpha_j_samples = self.get_alpha_j_samples(
            median_income_per_region=median_income_per_region,
            store_time_range=store_time_range,
        )
        beta_j_samples = self.get_beta_j_samples(
            median_income_per_region=median_income_per_region,
            store_time_range=store_time_range,
        )
        prices_samples = self.get_prices_samples(store_time_range=store_time_range)
        quantities_samples = self.get_quantities_samples(
            alpha_j_samples=alpha_j_samples,
            beta_j_samples=beta_j_samples,
            prices_samples=prices_samples,
        )
        return Store(
            id=id,
            items=[
                Sku(id=0, prices=prices_samples, quantities=quantities_samples)
            ],  # <- we only have one sku (id = 0)
        )

    def create_region(
        self, id: int, n_stores_per_region: int, median_income_per_region: float
    ) -> Region:
        stores: list[Store] = [
            self.create_store(id=i, median_income_per_region=median_income_per_region)
            for i in range(n_stores_per_region)
        ]
        return Region(id=id, stores=stores, median_income=median_income_per_region)

    def run(self) -> Market:
        n_stores_per_region_draws = self.get_n_stores_per_region_draws()
        median_income_per_region_draws = self.get_median_income_per_region_draws()

        regions: list[Region] = [
            self.create_region(
                id=j,
                n_stores_per_region=n_stores_per_region_draws[j],
                median_income_per_region=median_income_per_region_draws[j],
            )
            for j in tqdm(range(self.n_regions))
        ]

        return Market(regions=regions)
    


####################################################################################################
# STRUCTURAL TIME SERIES MODELS
####################################################################################################

import numpy as np
import pandas as pd

def stochastic_trend(
        rng, 
        N = 100, 
        sigma_local_trend = 1, 
        local_trend_0 = 1, 
        sigma_local_level = 1, 
        local_level_0 = 100
        ):
    """Simulate a stochastic trend 
    
    This trend consists of:
    
    - A local trend (a random walk)
    - A local level. Changes in local level are the local trend plus noise.
    
    The trend is the local level.

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
    """simulate a deterministic trend
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

