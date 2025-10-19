# !pip install xgboost
# !pip install lightgbm
# !pip install seaborn --upgrade
# !pip install factor_analyzer

-m pip install -r requirements.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import LeaveOneOut, cross_val_score

#plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
#sns
sns.set(rc={"figure.dpi":300, "savefig.dpi":300})
#%%
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

model_names = ["linear regression", "ridge", "lasso", "logistic regression", 
                "support vector machine", "decision tree", "random forest",
                "stochastic gradient descent", "gradient boosting", 
                "xgboost", "gaussian process", "lgbm"]

categorical_model = {"logistic regression": LogisticRegression(max_iter=1000),
                          "support vector machine": SVC(kernel='rbf', C=1.0),
                          "decision tree": DecisionTreeClassifier(max_depth=5),
                          "random forest": RandomForestClassifier(n_estimators=100),
                          "stochastic gradient descent": SGDClassifier(max_iter=1000, tol=1e-3),
                          "gradient boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),
                          "xgboost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
                          "gaussian process": GaussianProcessClassifier(kernel=RBF()),
                          "lgbm": LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, verbosity=-1),
                          }

continuous_model = {"linear regression": LinearRegression(),
                    "ridge": Ridge(),
                    "lasso": Lasso(),
                    "decision tree": DecisionTreeRegressor(max_depth=5),
                    "random forest": RandomForestRegressor(n_estimators=100),
                    "stochastic gradient descent": SGDRegressor(max_iter=1000, tol=1e-3),
                    "gradient boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
                    "xgboost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
                    "gaussian process": GaussianProcessRegressor(kernel=RBF()),
                    "lgbm": LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, verbosity=-1)
                    }

#%%
class Preprocessing:
    
    def __init__(self, df):
        self.df = df
        self.scalers = {}
        self.encoders = {}
    
    def get(self):
        return self.df
    
    def pop_target(self, col):
        return self.df.pop(col)
    
    def reset_index(self):
        self.df.reset_index(drop=True, inplace=True)
        
    def info(self):
        return self.df.info()
        
    def describe(self):
        return self.df.describe()
    
    def columns(self):
        return self.df.columns
    
    def check(self, n=5):
        return self.df.head(n)
    
    def check_missing(self):
        return self.df.isnull().sum()
    
    def preprocessing_outlier(self, col, kind="IQR", replace=False, fig=False):
        """
        kind: ["IQR", "zscore"]
        """
        
        if kind == "IQR":
            # IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)][col]
        elif kind == "zscore":
            z = np.abs(stats.zscore(self.df[col]))
            threshold = 3
            outliers = self.df[z > threshold][col]
        else:
            raise

        print("===outlier===")
        print(outliers)

        if fig:
            plt.boxplot(self.df[col],
                patch_artist=True,
                boxprops=dict(facecolor="#90CAF9", color="#1565C0", linewidth=2),
                whiskerprops=dict(color="#1565C0", linewidth=2),
                capprops=dict(color="#1565C0", linewidth=2),
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker="o", markerfacecolor="orange", markersize=2, alpha=0.7)
                )
            plt.xticks([])
            plt.title(f"Outlier Check: {col}", fontsize=14, fontweight="bold")
            plt.show()
        
        if replace:
            self.df = self.df.drop(outliers.index)
    
    def preprocessing_replace(self, col, old, new):
        self.df[col].replace(old, new, inplace=True)
        print(f"{old} was replaced with {new}")
        
    def preprocessing_missing(self, col=None, kind="mean", new_col=False, replace_value=None, save=False):
        """
        col: column name
        
        kind: ["replace", "mean", "median", "mode", "ffill", "bfill", "linear_interpolate", "cubic_spline"]
        
        new_col: create new column name
        """
        if col is None:
            col = self.columns()

        if kind == "drop":
            result = self.df[col].dropna()
        elif kind == "replace":
            result = self.df[col].fillna(replace_value)
        elif kind == "mean":
            result = self.df[col].fillna(self.df[col].mean())
        elif kind == "median":
            result = self.df[col].fillna(self.df[col].median())
        elif kind == "mode":
            result = self.df[col].fillna(self.df[col].mode().iloc[0])
        elif kind == "ffill":
            result = self.df[col].fillna(method=kind)
        elif kind == "bfill":
            result = self.df[col].fillna(method=kind)
        elif kind == "linear_interpolate":
            result = self.df[col].interpolate(method="linear")
        elif kind == "cubic_spline":
            result = self.df[col].interpolate(method="spline", order=3)
        else:
            raise
        
        if save:
            self.df[col] = result
            
        return result
    
    def correlation(self, col=None, kind="pearson", fig=False):
        """
        kind: ["pearson", "spearman", "kendall"]
        """
        if col is None:
            col = self.columns()
        
        corr = self.df[col].corr(method=kind)
            
        if fig:
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                corr,
                annot=True,
                cmap="RdBu",
                center=0,
                fmt=".2f",
                vmax=1,
                vmin=-1
            )
            plt.title("Correlation Heatmap")
            plt.show()
            
        return corr
    
    def preprocessing_encoding(self, col, kind="label", is_test=False, save=False):
        """
        kind: ["label"]
        """
        if kind == "label":
            encoder = LabelEncoder()
            if is_test:
                encoder = self.encoders[col]
                encoded = encoder.transform(self.df[col])
            else:
                encoder.fit(self.df[col])
                encoded = encoder.transform(self.df[col])
                self.encoders[col] = encoder
        # elif kind == "onehot":
        #     encoder = OneHotEncoder(sparse_output=False)
        #     if is_test:
        #         encoder = self.encoders[col]
        #         encoded = encoder.transform(self.df[[col]])
        #     else:
        #         encoder.fit(self.df[[col]])
        #         encoded = encoder.transform(self.df[[col]])
        #         self.encoders[col] = encoder
        else:
            raise
        
        # save data
        if save:
            self.df[col] = encoded
        print("Encoding Complete")
        return encoded
        
    # def check_encoding_class(self, col):
    #     try:
    #         return self.encoding_class[col]
    #     except:
    #         print("Enpty")
            
    def preprocessing_scaling(self, col=None, kind="standard", is_test=False, save=False):
        """
        kind: ["standard", "minmax", "robust"]
        """
        if col is None:
            col = self.columns()
            
        if kind == "standard":
            scaler = StandardScaler()
        elif kind == "minmax":
            scaler = MinMaxScaler()
        elif kind == "robust":
            scaler = RobustScaler()
        else:
            raise

        if is_test:
            scaler = self.scalers[tuple(col)]
            scaled = scaler.transform(self.df[col])
        else:
            scaler.fit(self.df[col])
            scaled = scaler.transform(self.df[col])
            self.scalers[tuple(col)] = scaler

        # save data
        if save:
            self.df = pd.DataFrame(scaled, columns = col)
        print("Scaling Complete")
        return pd.DataFrame(scaled, columns = col)
    def cronbach_alpha(self, col=None):
        """
        Compute Cronbach"s Alpha
        """
        if col is None:
            col = self.columns()
            
        k = self.df[col].shape[1]
        item_var = self.df[col].var(axis=0, ddof=1)
        total_var = self.df[col].sum(axis=1).var(ddof=1)
        
        return (k / (k - 1)) * (1 - (item_var.sum() / total_var))
    
    def ttest_ind(self, group_col, value_col):
        groups = self.df[group_col].unique()
        
        if len(groups) != 2:
            raise
        
        group1 = self.df[self.df[group_col] == groups[0]][value_col]
        group2 = self.df[self.df[group_col] == groups[1]][value_col]
        
        
        t, p = stats.ttest_ind(group1, group2)
        return t, p
    
    def anova(self, group_col, value_col):
        """
        group_col: categorical
        
        """
        groups = [g[value_col].values for _, g in self.df.groupby(group_col)]
        
        f, p = stats.f_oneway(*groups)
        return f, p
    
    def bartlett_test(self, col=None):
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
        
        if col is None:
            col = self.columns()
        
        return calculate_bartlett_sphericity(self.df[col])
    
    def KMOtest(self, col=None):
        from factor_analyzer.factor_analyzer import calculate_kmo
        
        if col is None:
            col = self.columns()
        
        _, kmo_score = calculate_kmo(self.df[col])

        return kmo_score
    
    def fa(self, col=None, n_factors=None, method=None, fig=False, create=False):
        """
        col: column name
        
        n_factors: number of factors

        method: ["varimax", "promax", "oblimin", "oblimax", ]
        """
        from factor_analyzer import FactorAnalyzer
        
        if col is None:
            col = self.columns()
            
        if n_factors:
            fa = FactorAnalyzer(n_factors=n_factors, rotation=method)
        else:
            fa = FactorAnalyzer(n_factors=len(col), rotation=method)
        
        fa.fit(self.df[col])
        
        # Kaiser"s Rule
        ev, _ = fa.get_eigenvalues() # print(ev)
        ev_1 = [i for i in ev if i >= 1] # ev >= 1
        # print(ev_1)
        print(len(ev_1))
        fa = FactorAnalyzer(n_factors=len(ev_1), rotation=method)
        fa.fit(self.df[col])
        
        loadings = pd.DataFrame(fa.loadings_, index=col)
        variance = pd.DataFrame(fa.get_factor_variance(), index=["SS Loadings", "Proportion Var", "Cumulative Var"])
        
        if fig:
            plt.figure(figsize=(15,20))
            ax = sns.heatmap(loadings,
                        cmap="RdBu",
                        annot=True,
                        fmt=".2f",
                        vmax=1, vmin=-1)
            # plt.xticks([i for i in range(len(ev_1))], label=[f"{i+1} factor" for i in range(len(ev_1))])
            ax.set_xticks(np.arange(len(ev_1)) + 0.5)
            ax.set_xticklabels([f"factor {i}" for i in range(len(ev_1))], rotation=90, ha="right")

            plt.title("Factor Analysis", fontsize=20)
            plt.show()
        if create:
            if len(ev_1) > 1:
                loadings = loadings[abs(loadings) > 0.3].sort_values([i for i in range(len(ev_1))], ascending=False)
                loadings.columns = [f"factor {i}" for i in range(len(ev_1))]
                df_fa = pd.DataFrame([])
                for factor in [f"factor {i}" for i in range(len(ev_1))]:
                    df_temp = self.df[loadings[~loadings[factor].isna()].index] * loadings[factor].dropna()
                    df_fa = pd.concat([df_fa, df_temp.sum(axis=1)], axis=1)
                df_fa.columns = [f"factor {i}" for i in range(len(df_fa.columns))]
                
            else :
                loadings = loadings[abs(loadings) > 0.3][0]
                df_fa = self.df[loadings[~loadings.isna()].index] * loadings.dropna()
            return df_fa
        else:
            return loadings, variance

    def pca(self, col=None, n=None, fig=False, create=False):
        from sklearn.decomposition import PCA
        
        if col is None:
            col = self.columns()
        if n is None:
            n = len(col)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[col])
        pca = PCA(n_components=n)
        pca.fit(X_scaled)
        
        explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
        
        if fig:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(explained_variance_ratio) + 1),
                     # range(4+1, 40+1),
                     pca.explained_variance_ratio_[:],
                     marker="o",
                     linestyle="-",
                     color="k")
            # plt.vlines(n, 0, pca.explained_variance_ratio_[0], linestyle="--", color="r")
            plt.title("Scree Plot")
            plt.xlabel("Principal Component")
            plt.ylabel("Explained Variance Ratio")
            plt.grid(True)
            plt.show()

        if create:
            return pd.DataFrame(pca.transform(X_scaled),
                         columns=[f"PC{i+1}" for i in range(pca.n_components_)])
        else:
            return explained_variance_ratio

#%%
class DASML:
    """
    target_type: ["categorical", "continuous"]
    """
    def __init__(self,
                 X,
                 y,
                 target_type,
                 random_state=42,
                 ):
        
        self.X = X
        self.y = y
        self.target_type = target_type
        self.random_state = random_state
        self.model = None
        self.model_name = None
        
    def hold_out(self, test_size=0.2, stratify=None):
        if stratify is None and self.target_type == "categorical": 
            stratify = self.y
            
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                            test_size=test_size,
                                                            random_state = self.random_state,
                                                            stratify=stratify,
                                                            )
        return X_train, X_test, y_train, y_test
    
    def clear(self):
        self.model = None
        self.model_name = None
    
    def set_model(self, model_name):
        """
        model_name: ["linear regression", "ridge", "lasso", "logistic regression", 
                      "support vector machine", "decision tree", "random forest",
                      "stochastic gradient descent", "gradient boosting", 
                      "xgboost", "gaussian process",  "lgbm"]
        """
        if model_name not in model_names:
            raise
            
        if self.target_type == "categorical":
            self.model = categorical_model[model_name]
        elif self.target_type == "continuous":
            self.model = continuous_model[model_name]
        else:
            raise
        
        if "random_state" in self.model.get_params():
            self.model.set_params(random_state=self.random_state)
        
        self.model_name = model_name
        
    def get_params(self):
        if self.model is None:
            raise
        return self.model.get_params()
    
    def set_params(self, **params):
        if self.model is None:
            raise
        self.model.set_params(**params)
        
    def modeling(self, X_train, y_train):
        if self.model is None:
            raise
        
        self.model.fit(X_train, y_train)
        
        return "modeling complete"
    
    def predict(self, X_test):
        if self.model is None:
            raise 
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        if self.model is None:
            raise 
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        else:
            pass

    def evaluation(self, y_test, y_pred):
        """
        categorical: accuracy, precision, recall, f1score
        continuous: mse, mae, r2score
        """
        if self.target_type == "categorical":
            return {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
            
        elif self.target_type == "continuous":
            return {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2score": float(r2_score(y_test, y_pred))
            }
        else:
            raise
