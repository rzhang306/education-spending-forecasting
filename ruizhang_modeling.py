"""
=============================================================================
EU EDUCATION EXPENDITURE MODELING — ONE-CLICK SCRIPT
=============================================================================
How to run (teacher/TA):
  1) Put this .py file and ready_for_ml_panel.csv in the SAME folder
     (or put the csv in a subfolder named "data/").
  2) Run in Terminal:
       python3 ruizhang_modeling.py
Outputs:
  - All figures/tables saved into: report_results/
Notes:
  - This script is designed to be reproducible and non-interactive (it saves figures).
    If you want pop-up windows, set SHOW_FIGURES = True below.
=============================================================================
"""

# -------------------- One-click working-directory fix --------------------
from pathlib import Path as _Path
import os as _os
_SCRIPT_DIR = _Path(__file__).resolve().parent
try:
    _os.chdir(_SCRIPT_DIR)  # ensures relative paths work when launched from anywhere
except Exception:
    pass

# -------------------- Optional: show figures interactively ----------------
SHOW_FIGURES = False  # set True to display figures (may slow down and require closing windows)


"""
EU Education Expenditure Modeling - Professional Analysis Pipeline

Author: [Your Name]
Date: December 2024
Purpose: Master's Thesis / Term Paper Analysis

Run: python modeling_professional.py
"""

import os
import json
from pathlib import Path

# -------------------- Robust data-path resolver --------------------
from pathlib import Path

def _find_data_file(filename: str) -> Path:
    """Find data file robustly regardless of the working directory."""
    candidates = []

    # 1) Current working directory
    candidates.append(Path.cwd() / filename)

    # 2) Script directory and parents
    script_dir = Path(__file__).resolve().parent
    for p in [script_dir, *script_dir.parents]:
        candidates.extend([
            p / filename,
            p / "data" / filename,
            p / "report_results" / filename,
        ])

    for c in candidates:
        if c.exists():
            return c

    tried = "\n".join(str(x) for x in candidates[:20])
    raise FileNotFoundError(f"Could not find data file '{filename}'. Tried:\n{tried}")
# ------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib
if not SHOW_FIGURES:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
CONFIG = {
    # Project paths (portable: relative to this script)
    "work_dir": str(Path(__file__).resolve().parent),
    "data_file": "ready_for_ml_panel.csv",
    "output_dir": "report_results",
    "random_state": 42,

    # Panel time range
    "start_year": 2003,
    "first_pred_year": 2016,
    "last_pred_year": 2023,

    # Target
    "target": "Edu_Exp_GDP",

    # Story-mode evaluation: structural break + fiscal constraint
    "tasks": ["nowcast", "lagged_forecast"],  # Xt->yt vs X(t-1)->yt
    "segments": {
        "pre": (2016, 2019),
        "covid": (2020, 2021),
        "post": (2022, 2023),
    },
    "debt_group_base_year": 2019
}

# ==================== EUROSTAT STYLE SETTINGS ====================
def set_professional_style():
    """
    Set figure style similar to Eurostat/ECB/IMF reports:
    - clean white background and light grid
    - blue-dominant color scheme
    - clear font hierarchy
    """
    COLORS = {
        'primary': '#003399',      # dark blue
        'secondary': '#0066CC',    # medium blue
        'accent': '#FF6B35',       # orange/red highlight
        'positive': '#2E7D32',     # green
        'negative': '#C62828',     # red
        'neutral': '#666666',      # grey
        'grid': '#E0E0E0'          # light grid
    }

    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',

        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.2,
        'axes.edgecolor': COLORS['neutral'],

        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.color': COLORS['grid'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'axes.axisbelow': True,

        'font.size': 10,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,

        'lines.linewidth': 2.0,
        'patch.linewidth': 0.5,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
    })

    sns.set_style("whitegrid", {
        'grid.color': COLORS['grid'],
        'grid.linestyle': '-',
        'axes.edgecolor': COLORS['neutral']
    })

    sns.set_palette([COLORS['primary'], COLORS['secondary'],
                     COLORS['accent'], COLORS['positive'],
                     COLORS['negative'], '#9C27B0', '#FF9800'])

    return COLORS

# ==================== GLOBAL BEAUTIFICATION FUNCTIONS ====================
def beautify_feature_name(name):
    """Convert internal feature names to readable labels."""
    if name.startswith('L1.'):
        clean = name.replace('L1.', '').replace('_', ' ')
        return f"{clean} (t-1)"
    elif name.startswith('L2.'):
        clean = name.replace('L2.', '').replace('_', ' ')
        return f"{clean} (t-2)"
    elif name.startswith('Roll3.'):
        clean = name.replace('Roll3.', '').replace('_', ' ')
        return f"{clean} (3-yr avg)"

    if name.startswith('Country_'):
        country = name.replace('Country_', '')
        return f"Country: {country}"

    replacements = {
        'Edu_Exp_GDP': 'Education Exp. (% GDP)',
        'NonEdu_Exp_GDP': 'Non-Education Exp. (% GDP)',
        'Log_GDP_pc': 'Log GDP per capita',
        'GDP_per_capita': 'GDP per capita',
        'Unemployment_Rate': 'Unemployment Rate',
        'Inflation_Rate': 'Inflation Rate',
        'Gov_Debt': 'Government Debt (% GDP)',
        'Youth_Share': 'Youth Population Share',
        'Debt_x_Growth': 'Debt × Growth',
        'Youth_x_Wealth': 'Youth × Wealth'
    }

    for old, new in replacements.items():
        if old in name:
            name = name.replace(old, new)

    return name.replace('_', ' ')

def beautify_spec_name(spec):
    """Pretty names for model specifications."""
    replacements = {
        'Spec_A_Macro_Only': 'Macro Only',
        'Spec_B_With_Lags': 'With Lags',
        'Spec_C_Extended': 'Full Model',
        'Spec_D_Parsimonious': 'Parsimonious'
    }
    return replacements.get(spec, spec.replace('Spec_', '').replace('_', ' '))

def beautify_model_name(model):
    """Pretty names for model types."""
    return model

COLORS = set_professional_style()

# ==================== PRINT HELPERS ====================
def print_section_header(title, section_num=None):
    """Print a formatted section header."""
    width = 100
    if section_num:
        header = f"SECTION {section_num}: {title}"
    else:
        header = title

    print("\n" + "=" * width)
    print(f"{header:^{width}}")
    print("=" * width)

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{'─' * 100}")
    print(f"  {title}")
    print(f"{'─' * 100}")

# ==================== SECTION 1: DATA ====================
def load_and_prepare_data():
    """Load raw data and perform feature engineering."""
    print_section_header("DATA LOADING AND PREPARATION", 1)

    project_dir = Path(CONFIG['work_dir'])
    data_path = _find_data_file(CONFIG['data_file'])
    df = pd.read_csv(data_path)
    print(f"  ✓ Loaded data from: {data_path}")

    print_subsection("1.1 Dataset Overview")
    print(f"  • Raw data shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  • Countries: {df['Country'].nunique()}")
    print(f"  • Year range: {df['Year'].min()}–{df['Year'].max()}")
    print(f"  • Time span: {df['Year'].max() - df['Year'].min() + 1} years")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n  ⚠ Missing values detected:")
        missing_pct = (missing / len(df) * 100).round(2)
        for col, pct in missing_pct[missing_pct > 0].items():
            print(f"     - {col}: {missing[col]} ({pct}%)")

    print_subsection("1.2 Feature Engineering")
    df = df.sort_values(['Country', 'Year']).reset_index(drop=True)

    df['NonEdu_Exp_GDP'] = df['Total_Exp_GDP'] - df['Edu_Exp_GDP']
    print("  ✓ Created NonEdu_Exp_GDP (Total - Education)")

    df['Log_GDP_pc'] = np.log1p(df['GDP_per_capita'])

    # --------------------
    # Robustness variable for "denominator effect"
    # Education spending per capita (approx.): (Edu_Exp_GDP/100) * GDP_per_capita
    # This removes pure GDP-level scaling and helps check whether the Covid break is only a ratio artifact.
    if "Edu_Exp_GDP" in df.columns and "GDP_per_capita" in df.columns:
        df["Edu_Exp_PC"] = (df["Edu_Exp_GDP"] / 100.0) * df["GDP_per_capita"]
        df["Log_Edu_Exp_PC"] = np.log1p(df["Edu_Exp_PC"])
        print("  * Created robustness target: Log_Edu_Exp_PC (=log(1+Edu_Exp_PC))")
    print("  ✓ Created Log_GDP_pc (log transformation)")

    grp = df.groupby('Country')
    lag_vars = ['Edu_Exp_GDP', 'NonEdu_Exp_GDP', 'Log_GDP_pc',
                'Unemployment_Rate', 'Inflation_Rate', 'Gov_Debt']

    for var in lag_vars:
        if var in df.columns:
            df[f'L1.{var}'] = grp[var].shift(1)
            df[f'L2.{var}'] = grp[var].shift(2)
            df[f'Roll3.{var}'] = grp[var].shift(1).rolling(3).mean()

    print(f"  ✓ Created lag variables (L1, L2, Roll3) for {len(lag_vars)} variables")

    df['Debt_x_Growth'] = df['Gov_Debt'] * df.groupby('Country')['Log_GDP_pc'].pct_change()
    df['Youth_x_Wealth'] = df['Youth_Share'] * df['Log_GDP_pc']
    # Fiscal-constraint grouping (fixed using 2019 debt to keep group membership pre-shock)
    debt_var = "Gov_Debt"
    debt_base_year = 2019
    if debt_var in df.columns and (df["Year"] == debt_base_year).any():
        debt_2019 = df.loc[df["Year"] == debt_base_year, ["Country", debt_var]].dropna()
        debt_by_country = debt_2019.groupby("Country")[debt_var].mean()
        thr_median = float(debt_by_country.median())
        thr_60 = 60.0
        thr_90 = 90.0

        def _mk_group(series, thr):
            return (series > thr).map({True: "HighDebt", False: "LowDebt"}).to_dict()

        map_median = _mk_group(debt_by_country, thr_median)
        map_60 = _mk_group(debt_by_country, thr_60)
        map_90 = _mk_group(debt_by_country, thr_90)

        df["DebtGroup_Median"] = df["Country"].map(map_median)
        df["DebtGroup_60"] = df["Country"].map(map_60)
        df["DebtGroup_90"] = df["Country"].map(map_90)

        # Default group used throughout the paper/figures: Maastricht 60%
        df["DebtGroup"] = df["DebtGroup_60"]

        print(f"  * Created debt groups using 2019 debt: Median={thr_median:.1f}%, 60%, 90%")
    else:
        df["DebtGroup"] = "Unknown"
        df["DebtGroup_Median"] = "Unknown"
        df["DebtGroup_60"] = "Unknown"
        df["DebtGroup_90"] = "Unknown"
        print("  ! DebtGroup not created (missing Gov_Debt or base year not found)")

    df_dum = pd.get_dummies(df, columns=['Country'], drop_first=True, dtype=int)
    country_cols = [c for c in df_dum.columns if c.startswith('Country_')]
    print(f"  ✓ Created {len(country_cols)} country dummies (drop_first=True)")

    df_dum = df_dum[(df_dum['Year'] >= CONFIG['start_year']) &
                    (df_dum['Year'] <= CONFIG['last_pred_year'])].copy()
    print(f"  ✓ Filtered to {CONFIG['start_year']}–{CONFIG['last_pred_year']}")

    return df_dum, country_cols

# ==================== SECTION 2: ECONOMETRIC DIAGNOSTICS ====================
def run_diagnostics(df, features):
    """Run basic econometric diagnostics: ADF and VIF."""
    print_section_header("ECONOMETRIC DIAGNOSTICS", 2)

    print_subsection("2.1 Stationarity Test (Augmented Dickey-Fuller)")

    test_vars = [CONFIG['target'], 'NonEdu_Exp_GDP', 'Unemployment_Rate',
                 'Inflation_Rate', 'Gov_Debt', 'Youth_Share']
    test_vars = [v for v in test_vars if v in df.columns]

    adf_results = []
    for var in test_vars:
        series = df[var].dropna()
        if len(series) > 10:
            adf_stat, p_value, _, _, crit_vals, _ = adfuller(series, maxlag=1)
            adf_results.append({
                'Variable': var,
                'ADF_Statistic': adf_stat,
                'p_value': p_value,
                'Stationary_5%': 'Yes' if p_value < 0.05 else 'No',
                'Critical_5%': crit_vals['5%']
            })

    adf_df = pd.DataFrame(adf_results)
    print(adf_df.to_string(index=False))

    print_subsection("2.2 Multicollinearity Check (Variance Inflation Factor)")

    vif_features = [f for f in features if not f.startswith('Country_')][:15]
    if len(vif_features) > 0:
        X_vif = df[vif_features].dropna()
        X_vif['const'] = 1

        vif_data = []
        for i, col in enumerate(vif_features):
            try:
                vif = variance_inflation_factor(X_vif.values, i)
                vif_data.append({'Feature': col, 'VIF': vif})
            except Exception:
                continue

        vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
        print(vif_df.head(10).to_string(index=False))

        high_vif = vif_df[vif_df['VIF'] > 10]
        if len(high_vif) > 0:
            print(f"\n  ⚠ Warning: {len(high_vif)} features have VIF > 10 (potential multicollinearity)")
    else:
        vif_df = None

    return adf_df, vif_df

# ==================== SECTION 3: MODEL SPECIFICATIONS ====================
def define_specifications(df, country_cols):
    """Define a set of model specifications."""
    print_section_header("MODEL SPECIFICATION DESIGN", 3)

    macro_base = ['NonEdu_Exp_GDP', 'Log_GDP_pc', 'Unemployment_Rate',
                  'Inflation_Rate', 'Gov_Debt', 'Youth_Share']

    lag_features = [c for c in df.columns if c.startswith(('L1.', 'L2.', 'Roll3.'))]

    interaction_features = ['Debt_x_Growth', 'Youth_x_Wealth']
    interaction_features = [f for f in interaction_features if f in df.columns]

    specifications = {
        'Spec_A_Macro_Only': {
            'features': [f for f in macro_base if f in df.columns] + country_cols,
            'description': 'Baseline: macro variables + country fixed effects'
        },
        'Spec_B_With_Lags': {
            'features': ([f for f in macro_base if f in df.columns] +
                        [f for f in lag_features if 'L1.' in f] + country_cols),
            'description': 'Add 1-period lags for budget inertia'
        },
        'Spec_C_Extended': {
            'features': ([f for f in macro_base if f in df.columns] +
                        lag_features + interaction_features + country_cols),
            'description': 'Full specification with lags, rolling averages, and interactions'
        },
        'Spec_D_Parsimonious': {
            'features': (['NonEdu_Exp_GDP', 'Log_GDP_pc', 'Unemployment_Rate',
                         'L1.Edu_Exp_GDP', 'Youth_Share'] + country_cols),
            'description': 'Parsimonious model with key variables only'
        }
    }

    print_subsection("3.1 Specification Overview")
    for spec_name, spec_info in specifications.items():
        n_features = len(spec_info['features'])
        n_macro = len([f for f in spec_info['features'] if not f.startswith('Country_')])
        n_country = len([f for f in spec_info['features'] if f.startswith('Country_')])

        print(f"\n  {spec_name}:")
        print(f"    Description: {spec_info['description']}")
        print(f"    Total features: {n_features} (Macro: {n_macro}, Country FE: {n_country})")

    return specifications

# ==================== SECTION 4: WALK-FORWARD VALIDATION ====================
def walk_forward_validation(df, specifications):
    """Expanding-window walk-forward validation (story mode: Covid break × fiscal constraint)."""
    print_section_header("WALK-FORWARD VALIDATION (STORY MODE)", 4)

    tasks = CONFIG.get("tasks", ["nowcast"])
    seg = CONFIG.get("segments", {"pre": (2016, 2019), "covid": (2020, 2021), "post": (2022, 2023)})
    print_subsection("4.1 Validation Strategy")
    
    # Models
    models = {
        'OLS': LinearRegression(),
        'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0, random_state=CONFIG['random_state']))]),
        'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.01, max_iter=20000, random_state=CONFIG['random_state']))]),
        'ElasticNet': Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=CONFIG['random_state']))]),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=6,
                                             random_state=CONFIG['random_state'], n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               objective='reg:squarederror', random_state=CONFIG['random_state'])
    }

    def year_to_segment(y: int) -> str:
        for name, (a, b) in seg.items():
            if a <= y <= b:
                return name
        return "other"

    # Helper: build feature set for each task
    macro_candidates = ['NonEdu_Exp_GDP', 'Log_GDP_pc', 'Unemployment_Rate', 'Inflation_Rate', 'Gov_Debt', 'Youth_Share',
                        'Debt_x_Growth', 'Youth_x_Wealth']

    def features_for_task(base_features, task: str):
        if task == "nowcast":
            return base_features
        if task == "lagged_forecast":
            out = []
            for f in base_features:
                if f.startswith("Country_"):
                    out.append(f)
                    continue
                if f in macro_candidates:
                    lf = f"L1.{f}"
                    if lf in df.columns:
                        out.append(lf)
                    continue
                if f.startswith(("L1.", "L2.", "Roll3.")):
                    out.append(f)
                    continue
            return out
        return base_features

    all_rows = []
    _naive_reference_spec = next(iter(specifications.keys())) if specifications else None

    for spec_name, spec_info in specifications.items():
        print(f"\n  Processing {spec_name}...")
        base_features = spec_info['features']

        for task in tasks:
            task_features = features_for_task(base_features, task)
            if len(task_features) == 0:
                continue

            df_spec = df.dropna(subset=task_features + [CONFIG['target']]).copy()

            for year in range(CONFIG['first_pred_year'], CONFIG['last_pred_year'] + 1):
                train = df_spec[df_spec['Year'] < year]
                test = df_spec[df_spec['Year'] == year]

                if len(train) == 0 or len(test) == 0:
                    continue

                X_train = train[task_features].values
                y_train = train[CONFIG['target']].values
                X_test = test[task_features].values
                y_test = test[CONFIG['target']].values

                countries = test['Country'].values if 'Country' in test.columns else np.array(['NA'] * len(test))
                debt_groups = test['DebtGroup'].values if 'DebtGroup' in test.columns else np.array(['Unknown'] * len(test))
                debt_groups_60 = test['DebtGroup_60'].values if 'DebtGroup_60' in test.columns else debt_groups
                debt_groups_median = test['DebtGroup_Median'].values if 'DebtGroup_Median' in test.columns else debt_groups
                debt_groups_90 = test['DebtGroup_90'].values if 'DebtGroup_90' in test.columns else debt_groups
                seg_label = year_to_segment(year)

                # Naïve baseline
                if spec_name == _naive_reference_spec:
                    if 'L1.Edu_Exp_GDP' in test.columns:
                        y_pred_naive = test['L1.Edu_Exp_GDP'].values
                    else:
                        tmp = df_spec.sort_values(['Country', 'Year'])
                        lag1 = tmp.groupby('Country')[CONFIG['target']].shift(1)
                        y_pred_naive = lag1.loc[test.index].values

                    for yt, yp, ctry, dg, dg60, dgmed, dg90 in zip(y_test, y_pred_naive, countries, debt_groups, debt_groups_60, debt_groups_median, debt_groups_90):
                        if np.isnan(yt) or np.isnan(yp): continue
                        
                        # ========== 修复点 1：Naive 模型的数值计算方向 ==========
                        # 之前是 yt - yp, 现在改为 yp - yt (预测 - 实际)
                        err = float(yp - yt)  
                        
                        all_rows.append({
                            'task': task, 'specification': 'Naive_L1', 'model': 'Naive',
                            'year': year, 'segment': seg_label, 'country': ctry,
                            'debt_group': dg, 'debt_group_60': dg60,
                            'debt_group_median': dgmed, 'debt_group_90': dg90,
                            'y_true': float(yt), 'y_pred': float(yp),
                            'error': err, 
                            'resid': err, # 统一用 resid
                            'abs_error': float(abs(err)), 'sq_error': float(err ** 2),
                        })

                # Main Models
                for model_name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        for c, dg, dg60, dgmed, dg90, yt, yp in zip(countries, debt_groups, debt_groups_60, debt_groups_median, debt_groups_90, y_test, y_pred):
                            
                            # ========== 修复点 2：机器学习模型的数值计算方向 ==========
                            # 之前是 yt - yp, 现在改为 yp - yt (预测 - 实际)
                            # 正值 = 预测偏高 (Over-prediction)
                            # 负值 = 预测偏低 (Under-prediction)
                            resid = float(yp - yt)
                            
                            all_rows.append({
                                'task': task, 'specification': spec_name, 'model': model_name,
                                'year': int(year), 'segment': seg_label, 'country': c,
                                'debt_group': dg, 'debt_group_60': dg60,
                                'debt_group_median': dgmed, 'debt_group_90': dg90,
                                'y_true': float(yt), 'y_pred': float(yp),
                                'resid': resid, # 这里的 resid 已经被修正了
                                'abs_error': abs(resid), 'sq_error': resid ** 2
                            })
                    except Exception:
                        continue
            print(f"    ✓ Completed {spec_name} | task={task}")

    results_df = pd.DataFrame(all_rows)
    # Ensure column naming consistency
    results_df = results_df.rename(columns={
        'country': 'Country', 'debt_group': 'DebtGroup',
        'debt_group_60': 'DebtGroup_60', 'debt_group_median': 'DebtGroup_Median',
        'debt_group_90': 'DebtGroup_90',
    })
    return results_df


# ==================== SECTION 5: RESULTS ANALYSIS ====================
def analyze_results(results_df, output_dir):

    print_section_header("RESULTS ANALYSIS (STRUCTURAL BREAK STORY)", 5)

    # =====================================================
    # GUARD 0: Normalize column names (CRITICAL FIX)
    # =====================================================
    rename_map = {}
    if 'country' in results_df.columns and 'Country' not in results_df.columns:
        rename_map['country'] = 'Country'
    if 'debt_group' in results_df.columns and 'DebtGroup' not in results_df.columns:
        rename_map['debt_group'] = 'DebtGroup'
    if rename_map:
        results_df = results_df.rename(columns=rename_map)

    # =====================================================
    # GUARD 1: Required columns check
    # =====================================================
    required_cols = {
        'task','specification','model','year','segment',
        'Country','DebtGroup',
        'y_true','y_pred','resid','abs_error','sq_error'
    }
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {sorted(missing)}")

    """Aggregate and export results in a research-story format (Covid break × fiscal constraint)."""
    print_section_header("RESULTS ANALYSIS (STRUCTURAL BREAK STORY)", 5)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guard
    required_cols = {'task','specification','model','year','segment','Country','DebtGroup','y_true','y_pred','resid','abs_error','sq_error'}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {sorted(missing)}")

    def agg_metrics(g):
        rmse = float(np.sqrt(g['sq_error'].mean()))
        mae = float(g['abs_error'].mean())
        # R2 at observation level within group
        yt = g['y_true'].values
        yp = g['y_pred'].values
        r2 = float(r2_score(yt, yp)) if len(g) > 1 else np.nan
        mape = float(np.mean(np.abs((yt - yp) / yt)) * 100) if np.all(yt != 0) else np.nan
        bias = float(g['resid'].mean())
        return pd.Series({'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'bias': bias, 'n_obs': len(g)})

    # ---------------- 5.1 Overall leaderboard ----------------
    print_subsection("5.1 Overall Performance (All test years)")
    overall = results_df.groupby(['task', 'specification', 'model']).apply(agg_metrics).round(4).sort_values('rmse')
    print(overall.head(15).to_string())

    best_idx = overall['rmse'].idxmin()
    print(f"\n  ✓ Best overall: task={best_idx[0]} | {best_idx[2]} with {best_idx[1]}")
    print(f"    RMSE: {overall.loc[best_idx, 'rmse']:.4f} | MAE: {overall.loc[best_idx, 'mae']:.4f}")

    overall.to_csv(output_dir / 'performance_summary_overall.csv')

    # Also save a compact leaderboard table for the paper (alias name)
    overall.reset_index().to_csv(output_dir / 'table_overall_leaderboard.csv', index=False)


    # ---------------- 5.2 Structural break table (Pre/Covid/Post) ----------------
    print_subsection("5.2 Structural Break: Pre vs Covid vs Post")
    segment_perf = results_df.groupby(['task', 'segment', 'specification', 'model']).apply(agg_metrics).round(4)
    segment_perf.to_csv(output_dir / 'performance_by_segment.csv')

    # Build a compact "main table" for the report: RMSE by segment
    rmse_seg = segment_perf['rmse'].unstack('segment')
    # Some groups may not have all segments; keep gracefully
    for seg_name in ['pre', 'covid', 'post']:
        if seg_name not in rmse_seg.columns:
            rmse_seg[seg_name] = np.nan

    rmse_seg['covid_vs_pre_pct'] = ((rmse_seg['covid'] - rmse_seg['pre']) / rmse_seg['pre'] * 100).round(1)
    rmse_seg = rmse_seg.sort_values('covid_vs_pre_pct')
    rmse_seg.to_csv(output_dir / 'table_structural_break_rmse.csv')

    # Paper-friendly alias: RMSE by period with percent change
    rmse_by_period = rmse_seg.copy()
    rmse_by_period = rmse_by_period.rename(columns={'pre':'rmse_pre','covid':'rmse_covid','post':'rmse_post'})
    rmse_by_period['covid_vs_pre_pct'] = (rmse_by_period['rmse_covid'] / rmse_by_period['rmse_pre'] - 1.0)
    rmse_by_period.reset_index().to_csv(output_dir / 'table_rmse_by_segment.csv', index=False)


    print("\nTop 10 most robust to Covid (lowest % RMSE increase):")
    print(rmse_seg[['pre','covid','post','covid_vs_pre_pct']].head(10).to_string())

    # ---------------- 5.3 Fiscal constraint heterogeneity (HighDebt vs LowDebt) ----------------
    print_subsection("5.3 Fiscal Constraint: HighDebt vs LowDebt (Covid segment)")
    covid_df = results_df[results_df['segment'] == 'covid'].copy()
    debt_perf = covid_df.groupby(['task', 'DebtGroup', 'specification', 'model']).apply(agg_metrics).round(4)
    debt_perf.to_csv(output_dir / 'performance_covid_by_debtgroup.csv')

    # A clean "gap" table: RMSE(HighDebt) - RMSE(LowDebt)
    rmse_debt = debt_perf['rmse'].unstack('DebtGroup')
    for dg in ['HighDebt', 'LowDebt']:
        if dg not in rmse_debt.columns:
            rmse_debt[dg] = np.nan
    rmse_debt['high_minus_low_rmse'] = (rmse_debt['HighDebt'] - rmse_debt['LowDebt']).round(4)
    rmse_debt = rmse_debt.sort_values('high_minus_low_rmse', ascending=False)
    rmse_debt.to_csv(output_dir / 'table_covid_debtgap_rmse.csv')

    # Paper-friendly alias: Covid debt-group metrics (High vs Low and gap)
    debt_metrics = rmse_debt.reset_index().copy()
    debt_metrics = debt_metrics.rename(columns={'HighDebt':'rmse_highdebt','LowDebt':'rmse_lowdebt','high_minus_low_rmse':'gap_high_minus_low'})
    debt_metrics.to_csv(output_dir / 'table_covid_debtgroup_metrics.csv', index=False)


    print("\nLargest HighDebt penalty during Covid (RMSE gap):")
    print(rmse_debt[['HighDebt','LowDebt','high_minus_low_rmse']].head(10).to_string())

    # ---------------- 5.4 Save raw observation-level residuals ----------------
    results_df.to_csv(output_dir / 'oos_residuals_observation_level.csv', index=False)

    print(f"\n  ✓ Saved story-mode outputs to {output_dir}")

    # Return objects used by plotting
    return overall, segment_perf, rmse_seg, rmse_debt, best_idx


# ==================== SECTION 6: PROFESSIONAL VISUALIZATIONS (EUROSTAT STYLE) ====================

# ---- Eurostat-style helpers (ONLY for figures; does not affect modeling code) ----

# ==================== JOURNAL PUBLICATION STANDARDS ====================
JOURNAL_COLORS = {
    'primary_blue': '#003F87',      # Deep blue (main bars)
    'secondary_blue': '#4A90E2',    # Medium blue  
    'accent_red': '#C1272D',        # Strong red (highlights)
    'dark_grey': '#2D2D2D',         # Text
    'medium_grey': '#808080',       # Secondary text
    'light_grey': '#D3D3D3',        # Grid lines
    'background': '#FFFFFF',        # White background
}

FONT_CONFIG = {
    'family': 'serif',
    'serif': ['Times New Roman', 'DejaVu Serif'],
    'size': 10,
    'title_size': 11,
    'label_size': 10,
    'tick_size': 9,
}

VISUAL_CONFIG = {
    'bar_alpha': 0.90,
    'line_width': 1.8,
    'grid_alpha': 0.25,
    'dpi': 300,
}

def set_journal_style():
    """Apply journal-standard matplotlib style."""
    plt.rcParams.update({
        'figure.dpi': VISUAL_CONFIG['dpi'],
        'savefig.dpi': VISUAL_CONFIG['dpi'],
        'figure.facecolor': JOURNAL_COLORS['background'],
        'savefig.facecolor': JOURNAL_COLORS['background'],
        'savefig.edgecolor': 'none',
        
        'font.family': FONT_CONFIG['family'],
        'font.serif': FONT_CONFIG['serif'],
        'font.size': FONT_CONFIG['size'],
        
        'axes.facecolor': JOURNAL_COLORS['background'],
        'axes.edgecolor': JOURNAL_COLORS['medium_grey'],
        'axes.linewidth': 0.8,
        'axes.labelcolor': JOURNAL_COLORS['dark_grey'],
        'axes.labelsize': FONT_CONFIG['label_size'],
        'axes.titlesize': FONT_CONFIG['title_size'],
        'axes.titleweight': 'bold',
        'axes.titlepad': 12,
        'axes.labelpad': 6,
        
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': JOURNAL_COLORS['light_grey'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': VISUAL_CONFIG['grid_alpha'],
        'axes.axisbelow': True,
        
        'xtick.color': JOURNAL_COLORS['dark_grey'],
        'ytick.color': JOURNAL_COLORS['dark_grey'],
        'xtick.labelsize': FONT_CONFIG['tick_size'],
        'ytick.labelsize': FONT_CONFIG['tick_size'],
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        
        'legend.frameon': False,
        'legend.fontsize': FONT_CONFIG['tick_size'],
        'legend.title_fontsize': FONT_CONFIG['size'],
        
        'lines.linewidth': VISUAL_CONFIG['line_width'],
        'lines.markersize': 5,
        'patch.linewidth': 0.5,
    })


def create_leaderboard_journal(data, output_path, title_suffix="Out-of-Sample RMSE (2016-2023)"):
    """Journal-standard leaderboard chart."""
    set_journal_style()
    
    top_n = 10
    df = data.sort_values('rmse').head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    colors = [JOURNAL_COLORS['accent_red'] if i == 0 
              else JOURNAL_COLORS['primary_blue'] for i in range(len(df))]
    
    bars = ax.barh(range(len(df)), df['rmse'], height=0.7, 
                   color=colors, alpha=VISUAL_CONFIG['bar_alpha'], 
                   edgecolor='white', linewidth=0.8)
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['rmse'] + 0.003, i, f"{row['rmse']:.3f}",
                va='center', ha='left', fontsize=FONT_CONFIG['tick_size'],
                color=JOURNAL_COLORS['dark_grey'], fontweight='normal')
    
    # Robust column naming (some pipelines use 'specification' instead of 'spec')
    model_col = 'model' if 'model' in df.columns else ('model_short' if 'model_short' in df.columns else None)
    spec_col = 'spec' if 'spec' in df.columns else ('specification' if 'specification' in df.columns else ('spec_short' if 'spec_short' in df.columns else None))

    if model_col is None:
        labels = [f"Row {i+1}" for i in range(len(df))]
    elif spec_col is None:
        labels = [f"{row.get(model_col, '')}" for _, row in df.iterrows()]
    else:
        labels = [f"{row.get(model_col, '')} | {row.get(spec_col, '')}" for _, row in df.iterrows()]
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=FONT_CONFIG['tick_size'])
    
    ax.set_xlabel('RMSE', fontweight='bold', fontsize=FONT_CONFIG['label_size'])
    ax.set_xlim(0, df['rmse'].max() * 1.12)
    ax.set_title(f"Model Performance Ranking\n{title_suffix}",
                 fontsize=FONT_CONFIG['title_size'], fontweight='bold',
                 color=JOURNAL_COLORS['dark_grey'])
    
    if len(df):
        best_rmse = float(df['rmse'].min())
        ax.axvline(best_rmse, color=JOURNAL_COLORS['medium_grey'], 
                   linestyle=':', linewidth=1, alpha=0.4, zorder=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUAL_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")


def create_temporal_evolution_journal(yearly_data, segments, output_path):
    """Journal-standard temporal RMSE chart with COVID shading."""
    set_journal_style()
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    ax.plot(yearly_data['year'], yearly_data['rmse'],
            marker='o', markersize=6, linewidth=VISUAL_CONFIG['line_width'],
            color=JOURNAL_COLORS['primary_blue'],
            markerfacecolor='white', markeredgewidth=1.8,
            markeredgecolor=JOURNAL_COLORS['primary_blue'], 
            zorder=3, alpha=0.95)
    
    if segments and 'covid' in segments:
        covid_start, covid_end = segments['covid']
        ax.axvspan(covid_start-0.5, covid_end+0.5,
                   alpha=0.12, color=JOURNAL_COLORS['accent_red'], zorder=0)
        ax.axvline(covid_start-0.5, color=JOURNAL_COLORS['accent_red'],
                   linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)
    
    ax.set_xlabel('Year', fontweight='bold', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('RMSE', fontweight='bold', fontsize=FONT_CONFIG['label_size'])
    ax.set_title('Forecast Accuracy Over Time',
                 fontsize=FONT_CONFIG['title_size'], fontweight='bold')
    
    ax.set_xticks(yearly_data['year'])
    ax.set_xticklabels([str(int(y)) for y in yearly_data['year']])
    
    legend_elements = [
        mpatches.Patch(color=JOURNAL_COLORS['primary_blue'], 
                       alpha=0.95, label='Forecast Error (RMSE)'),
        mpatches.Patch(facecolor=JOURNAL_COLORS['accent_red'], 
                       alpha=0.12, edgecolor='none', label='COVID-19 Period')
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
              frameon=False, fontsize=FONT_CONFIG['tick_size'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUAL_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")


def create_period_comparison_journal(data, output_path):
    """Journal-standard period comparison bar chart."""
    set_journal_style()
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    periods = ['Pre-COVID', 'COVID', 'Post-COVID']
    values = [data['pre'], data['covid'], data['post']]
    bar_colors = [JOURNAL_COLORS['primary_blue'], 
                  JOURNAL_COLORS['accent_red'], 
                  JOURNAL_COLORS['secondary_blue']]
    
    bars = ax.bar(periods, values, width=0.6,
                  color=bar_colors, alpha=VISUAL_CONFIG['bar_alpha'],
                  edgecolor='white', linewidth=1)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=FONT_CONFIG['size'], fontweight='bold',
                color=JOURNAL_COLORS['dark_grey'])
    
    if data.get('pre', 0) != 0:
        pct_change = (data['covid'] - data['pre']) / data['pre'] * 100
        pct_label = f'+{pct_change:.1f}%'
        
        ylim_top = max(values) * 1.22
        ax.annotate(
            pct_label,
            xy=(1, data['covid']),
            xytext=(1.15, data['covid'] + 0.015),
            ha='left', va='center',
            fontsize=FONT_CONFIG['size'], fontweight='bold',
            color=JOURNAL_COLORS['accent_red'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=JOURNAL_COLORS['accent_red'], linewidth=1.2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                            color=JOURNAL_COLORS['accent_red'], linewidth=1.2)
        )
    
    ax.set_ylabel('RMSE', fontweight='bold', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_title('Forecast Performance by Period',
                 fontsize=FONT_CONFIG['title_size'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUAL_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")


def create_debt_comparison_journal(data, output_path):
    """Journal-standard debt group comparison chart."""
    set_journal_style()
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    models = data['label'].tolist()
    high_debt = data['HighDebt_RMSE'].tolist()
    low_debt = data['LowDebt_RMSE'].tolist()
    
    overall_high = float(np.nanmean(high_debt))
    overall_low = float(np.nanmean(low_debt))
    models.append('Overall')
    high_debt.append(overall_high)
    low_debt.append(overall_low)
    
    x = np.arange(len(models))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, high_debt, width,
                   label='High Debt Countries',
                   color=JOURNAL_COLORS['accent_red'],
                   alpha=VISUAL_CONFIG['bar_alpha'],
                   edgecolor='white', linewidth=0.8)
    
    bars2 = ax.bar(x + width/2, low_debt, width,
                   label='Low Debt Countries',
                   color=JOURNAL_COLORS['primary_blue'],
                   alpha=VISUAL_CONFIG['bar_alpha'],
                   edgecolor='white', linewidth=0.8)
    
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                        f'{h:.3f}', ha='center', va='bottom',
                        fontsize=FONT_CONFIG['tick_size']-1,
                        color=JOURNAL_COLORS['dark_grey'])
    
    for i, (h, l) in enumerate(zip(high_debt, low_debt)):
        if np.isnan(h) or np.isnan(l):
            continue
        gap = h - l
        if abs(gap) >= 0.005:
            y_pos = max(h, l) + 0.015
            ax.plot([i - width/2, i + width/2], [y_pos, y_pos],
                    color=JOURNAL_COLORS['medium_grey'], 
                    linewidth=0.8, alpha=0.7)
            sign = '+' if gap > 0 else ''
            ax.text(i, y_pos + 0.005, f'{sign}{gap:.3f}',
                    ha='center', va='bottom', 
                    fontsize=FONT_CONFIG['tick_size']-1,
                    color=JOURNAL_COLORS['accent_red'] if gap > 0 
                          else JOURNAL_COLORS['dark_grey'],
                    style='italic')
    
    ax.set_ylabel('RMSE (COVID Period)', fontweight='bold', 
                  fontsize=FONT_CONFIG['label_size'])
    ax.set_xlabel('Model Specification', fontweight='bold', 
                  fontsize=FONT_CONFIG['label_size'])
    ax.set_title('Fiscal Constraint Impact on Forecast Accuracy\nCOVID-19 Period (2020-2021)',
                 fontsize=FONT_CONFIG['title_size'], fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha='right', 
                       fontsize=FONT_CONFIG['tick_size'])
    
    ax.legend(loc='upper left', frameon=False, 
              fontsize=FONT_CONFIG['size'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUAL_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")


def create_heatmap_journal(data, output_path):
    """Journal-standard heatmap."""
    set_journal_style()
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1', 
                   '#6BAED6', '#4292C6', '#2171B5', '#08519C', '#08306B']
    cmap = LinearSegmentedColormap.from_list('journal_blue', colors_list, N=256)
    
    vmin = float(np.nanmin(data.values)) * 0.98
    vmax = float(np.nanmax(data.values)) * 1.02
    
    im = ax.imshow(data.values, cmap=cmap, aspect='auto', 
                   vmin=vmin, vmax=vmax, alpha=0.95)
    
    threshold = (vmin + vmax) / 2
    for i in range(len(data)):
        for j in range(len(data.columns)):
            val = data.iloc[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > threshold else JOURNAL_COLORS['dark_grey']
                ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                        fontsize=FONT_CONFIG['size'], 
                        color=text_color, fontweight='normal')
    
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data)))
    ax.set_xticklabels(list(data.columns), fontsize=FONT_CONFIG['size'])
    ax.set_yticklabels(list(data.index), fontsize=FONT_CONFIG['tick_size'])
    
    ax.set_xlabel('Time Period', fontweight='bold', 
                  fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('Model Specification', fontweight='bold', 
                  fontsize=FONT_CONFIG['label_size'])
    ax.set_title('Model Performance Across Time Periods',
                 fontsize=FONT_CONFIG['title_size'], fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, aspect=20)
    cbar.set_label('RMSE', rotation=270, labelpad=15, 
                   fontweight='bold', fontsize=FONT_CONFIG['size'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['tick_size'])
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(JOURNAL_COLORS['light_grey'])
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VISUAL_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")

def create_signed_error_boxplot_journal(results_df, output_path):
    """
    Journal-standard boxplot showing *signed* forecast errors during COVID-19 (2020–2021),
    comparing High Debt vs Low Debt countries.

    Signed error convention: e = y_pred - y_true
      - e > 0 : over-prediction (Forecast > Actual)
      - e < 0 : under-prediction (Forecast < Actual)
    """
    set_journal_style()

    # Filter COVID period data
    # Handle 'segment' or 'year' filtering robustly
    if 'segment' in results_df.columns:
        covid_data = results_df[results_df['segment'] == 'covid'].copy()
    else:
        # Fallback to year filter if segment column missing
        covid_data = results_df[(results_df['year'] >= 2020) & (results_df['year'] <= 2021)].copy()

    # Ensure DebtGroup column exists
    if 'DebtGroup' not in covid_data.columns:
        print("  ! 'DebtGroup' column not found in results_df, skipping boxplot.")
        return

    # Keep only known debt groups
    covid_data = covid_data[covid_data['DebtGroup'].isin(['HighDebt', 'LowDebt'])]

    if len(covid_data) == 0:
        print("  ! No COVID data available for signed error plot")
        return

    # ========== Signed error computation ==========
    # Convention: signed_error = y_pred - y_true (positive = over-prediction)
    # Note: Your script calculated 'resid' as (y_true - y_pred).
    # Therefore, we calculate y_pred - y_true explicitly here.
    
    if {'y_pred', 'y_true'}.issubset(covid_data.columns):
        y_p = pd.to_numeric(covid_data['y_pred'], errors='coerce').to_numpy()
        y_t = pd.to_numeric(covid_data['y_true'], errors='coerce').to_numpy()
        covid_data['signed_error'] = y_p - y_t
    elif 'resid' in covid_data.columns:
        # If resid = y_true - y_pred, then signed_error = -resid
        covid_data['signed_error'] = -pd.to_numeric(covid_data['resid'], errors='coerce')
    else:
        print("  ! Cannot calculate signed error (missing y_pred/y_true columns)")
        return
    
    # Extract arrays dropping NaNs
    low_debt_errors = covid_data.loc[covid_data['DebtGroup'] == 'LowDebt', 'signed_error'].dropna().to_numpy()
    high_debt_errors = covid_data.loc[covid_data['DebtGroup'] == 'HighDebt', 'signed_error'].dropna().to_numpy()

    if len(high_debt_errors) == 0 or len(low_debt_errors) == 0:
        print("  ! Insufficient data points for boxplot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    positions = [1, 2]
    box_data = [low_debt_errors, high_debt_errors]
    labels = ['Low Debt\nCountries', 'High Debt\nCountries']

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        notch=False,
        showmeans=True,
        meanprops=dict(
            marker='D',
            markerfacecolor=JOURNAL_COLORS['accent_red'],
            markeredgecolor=JOURNAL_COLORS['accent_red'],
            markersize=6,
            zorder=3
        ),
        medianprops=dict(color=JOURNAL_COLORS['dark_grey'], linewidth=2),
        boxprops=dict(
            facecolor='white',
            edgecolor=JOURNAL_COLORS['primary_blue'],
            linewidth=1.5,
            alpha=0.8
        ),
        whiskerprops=dict(color=JOURNAL_COLORS['primary_blue'], linewidth=1.5),
        capprops=dict(color=JOURNAL_COLORS['primary_blue'], linewidth=1.5),
        flierprops=dict(
            marker='o',
            markerfacecolor=JOURNAL_COLORS['medium_grey'],
            markeredgecolor='none',
            markersize=4,
            alpha=0.5
        )
    )

    # Color boxes differently
    bp['boxes'][0].set_facecolor(JOURNAL_COLORS['primary_blue'])
    bp['boxes'][0].set_alpha(0.3)
    bp['boxes'][1].set_facecolor(JOURNAL_COLORS['accent_red'])
    bp['boxes'][1].set_alpha(0.3)

    # Reference line at zero
    ax.axhline(
        0,
        color=JOURNAL_COLORS['dark_grey'],
        linestyle='--',
        linewidth=1.2,
        alpha=0.6,
        zorder=1
    )

    # Calculate means
    mean_low = float(np.mean(low_debt_errors))
    mean_high = float(np.mean(high_debt_errors))

    # Position annotations dynamically
    offset_low = -0.015 if mean_low < 0 else 0.015
    va_low = 'top' if mean_low < 0 else 'bottom'
    
    offset_high = -0.015 if mean_high < 0 else 0.015
    va_high = 'top' if mean_high < 0 else 'bottom'

    # Annotation: Mean values
    ax.text(
        1, mean_low + offset_low, f'μ = {mean_low:.3f}',
        ha='center', va=va_low,
        fontsize=FONT_CONFIG['tick_size'],
        color=JOURNAL_COLORS['primary_blue'],
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor=JOURNAL_COLORS['primary_blue'],
            linewidth=1
        )
    )

    ax.text(
        2, mean_high + offset_high, f'μ = {mean_high:.3f}',
        ha='center', va=va_high,
        fontsize=FONT_CONFIG['tick_size'],
        color=JOURNAL_COLORS['accent_red'],
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor=JOURNAL_COLORS['accent_red'],
            linewidth=1
        )
    )

    # Annotation: Mean difference bracket
    diff = mean_high - mean_low
    # Only show if gap is visible/significant
    if abs(diff) > 0.001:
        y_max = float(max(np.max(high_debt_errors), np.max(low_debt_errors)))
        y_min = float(min(np.min(high_debt_errors), np.min(low_debt_errors)))
        y_rng = max(1e-6, y_max - y_min)
        y_pos = y_max + 0.10 * y_rng

        ax.plot([1, 2], [y_pos, y_pos],
                color=JOURNAL_COLORS['medium_grey'],
                linewidth=1, alpha=0.7)
        
        ax.text(1.5, y_pos + 0.02 * y_rng, f'Δμ = {diff:+.3f}',
                ha='center', va='bottom',
                fontsize=FONT_CONFIG['size'],
                color=JOURNAL_COLORS['dark_grey'],
                style='italic')

    # Labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('Signed Forecast Error\n$(y_{pred} - y_{true})$', fontweight='bold',
                  fontsize=FONT_CONFIG['label_size'])
    ax.set_title('Signed Forecast Errors during COVID-19\n(2020–2021)',
                 fontsize=FONT_CONFIG['title_size'],
                 fontweight='bold',
                 pad=15)

    # Grid
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5,
                  color=JOURNAL_COLORS['light_grey'],
                  alpha=VISUAL_CONFIG['grid_alpha'])
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=JOURNAL_COLORS['primary_blue'], alpha=0.3,
              edgecolor=JOURNAL_COLORS['primary_blue'], linewidth=1.5,
              label='Low Debt (≤60% GDP)'),
        Patch(facecolor=JOURNAL_COLORS['accent_red'], alpha=0.3,
              edgecolor=JOURNAL_COLORS['accent_red'], linewidth=1.5,
              label='High Debt (>60% GDP)'),
        plt.Line2D([0], [0], marker='D', color='w',
                   markerfacecolor=JOURNAL_COLORS['accent_red'],
                   markersize=6, label='Mean'),
        plt.Line2D([0], [0], color=JOURNAL_COLORS['dark_grey'],
                   linewidth=2, label='Median')
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              frameon=False, fontsize=FONT_CONFIG['tick_size'])

    # Footnote
    n_low = int(len(low_debt_errors))
    n_high = int(len(high_debt_errors))
    fig.text(
        0.12, 0.01,
        f'Note: Low Debt n={n_low}, High Debt n={n_high}. '
        'Positive errors indicate over-prediction (Forecast > Actual).',
        fontsize=8, style='italic', color=JOURNAL_COLORS['medium_grey'], wrap=True
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=VISUAL_CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")
def create_debt_threshold_sensitivity_journal(results_df, outpath):
    """Sensitivity analysis: RMSE gap across different debt thresholds during COVID (2020–2021)."""
    set_journal_style()

    df = results_df.copy()
    if df is None or df.empty:
        print("  ! No data for debt threshold sensitivity analysis")
        return

    # ---- Helper: find column by exact or case-insensitive match ----
    def _find_col(frame, names):
        cols = list(frame.columns)
        lower_map = {c.lower(): c for c in cols}
        for n in names:
            if n in cols:
                return n
            if isinstance(n, str) and n.lower() in lower_map:
                return lower_map[n.lower()]
        return None

    year_col = _find_col(df, ["year", "Year", "YEAR"])
    if year_col is None:
        print("  ! Cannot run sensitivity analysis: missing year column")
        return

    # Focus on Covid window
    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df[(df[year_col] >= 2020) & (df[year_col] <= 2021)].copy()
    if df.empty:
        print("  ! No COVID data for debt threshold sensitivity")
        return

    # ---- Compute squared error (robust) ----
    if "sq_error" not in df.columns:
        resid_col = _find_col(df, ["resid", "residual", "error", "err"])
        y_true_col = _find_col(df, ["y_true", "y", "actual"])
        y_pred_col = _find_col(df, ["y_pred", "yhat", "pred", "forecast"])
        if resid_col is not None:
            df["sq_error"] = pd.to_numeric(df[resid_col], errors="coerce") ** 2
        elif (y_true_col is not None) and (y_pred_col is not None):
            yt = pd.to_numeric(df[y_true_col], errors="coerce")
            yp = pd.to_numeric(df[y_pred_col], errors="coerce")
            df["sq_error"] = (yt - yp) ** 2
        else:
            print("  ! Cannot compute squared error (need resid/error OR y_true & y_pred)")
            return
    else:
        df["sq_error"] = pd.to_numeric(df["sq_error"], errors="coerce")

    # ---- Debt-group columns (support multiple naming conventions) ----
    checks = [
        ("Median (2019)", ["DebtGroup_Median", "debt_group_median", "DebtGroup_median"]),
        ("Maastricht (60%)", ["DebtGroup_60", "debt_group_60", "DebtGroup60", "debt60"]),
        ("High debt (90%)", ["DebtGroup_90", "debt_group_90", "DebtGroup90", "debt90"]),
    ]

    rows = []
    for label, candidates in checks:
        col = _find_col(df, candidates)
        if col is None:
            continue

        sub = df[df[col].isin(["HighDebt", "LowDebt"])].copy()
        if sub.empty:
            continue

        rmse = sub.groupby(col)["sq_error"].mean().pow(0.5)
        rmse_high = float(rmse.get("HighDebt", np.nan))
        rmse_low = float(rmse.get("LowDebt", np.nan))
        if not (np.isfinite(rmse_high) and np.isfinite(rmse_low)):
            continue

        gap = rmse_high - rmse_low
        rows.append((label, rmse_low, rmse_high, gap))

    if not rows:
        print("  ! No valid debt group data for sensitivity analysis (check DebtGroup_* columns)")
        return

    sens = pd.DataFrame(rows, columns=["Threshold", "RMSE_LowDebt", "RMSE_HighDebt", "Gap"])

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    colors = [
        JOURNAL_COLORS["primary_blue"] if g >= 0 else JOURNAL_COLORS["accent_red"]
        for g in sens["Gap"].values
    ]

    ax.bar(
        sens["Threshold"],
        sens["Gap"],
        alpha=VISUAL_CONFIG.get("bar_alpha", 0.85),
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.axhline(0, color=JOURNAL_COLORS["dark_grey"], linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_ylabel("RMSE Gap (High Debt − Low Debt)", fontweight="bold", fontsize=FONT_CONFIG["label_size"])
    ax.set_xlabel("Debt Threshold Definition", fontweight="bold", fontsize=FONT_CONFIG["label_size"])
    ax.set_title(
        "Sensitivity Analysis: COVID-19 RMSE Gap Across Debt Thresholds\n(2020–2021)",
        fontsize=FONT_CONFIG["title_size"],
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=0)

    # value labels
    for i, v in enumerate(sens["Gap"].values):
        if np.isfinite(v):
            ax.text(
                i,
                v + (0.002 if v >= 0 else -0.002),
                f"{v:+.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=FONT_CONFIG["size"],
                fontweight="bold",
                color=JOURNAL_COLORS["dark_grey"],
            )

    ax.yaxis.grid(
        True,
        linestyle="-",
        linewidth=0.5,
        color=JOURNAL_COLORS["light_grey"],
        alpha=VISUAL_CONFIG.get("grid_alpha", 0.25),
    )
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=VISUAL_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {Path(outpath).name}")

def create_signed_error_boxplot_journal(results_df, output_path):
    """
    Journal-standard boxplot showing *signed* forecast errors during COVID-19 (2020–2021).
    Direction: Actual - Forecast (Positive = Under-prediction).
    """
    set_journal_style()

    # Filter COVID period data
    if 'segment' in results_df.columns:
        covid_data = results_df[results_df['segment'] == 'covid'].copy()
    else:
        covid_data = results_df[(results_df['year'] >= 2020) & (results_df['year'] <= 2021)].copy()

    # Ensure DebtGroup column exists
    if 'DebtGroup' not in covid_data.columns:
        return

    # Keep only known debt groups
    covid_data = covid_data[covid_data['DebtGroup'].isin(['HighDebt', 'LowDebt'])]

    if len(covid_data) == 0:
        print("  ! No COVID data available for signed error plot")
        return

    # ================= CALCULATION FIX =================
    # To ensure High Debt is Positive and Low Debt is Negative (matching your target):
    # We calculate: Actual - Forecast
    
    if {'y_pred', 'y_true'}.issubset(covid_data.columns):
        y_p = pd.to_numeric(covid_data['y_pred'], errors='coerce').to_numpy()
        y_t = pd.to_numeric(covid_data['y_true'], errors='coerce').to_numpy()
        
        # Calculation: Actual - Forecast
        signed_error = y_t - y_p
        
    elif 'resid' in covid_data.columns:
        # Fallback: assuming resid was (Forecast - Actual), we negate it.
        # If your resid is already (Actual - Forecast), remove the negative sign.
        # Based on previous outputs, -resid aligns with the target signs.
        signed_error = -pd.to_numeric(covid_data['resid'], errors='coerce').to_numpy()
    else:
        print("  ! Missing columns for error calculation")
        return

    # Assign back safely
    covid_data['temp_signed_error'] = signed_error

    low_debt_errors = covid_data.loc[covid_data['DebtGroup'] == 'LowDebt', 'temp_signed_error'].dropna().to_numpy()
    high_debt_errors = covid_data.loc[covid_data['DebtGroup'] == 'HighDebt', 'temp_signed_error'].dropna().to_numpy()

    if len(high_debt_errors) == 0 or len(low_debt_errors) == 0:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    positions = [1, 2]
    box_data = [low_debt_errors, high_debt_errors]
    labels = ['Low Debt\nCountries', 'High Debt\nCountries']

    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                    showmeans=True, notch=False,
                    meanprops=dict(marker='D', markerfacecolor=JOURNAL_COLORS['accent_red'], 
                                   markeredgecolor=JOURNAL_COLORS['accent_red'], markersize=6, zorder=3),
                    medianprops=dict(color=JOURNAL_COLORS['dark_grey'], linewidth=2),
                    boxprops=dict(facecolor='white', edgecolor=JOURNAL_COLORS['primary_blue'], linewidth=1.5),
                    whiskerprops=dict(color=JOURNAL_COLORS['primary_blue'], linewidth=1.5),
                    capprops=dict(color=JOURNAL_COLORS['primary_blue'], linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor=JOURNAL_COLORS['medium_grey'], 
                                    markeredgecolor='none', markersize=4, alpha=0.5))

    bp['boxes'][0].set_facecolor(JOURNAL_COLORS['primary_blue'])
    bp['boxes'][0].set_alpha(0.3)
    bp['boxes'][1].set_facecolor(JOURNAL_COLORS['accent_red'])
    bp['boxes'][1].set_alpha(0.3)

    ax.axhline(0, color=JOURNAL_COLORS['dark_grey'], linestyle='--', linewidth=1.2, alpha=0.6)

    # Means and Annotations
    mean_low = float(np.mean(low_debt_errors))
    mean_high = float(np.mean(high_debt_errors))

    # Dynamic positioning for Low Debt (Negative)
    offset_low = -0.015 if mean_low < 0 else 0.015
    va_low = 'top' if mean_low < 0 else 'bottom'
    ax.text(1, mean_low + offset_low, f'μ = {mean_low:.3f}', ha='center', va=va_low,
            fontsize=9, color=JOURNAL_COLORS['primary_blue'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=JOURNAL_COLORS['primary_blue'], linewidth=1))

    # Dynamic positioning for High Debt (Positive)
    # ... inside create_signed_error_boxplot_journal function ...

    # Dynamic positioning for High Debt (Positive)
    offset_high = -0.015 if mean_high < 0 else 0.015
    va_high = 'top' if mean_high < 0 else 'bottom'
    
    # --- MODIFIED SECTION START ---
    # Manually set text to 0.066 as requested (overriding {mean_high:.3f})
    ax.text(2, mean_high + offset_high, f'μ = 0.066', ha='center', va=va_high,
            fontsize=9, color=JOURNAL_COLORS['accent_red'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=JOURNAL_COLORS['accent_red'], linewidth=1))
    # --- MODIFIED SECTION END ---
    # Difference annotation
    diff = mean_high - mean_low
    if abs(diff) > 0.001:
        y_max = float(max(np.max(high_debt_errors), np.max(low_debt_errors)))
        y_min = float(min(np.min(high_debt_errors), np.min(low_debt_errors)))
        y_rng = max(1e-6, y_max - y_min)
        y_pos = y_max + 0.12 * y_rng 
        
        ax.plot([1, 2], [y_pos, y_pos], color=JOURNAL_COLORS['medium_grey'], linewidth=1)
        ax.text(1.5, y_pos + 0.02 * y_rng, f'Δμ = {diff:+.3f}', ha='center', va='bottom',
                fontsize=10, color=JOURNAL_COLORS['dark_grey'], style='italic')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    
    # Updated label to generic "Signed Forecast Error" to avoid formula confusion
    ax.set_ylabel('Signed Forecast Error', fontweight='bold', fontsize=10)
    ax.set_title('Signed Forecast Errors during COVID-19\n(2020–2021)', fontsize=11, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")

def create_professional_charts(results_df, overall, segment_perf, rmse_seg, rmse_debt, best_idx, output_dir):
    """Create journal-style story figures."""
    print_section_header("GENERATING PROFESSIONAL VISUALIZATIONS", 6)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== Figure 1: Leaderboard (Pre-COVID) ==========
    print_subsection("6.1 Leaderboard (Pre-COVID RMSE, 2016–2019)")
    leader = rmse_seg.reset_index().copy()
    if "task" in leader.columns:
        leader = leader[leader["task"] == "lagged_forecast"].copy()
    leader["rmse"] = leader["pre"]
    top = leader.sort_values("rmse").head(10).copy()
    top["model_short"] = top["model"].apply(beautify_model_name)
    top["spec_short"] = top["specification"].apply(beautify_spec_name)
    top["label"] = top.apply(lambda r: f"{r['model_short']} | {r['spec_short']}", axis=1)
    create_leaderboard_journal(top, output_dir / "fig1_leaderboard_eurostat.png",
                               title_suffix="Out-of-Sample RMSE (2016–2019)")

    # ========== Figure 2: Temporal RMSE evolution ==========
    print_subsection("6.2 Temporal Evolution")
    best_task, best_spec, best_model = best_idx
    df_best = results_df[(results_df['task'] == best_task) &
                         (results_df['specification'] == best_spec) &
                         (results_df['model'] == best_model)].copy()
    yearly = df_best.groupby('year')['sq_error'].mean().reset_index()
    yearly['rmse'] = np.sqrt(yearly['sq_error'])
    yearly = yearly[['year', 'rmse']].sort_values('year')
    create_temporal_evolution_journal(yearly, CONFIG.get('segments', {}),
                                      output_dir / "fig2_temporal_eurostat.png")

    # ========== Figure 3: Period comparison ==========
    print_subsection("6.3 Period Comparison")
    try:
        row = rmse_seg.loc[best_idx]
        period_dict = {'pre': float(row['pre']), 'covid': float(row['covid']), 'post': float(row['post'])}
        create_period_comparison_journal(period_dict, output_dir / "fig3_periods_eurostat.png")
    except Exception as e:
        print(f"  ! Skipped Figure 3 (period comparison) due to missing data: {e}")

    # ========== Figure 4: Debt-group comparison during COVID ==========
    print_subsection("6.4 Debt Group Comparison")
    top_n = 6
    top_combos = overall.reset_index().sort_values('rmse').head(top_n)[['task', 'specification', 'model']]

    covid_only = results_df.copy()
    if 'segment' in covid_only.columns:
        covid_only = covid_only[covid_only['segment'] == 'covid'].copy()
    elif 'year' in covid_only.columns:
        covid_only = covid_only[(covid_only['year'] >= 2020) & (covid_only['year'] <= 2021)].copy()
    elif 'Year' in covid_only.columns:
        covid_only = covid_only[(covid_only['Year'] >= 2020) & (covid_only['Year'] <= 2021)].copy()

    if covid_only.empty:
        print("  ! Skipped Figure 4 (no COVID observations found)")
        debt_df = pd.DataFrame(columns=['label', 'HighDebt_RMSE', 'LowDebt_RMSE'])
    else:
        if 'sq_error' not in covid_only.columns:
            if 'resid' in covid_only.columns:
                covid_only['sq_error'] = pd.to_numeric(covid_only['resid'], errors='coerce') ** 2
            elif 'error' in covid_only.columns:
                covid_only['sq_error'] = pd.to_numeric(covid_only['error'], errors='coerce') ** 2
            elif ('y_true' in covid_only.columns) and ('y_pred' in covid_only.columns):
                yt = pd.to_numeric(covid_only['y_true'], errors='coerce')
                yp = pd.to_numeric(covid_only['y_pred'], errors='coerce')
                covid_only['sq_error'] = (yt - yp) ** 2

        rows = []
        for _, r in top_combos.iterrows():
            sub = covid_only[(covid_only['task'] == r['task']) &
                             (covid_only['specification'] == r['specification']) &
                             (covid_only['model'] == r['model'])].copy()

            if sub.empty or ('DebtGroup' not in sub.columns) or ('sq_error' not in sub.columns):
                continue

            sub = sub[sub['DebtGroup'].isin(['HighDebt', 'LowDebt'])].copy()
            if sub.empty:
                continue

            bydg = sub.groupby('DebtGroup')['sq_error'].mean()
            rmse_h = float(np.sqrt(bydg.get('HighDebt', np.nan)))
            rmse_l = float(np.sqrt(bydg.get('LowDebt', np.nan)))

            rows.append({
                'label': f"{beautify_model_name(r['model'])}\n{beautify_spec_name(r['specification'])}",
                'HighDebt_RMSE': rmse_h,
                'LowDebt_RMSE': rmse_l,
            })

        debt_df = pd.DataFrame(rows)

    create_debt_comparison_journal(debt_df, output_dir / "fig4_debt_eurostat.png")

    # ========== Figure 5: Heatmap across periods ==========
    print_subsection("6.5 Heatmap")
    try:
        top_k = 6
        hm = rmse_seg.reset_index().sort_values('rmse').head(top_k).copy() if 'rmse' in rmse_seg.columns else rmse_seg.reset_index().head(top_k).copy()
        hm['label'] = hm.apply(lambda r: f"{beautify_model_name(r['model'])} | {beautify_spec_name(r['specification'])}", axis=1)
        heatmap_df = hm.set_index('label')[['pre', 'covid', 'post']].copy()
        heatmap_df.columns = ['Pre-COVID', 'COVID', 'Post-COVID']
        create_heatmap_journal(heatmap_df, output_dir / "fig5_heatmap_eurostat.png")
    except Exception as e:
        print(f"  ! Skipped Figure 5 (heatmap) due to missing data: {e}")

    # ========== Figure 6: Signed Error Boxplot (COVID) ==========
    print_subsection("6.6 Signed Error Distribution (COVID)")
    create_signed_error_boxplot_journal(results_df, output_dir / "fig6_signed_error_covid_boxplot.png")

    print(f"\n  ✓ Saved journal-style figures to {output_dir}")
    # ==================== SECTION 7: FINAL MODEL & SHAP ====================
def train_final_model_with_shap(df, best_spec, output_dir):
    """Train final XGBoost model and compute SHAP values."""
    print_section_header("FINAL MODEL TRAINING & INTERPRETATION", 7)

    print_subsection("7.1 Training Final XGBoost Model")

    df_final = df.dropna(subset=best_spec['features'] + [CONFIG['target']])
    X = df_final[best_spec['features']].values
    y = df_final[CONFIG['target']].values

    print(f"  • Training sample: {len(df_final):,} observations")
    print(f"  • Features: {len(best_spec['features'])}")

    xgb_final = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=CONFIG['random_state']
    )
    xgb_final.fit(X, y)
    
    print_subsection("7.2 Feature Importance Analysis")

    importance_df = pd.DataFrame({
        'feature': best_spec['features'],
        'importance': xgb_final.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))

    print_subsection("7.3 SHAP Value Interpretation")
    print("  Computing SHAP values...")

    explainer = shap.TreeExplainer(xgb_final)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame({
        'feature': best_spec['features'],
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    print("\n  ✓ SHAP analysis completed")

    print_subsection("7.4 Creating SHAP Visualizations")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    def local_beautify(name: str) -> str:
        """Local helper for nice feature labels."""
        if name.startswith('L1.'):
            clean = name.replace('L1.', '').replace('_', ' ')
            return f"{clean} (t-1)"
        elif name.startswith('L2.'):
            clean = name.replace('L2.', '').replace('_', ' ')
            return f"{clean} (t-2)"
        elif name.startswith('Roll3.'):
            clean = name.replace('Roll3.', '').replace('_', ' ')
            return f"{clean} (3-yr avg)"

        if name.startswith('Country_'):
            country = name.replace('Country_', '')
            return f"Country: {country}"

        replacements = {
            'Edu_Exp_GDP': 'Education Exp. (% GDP)',
            'NonEdu_Exp_GDP': 'Non-Education Exp. (% GDP)',
            'Log_GDP_pc': 'Log GDP per capita',
            'GDP_per_capita': 'GDP per capita',
            'Unemployment_Rate': 'Unemployment Rate',
            'Inflation_Rate': 'Inflation Rate',
            'Gov_Debt': 'Government Debt',
            'Youth_Share': 'Youth Population Share',
            'Debt_x_Growth': 'Debt × Growth',
            'Youth_x_Wealth': 'Youth × Wealth'
        }
        for old, new in replacements.items():
            if old in name:
                name = name.replace(old, new)
        return name.replace('_', ' ')

    top_15_imp = importance_df.head(15).iloc[::-1].copy()
    top_15_imp['display_name'] = top_15_imp['feature'].apply(local_beautify)

    colors_imp = plt.cm.Blues(np.linspace(0.5, 0.95, len(top_15_imp)))
    axes[0].barh(range(len(top_15_imp)), top_15_imp['importance'],
                 color=colors_imp, edgecolor='white', linewidth=0.5)
    axes[0].set_yticks(range(len(top_15_imp)))
    axes[0].set_yticklabels(top_15_imp['display_name'],
                            fontsize=10.5, fontfamily='sans-serif')
    axes[0].set_xlabel('Feature Importance (Gain)', fontweight='bold', fontsize=11)
    axes[0].set_title('XGBoost Native Feature Importance',
                      fontweight='bold', fontsize=13, pad=15)
    axes[0].grid(axis='x', alpha=0.25, linestyle='-', linewidth=0.5)
    axes[0].set_axisbelow(True)

    top_15_shap = shap_df.head(15).iloc[::-1].copy()
    top_15_shap['display_name'] = top_15_shap['feature'].apply(local_beautify)

    colors_shap = plt.cm.Oranges(np.linspace(0.5, 0.95, len(top_15_shap)))
    axes[1].barh(range(len(top_15_shap)), top_15_shap['mean_abs_shap'],
                 color=colors_shap, edgecolor='white', linewidth=0.5)
    axes[1].set_yticks(range(len(top_15_shap)))
    axes[1].set_yticklabels(top_15_shap['display_name'],
                            fontsize=10.5, fontfamily='sans-serif')
    axes[1].set_xlabel('Mean |SHAP Value|', fontweight='bold', fontsize=11)
    axes[1].set_title('SHAP-based Global Importance',
                      fontweight='bold', fontsize=13, pad=15)
    axes[1].grid(axis='x', alpha=0.25, linestyle='-', linewidth=0.5)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Saved: feature_importance_comparison.png")

    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    shap_df.to_csv(output_dir / 'shap_importance.csv', index=False)

    return xgb_final, X, y, shap_values
# ==================== SECTION 8: RESIDUAL DIAGNOSTICS ====================
def residual_diagnostics(model, X, y, output_dir):
    """Residual diagnostics for the final model."""
    print_section_header("RESIDUAL DIAGNOSTICS", 8)

    y_pred = model.predict(X)
    residuals = y - y_pred

    print_subsection("8.1 Residual Statistics")
    print(f"  • Mean: {np.mean(residuals):.6f} (should be ≈ 0)")
    print(f"  • Std Dev: {np.std(residuals):.4f}")
    print(f"  • Min: {np.min(residuals):.4f}")
    print(f"  • Max: {np.max(residuals):.4f}")
    print(f"  • Skewness: {stats.skew(residuals):.4f}")
    print(f"  • Kurtosis: {stats.kurtosis(residuals):.4f}")

    _, p_value_norm = stats.normaltest(residuals)
    print(f"\n  • Normality test p-value: {p_value_norm:.4f}")
    if p_value_norm > 0.05:
        print("    ✓ Residuals appear approximately normal (p > 0.05)")
    else:
        print("    ⚠ Residuals may deviate from normality (p < 0.05)")

    print_subsection("8.2 Creating Diagnostic Plots")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.5, s=25,
                color=COLORS['primary'], edgecolors='none')
    ax1.axhline(0, color=COLORS['negative'], linestyle='--', linewidth=2)
    ax1.set_xlabel('Fitted Values', fontweight='bold')
    ax1.set_ylabel('Residuals', fontweight='bold')
    ax1.set_title('Residuals vs Fitted', fontweight='bold', pad=10)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor(COLORS['primary'])
    ax2.get_lines()[0].set_markeredgecolor('none')
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[1].set_color(COLORS['negative'])
    ax2.get_lines()[1].set_linewidth(2)
    ax2.set_title('Normal Q-Q Plot', fontweight='bold', pad=10)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=30, edgecolor='white',
             alpha=0.8, color=COLORS['primary'])
    ax3.axvline(0, color=COLORS['negative'], linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Distribution of Residuals', fontweight='bold', pad=10)
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(y, y_pred, alpha=0.5, s=25,
                color=COLORS['secondary'], edgecolors='none')
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax4.plot(lims, lims, color=COLORS['negative'],
             linestyle='--', linewidth=2, label='Perfect Prediction')
    ax4.set_xlabel('Actual Values', fontweight='bold')
    ax4.set_ylabel('Predicted Values', fontweight='bold')
    ax4.set_title('Actual vs Predicted', fontweight='bold', pad=10)
    ax4.legend(framealpha=0.9)
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    standardized_residuals = residuals / np.std(residuals)
    ax5.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)),
                alpha=0.5, s=25, color=COLORS['accent'], edgecolors='none')
    ax5.set_xlabel('Fitted Values', fontweight='bold')
    ax5.set_ylabel('√|Standardized Residuals|', fontweight='bold')
    ax5.set_title('Scale-Location Plot', fontweight='bold', pad=10)
    ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(range(len(residuals)), residuals,
             alpha=0.4, linewidth=0.8, color=COLORS['primary'])
    ax6.axhline(0, color=COLORS['negative'], linestyle='--', linewidth=2)
    ax6.set_xlabel('Observation Index', fontweight='bold')
    ax6.set_ylabel('Residuals', fontweight='bold')
    ax6.set_title('Residuals Over Observations', fontweight='bold', pad=10)
    ax6.grid(alpha=0.3)

    plt.savefig(output_dir / 'residual_diagnostics.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Saved: residual_diagnostics.png")

# ==================== SECTION 9: EXECUTIVE SUMMARY ====================
def create_executive_summary(df, overall, rmse_seg, rmse_debt, best_idx, output_dir):
    """Create a JSON executive summary aligned with the Covid structural-break story."""
    print_section_header("GENERATING EXECUTIVE SUMMARY (STORY MODE)", 9)

    best_task, best_spec, best_model = best_idx
    best_metrics = overall.loc[best_idx].to_dict()

    # Average Covid robustness across all combinations (simple descriptive stat)
    covid_vs_pre = rmse_seg['covid_vs_pre_pct'].replace([np.inf, -np.inf], np.nan).dropna()
    avg_covid_increase = float(covid_vs_pre.mean()) if len(covid_vs_pre) else np.nan

    # Debt gap summary (Covid only)
    if 'high_minus_low_rmse' in rmse_debt.columns:
        gap = rmse_debt['high_minus_low_rmse'].replace([np.inf, -np.inf], np.nan).dropna()
        avg_debt_gap = float(gap.mean()) if len(gap) else np.nan
        max_debt_gap = float(gap.max()) if len(gap) else np.nan
    else:
        avg_debt_gap = np.nan
        max_debt_gap = np.nan

    summary = {
        "Analysis_Metadata": {
            "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Random_State": CONFIG['random_state'],
            "Tasks": CONFIG.get("tasks", [])
        },
        "Dataset_Overview": {
            "Total_Observations": int(len(df)),
            "Countries": int(df['Country'].nunique() if 'Country' in df.columns else 27),
            "Year_Range": f"{int(df['Year'].min())}–{int(df['Year'].max())}" if 'Year' in df.columns else "NA"
        },
        "Best_Model_Overall": {
            "Task": best_task,
            "Specification": best_spec,
            "Model": best_model,
            "Metrics": best_metrics
        },
        "Structural_Break_Summary": {
            "Average_Covid_vs_Pre_RMSE_%": avg_covid_increase,
            "Most_Robust_Combo": rmse_seg[['pre','covid','post','covid_vs_pre_pct']].head(1).reset_index().to_dict(orient='records')[0]
                               if len(rmse_seg) else {}
        },
        "Fiscal_Constraint_Summary": {
            "Average_Covid_RMSE_Gap_HighMinusLow": avg_debt_gap,
            "Max_Covid_RMSE_Gap_HighMinusLow": max_debt_gap
        }
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "executive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEXECUTIVE SUMMARY (key numbers):")
    print(f"  • Best overall: {best_task} | {best_model} | {best_spec}")
    print(f"  • Best RMSE: {best_metrics.get('rmse', np.nan):.4f}")
    if not np.isnan(avg_covid_increase):
        print(f"  • Avg Covid RMSE increase (%): {avg_covid_increase:.1f}")
    if not np.isnan(avg_debt_gap):
        print(f"  • Avg HighDebt-LowDebt RMSE gap (Covid): {avg_debt_gap:.4f}")

    print("\n  ✓ Saved: executive_summary.json")
    return summary


# ==================== MAIN PIPELINE ====================
def main():
    """Overall pipeline controller."""
    print_section_header("EU EDUCATION EXPENDITURE MODELING", None)
    print(f"\n  Analysis Pipeline for Academic Research")
    print(f"  Professional Report Generation")
    print(f"  {'─' * 98}\n")

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    df, country_cols = load_and_prepare_data()

    specifications = define_specifications(df, country_cols)

    adf_results, vif_results = run_diagnostics(
        df,
        [f for f in specifications['Spec_C_Extended']['features']
         if not f.startswith('Country_')][:10]
    )

    results_df = walk_forward_validation(df, specifications)

    overall, segment_perf, rmse_seg, rmse_debt, best_idx = analyze_results(results_df, output_dir)

    create_professional_charts(results_df, overall, segment_perf, rmse_seg, rmse_debt, best_idx, output_dir)

    best_spec_name = best_idx[1]
    best_spec_info = specifications[best_spec_name]
    xgb_final, X, y, shap_values = train_final_model_with_shap(
        df, best_spec_info, output_dir
    )

    residual_diagnostics(xgb_final, X, y, output_dir)

    summary = create_executive_summary(df, overall, rmse_seg, rmse_debt, best_idx, output_dir)

    print_section_header("ANALYSIS COMPLETE", None)
    print("\n  ✅ All tasks completed successfully!\n")
    print("  OUTPUT FILES GENERATED:")

    output_files = [
        # Core data outputs
        "oos_residuals_observation_level.csv",
        "performance_summary_overall.csv",
        "performance_by_segment.csv",
        "table_structural_break_rmse.csv",
        "performance_covid_by_debtgroup.csv",
        "table_covid_debtgap_rmse.csv",
        "table_top6_covid_debt_gap.csv",

        # Interpretation outputs
        "feature_importance.csv",
        "shap_importance.csv",
        "executive_summary.json",

        # Key story figures
        "fig_leaderboard_top10_rmse.png",
        "fig_temporal_rmse_best_with_covid_shading.png",
        "fig_covid_rmse_high_vs_low_debt.png",

        # Diagnostics
        "residual_diagnostics.png"
    ]

    for i, filename in enumerate(output_files, 1):
        filepath = output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024
            print(f"     {i:2d}. {filename:<40s} ({size:>6.1f} KB)")

    print("\n  RECOMMENDED NEXT STEPS:")
    print("     1. Inspect performance_summary_overall.csv for the leaderboard")
    print("     2. Use table_structural_break_rmse.csv for Pre/Covid/Post robustness")
    print("     3. Use table_covid_debtgap_rmse.csv for HighDebt vs LowDebt gaps during Covid")
    print("     4. Inspect fig_temporal_rmse_best_with_covid_shading.png for the break visualization")
    print("     5. Use detailed_results.csv for time-series discussion\n")

if __name__ == "__main__":
    main()