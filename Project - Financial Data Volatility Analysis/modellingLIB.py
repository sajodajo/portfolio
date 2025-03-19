import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, normaltest, norm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import t, probplot
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import streamlit as st



def calcColumns(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df.drop(columns='Dividend Paid', inplace=True)

    df['Log_Return'] = (np.log(df['Opening Price'] / df['Opening Price'].shift(1))*100)
    df.dropna(inplace=True)

    # Compute absolute log-returns
    df['Abs_Log_Return'] = np.abs(df['Log_Return'])

    # Compute rolling 60-day standard deviation
    df['Rolling_Std'] = df['Log_Return'].rolling(window=60).std()

    # LOESS smoothing (LOWESS from statsmodels)
    lowess = sm.nonparametric.lowess
    global smoothed_abs
    smoothed_abs = lowess(df['Abs_Log_Return'], df.index, frac=0.03)  # frac=0.03 controls smoothness

    return df

def plotTS(df,companyName):
    # Plot time series
    fig, ax = plt.subplots(figsize=(12, 6))
    startYear = df.index[0].strftime("%b. %Y")
    endYear = df.index[-1].strftime("%b. %Y")
    fCompany = f"{companyName.title()}"

    ax.plot(df.index, df['Opening Price'], color='#389cfc', alpha=0.6, label='Opening Price')
    ax.set_ylabel(f"{fCompany.title()} Share Price")
    ax.set_title(f"{fCompany.title()} Share Price ({startYear}-{endYear})")
    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig, ax

def plotLR(df, smoothed_abs,companyName):
    # Plot time series
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

    startYear = df.index[0].strftime("%b. %Y")
    endYear = df.index[-1].strftime("%b. %Y")

    fCompany = f"{companyName.title()}"

    # Plot log-returns
    top_ax = axes[0]
    top_ax.plot(df.index, df['Log_Return'], color='#389cfc', alpha=0.6, label='Log-Returns')
    top_ax.set_ylabel("Log-Returns (%)")
    top_ax.set_title(f"{fCompany}\nLog-Returns ({startYear}-{endYear})")
    top_ax.legend()

    # Plot absolute log-returns with LOESS and rolling std
    bottom_ax = axes[1]
    bottom_ax.plot(df.index, df['Abs_Log_Return'], color='gray', alpha=0.5, label='Absolute Log-Returns')
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%y"))  
    bottom_ax.plot(df.index, df['Rolling_Std'], color='#389cfc', label='60-Day Rolling Std')
    bottom_ax.plot(df.index, smoothed_abs[:, 1], color='black', label='LOESS Smoothed Abs Returns')
    bottom_ax.set_ylabel(f"{fCompany}\nAbsolute Log-Returns (%)")
    bottom_ax.set_title(f"{fCompany}\nAbsolute Log-Returns with LOESS and Rolling Std\n({startYear}-{endYear})")
    bottom_ax.legend()

    #plt.tight_layout()
    #plt.show()

    return fig, axes

def fitNormalDist(df,companyName):
    # Compute statistics
    mean_return = np.mean(df['Log_Return'])
    variance_return = np.var(df['Log_Return'])
    skewness_return = skew(df['Log_Return'])
    kurtosis_return = kurtosis(df['Log_Return'])

    # Normality test (D'Agostino and Pearson’s test)
    stats_test, p_value = normaltest(df['Log_Return'])
    normality_result = "Normal" if p_value > 0.05 else "Not Normal"

    fCompany = f"{companyName.title()}"

    statsDict = {"Mean": f"{mean_return:.4f}","Variance": f"{variance_return:.4f}","Skewness": f"{skewness_return:.4f}","Kurtosis": f"{kurtosis_return:.4f}", "Normality Test p-value": f"{p_value:.4f} ({normality_result})"}

    # Plot histogram with overlaid normal distribution
    fig = plt.figure(figsize=(10, 5))
    count, bins, _ = plt.hist(df['Log_Return'], bins=50, alpha=0.7, color='#389cfc', edgecolor='black', density=True)

    # Compute normal distribution curve
    x = np.linspace(bins[0], bins[-1], 100)
    pdf = norm.pdf(x, mean_return, np.sqrt(variance_return))
    plt.plot(x, pdf, color='red', lw=2, label=f'Normal Dist (μ={mean_return:.2f}, σ²={variance_return:.2f})')

    plt.xlabel("Log-Returns (%)")
    plt.ylabel("Density")
    plt.title(f"Histogram of {fCompany} Log-Returns (2019-2024) with Normal Distribution")
    plt.legend()
    plt.grid()
    plt.show()

    return fig, statsDict


def fitTDist(df,companyName):
    # Fit a t-distribution to the log-returns
    params = t.fit(df['Log_Return'])  # Direct fitting

    # Extract fitted parameters
    global df_t, loc_t, scale_t
    df_t, loc_t, scale_t = params

    # Compute variance and kurtosis of the fitted t-distribution
    variance_t = (df_t / (df_t - 2)) * (scale_t ** 2) if df_t > 2 else np.nan
    kurtosis_t = (6 / (df_t - 4)) if df_t > 4 else np.inf  # Infinite for df <= 4

    fCompany = f"{companyName.title()}"

    # Print fitted parameters and statistics
    dof = print(f"Degrees of Freedom: {df_t:.4f}")
    loc = print(f"Location: {loc_t:.4f}")
    scale = print(f"Scale: {scale_t:.4f}")
    variance = print(f"Variance: {variance_t:.4f}")
    kurtosis = print(f"Kurtosis: {'Infinite' if np.isinf(kurtosis_t) else f'{kurtosis_t:.4f}'}")


    # Plot histogram with normal and t-distribution curves
    fig = plt.figure(figsize=(10, 5))
    count, bins, _ = plt.hist(df['Log_Return'], bins=50, alpha=0.7, color='#389cfc', edgecolor='black', density=True)

    # Compute normal and t-distribution curves
    x = np.linspace(bins[0], bins[-1], 100)
    pdf_norm = norm.pdf(x, np.mean(df['Log_Return']), np.std(df['Log_Return']))
    pdf_t = t.pdf(x, df_t, loc=loc_t, scale=scale_t)

    fCompany = f"{companyName.title()}"

    plt.plot(x, pdf_norm, color='red', lw=2, label=f'Normal Dist (μ={np.mean(df["Log_Return"]):.2f}, σ²={np.var(df["Log_Return"]):.2f})')
    plt.plot(x, pdf_t, color='green', lw=2, label=f't-Dist (df={df_t:.2f}, scale={scale_t:.2f})')

    plt.xlabel("Log-Returns (%)")
    plt.ylabel("Density")
    plt.title(f"Histogram of {fCompany} Log-Returns with Normal and t-Distributions")
    plt.legend()
    plt.grid()
    plt.show()

    statsDict = {"Degrees of Freedom": f"{df_t:.4f}","Location": f"{loc_t:.4f}","Scale": f"{scale_t:.4f}","Variance": f"{variance_t:.4f}","Kurtosis": f"{'Infinite' if np.isinf(kurtosis_t) else f'{kurtosis_t:.4f}'}"}


    return fig, statsDict


def autocorrChecks(df):
    nlags = int(len(df) ** (1/2))
    # Compute autocorrelation of log-returns
    acf = sm.tsa.acf(df['Log_Return'], nlags=nlags, fft=False)
    pacf = sm.tsa.pacf(df['Log_Return'], nlags=nlags)

    # Plot ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot ACF
    sm.graphics.tsa.plot_acf(df['Log_Return'], lags=nlags, ax=axes[0])
    axes[0].set_title("Autocorrelation Function (ACF)")

    # Plot PACF
    sm.graphics.tsa.plot_pacf(df['Log_Return'], lags=nlags, ax=axes[1])
    axes[1].set_title("Partial Autocorrelation Function (PACF)")

    return fig, axes


def ljungboxTest(time_series):
    numLags = len(time_series) ** (1/2)
    ljung_box_test = acorr_ljungbox(time_series, lags=[numLags], return_df=True)
    return ljung_box_test


def ljung_box_test_with_interpretation(time_series, max_lags=None):
    if max_lags is None:
        max_lags = int(len(time_series) ** 0.5)
    
    test_results = acorr_ljungbox(time_series, lags=[max_lags], return_df=True)
    
    lb_stat = test_results["lb_stat"].values[0]
    lb_pvalue = test_results["lb_pvalue"].values[0]

    if lb_pvalue < 0.05:
        interpretation = (f"The Ljung-Box test suggests significant autocorrelation up to {max_lags} lags "
                          f"(lb_stat = {lb_stat:.4f}, p-value = {lb_pvalue:.4f}). "
                          "This indicates that past values influence future values.")
    else:
        interpretation = (f"The Ljung-Box test does not detect significant autocorrelation up to {max_lags} lags "
                          f"(lb_stat = {lb_stat:.4f}, p-value = {lb_pvalue:.4f}). "
                          "The series may be behaving like white noise.")

    return test_results, interpretation





def visVolatility(df,returns, garch_fit):
    # Get the conditional volatility (sigma_t) and standardized residuals
    conditional_volatility = garch_fit.conditional_volatility
    absolute_log_returns = np.abs(returns)

    # Combined plot with two subplots stacked vertically
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Log returns with shaded ±2 conditional standard deviations
    axs[0].plot(returns, label='Log Returns', color='#389cfc')
    upper_bound = 2 * conditional_volatility
    lower_bound = -2 * conditional_volatility
    axs[0].fill_between(df.index, lower_bound, upper_bound, color='red', alpha=0.2, label='±2 Conditional Std Dev')
    axs[0].set_title('Log Returns with Conditional Std Deviations (Shaded ±2σ)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Log Returns / Volatility')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Conditional Std Dev vs. Absolute Log Returns
    axs[1].plot(absolute_log_returns, label='|Log Returns|', color='gray', alpha=0.2)
    axs[1].plot(conditional_volatility, label='Conditional Std Dev (GARCH)', color='#389cfc')
    axs[1].set_title('Conditional Std Dev vs Absolute Log Returns')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Values')
    axs[1].legend()
    axs[1].grid(True)

    return fig, axs


def adf_test_summary(df, column='Log_Return', signif=0.05):
    """Performs ADF test and returns results as a DataFrame and interpretation text."""
    series = df[column].dropna()
    result = adfuller(series, maxlag=0, autolag=None)
    
    test_statistic, p_value, used_lags, n_obs, critical_values = result
    is_stationary = p_value < signif

    # Create a results DataFrame
    adf_results = pd.DataFrame({
        "Test Statistic": [test_statistic],
        "p-value": [p_value],
        "Used Lags": [used_lags],
        "Number of Observations": [n_obs],
        "Stationary": [is_stationary]
    })
    
    for key, value in critical_values.items():
        adf_results[f"Critical Value ({key})"] = [value]

    # Interpretation text
    interpretation = ""
    if p_value < 0.01:
        interpretation += "✅ The series is **STATIONARY** at the **1% significance level** (Strong rejection of H0).\n"
    elif p_value < 0.05:
        interpretation += "✅ The series is **STATIONARY** at the **5% significance level** (Moderate rejection of H0).\n"
    elif p_value < 0.10:
        interpretation += "⚠️ The series is **WEAKLY STATIONARY** at the **10% significance level** (Weak rejection of H0).\n"
    else:
        interpretation += "❌ The series is **NON-STATIONARY** (Fail to reject H0).\n"

    for level, value in critical_values.items():
        if test_statistic < value:
            interpretation += f"\n✔️ Test Statistic is below the **{level}** critical value ({value:.4f}). The series is **STATIONARY** at {level} level.\n"
        else:
            interpretation += f"\n❌ Test Statistic is above the **{level}** critical value ({value:.4f}). The series is **NOT stationary** at {level} level.\n"

    return adf_results, interpretation


### MODELS ###

## ARCH Model
def fit_arch_model(returns):
    model = arch_model(returns, vol="ARCH", p=1)
    resultARCH = model.fit(disp="off")
    bicARCH = resultARCH.bic
    aicARCH = resultARCH.aic
    llARCH = resultARCH.loglikelihood
    return resultARCH, bicARCH, aicARCH, llARCH

## GARCH Model
def fit_garch_model(returns):
    model = arch_model(returns, vol="GARCH", p=1, q=1)
    resultGARCH = model.fit(disp="off")
    bicGARCH = resultGARCH.bic
    aicGARCH = resultGARCH.aic
    llGARCH = resultGARCH.loglikelihood
    return resultGARCH, bicGARCH, aicGARCH, llGARCH

## GJR-GARCH Model
def fit_gjr_garch_model(returns):
    model = arch_model(returns, vol="GARCH", p=1, q=1, o=1)
    resultGJRGARCH = model.fit(disp="off")
    bicGJRGARCH = resultGJRGARCH.bic
    aicGJRGARCH = resultGJRGARCH.aic
    llGJRGARCH = resultGJRGARCH.loglikelihood
    return resultGJRGARCH, bicGJRGARCH, aicGJRGARCH, llGJRGARCH

## EGARCH Model
def fit_egarch_model(returns):
    model = arch_model(returns, vol="EGARCH", p=1, q=1)
    resultEGARCH = model.fit(disp="off")
    bicEGARCH = resultEGARCH.bic
    aicEGARCH = resultEGARCH.aic
    llEGARCH = resultEGARCH.loglikelihood
    return resultEGARCH, bicEGARCH, aicEGARCH, llEGARCH

## ARCH-t Model
def fit_arch_t_model(returns):
    model = arch_model(returns, vol="ARCH", p=1,dist='t')
    resultARCHt = model.fit(disp="off")
    bicARCHt = resultARCHt.bic
    aicARCHt = resultARCHt.aic
    llARCHt = resultARCHt.loglikelihood
    return resultARCHt, bicARCHt, aicARCHt, llARCHt


## GARCH-t Model
def fit_garch_t_model(returns):
    model = arch_model(returns, vol="GARCH", p=1, q=1,dist='t')
    resultGARCHt = model.fit(disp="off")
    bicGARCHt = resultGARCHt.bic
    aicGARCHt = resultGARCHt.aic
    llGARCHt = resultGARCHt.loglikelihood
    return resultGARCHt, bicGARCHt, aicGARCHt, llGARCHt

## GJR-GARCH-t Model
def fit_gjr_garch_t_model(returns):
    model = arch_model(returns, vol="GARCH", p=1, q=1, o=1,dist='t')
    resultGJRGARCHt = model.fit(disp="off")
    bicGJRGARCHt = resultGJRGARCHt.bic
    aicGJRGARCHt = resultGJRGARCHt.aic
    llGJRGARCHt = resultGJRGARCHt.loglikelihood
    return resultGJRGARCHt, bicGJRGARCHt, aicGJRGARCHt, llGJRGARCHt

## EGARCH-t Model
def fit_egarch_t_model(returns):
    model = arch_model(returns, vol="EGARCH", p=1, q=1,dist='t')
    resultEGARCHt = model.fit(disp="off")
    bicEGARCHt = resultEGARCHt.bic
    aicEGARCHt = resultEGARCHt.aic
    llEGARCHt = resultEGARCHt.loglikelihood
    return resultEGARCHt, bicEGARCHt, aicEGARCHt, llEGARCHt


def modelChoice(comparison_df):
    best_models = {
        "AIC": comparison_df["AIC"].idxmin(),  
        "BIC": comparison_df["BIC"].idxmin(), 
        "Log-Likelihood": comparison_df["Log-Likelihood"].idxmax() 
    }

    model_scores = comparison_df.index.to_series().apply(lambda x: sum(1 for metric in best_models.values() if metric == x))

    global best_model
    best_model = model_scores.idxmax()
    best_count = model_scores.max()

    # Highlight best models in each criterion
    highlight_df = comparison_df.copy()
    highlight_df["Best AIC"] = comparison_df.index == best_models["AIC"]
    highlight_df["Best BIC"] = comparison_df.index == best_models["BIC"]
    highlight_df["Best Log-Likelihood"] = comparison_df.index == best_models["Log-Likelihood"]

    st.write("### Best Model by Each Criterion")
    st.dataframe(highlight_df)

    # Decision logic
    if best_count == 3:
        message = f"🏆 {best_model} is the best model as it performs best in all three criteria."
    elif best_count == 2:
        message = f"✅ {best_model} is the most favorable model, leading in 2 out of 3 criteria."
    else:
        message = "📊 No single model is the best in all criteria. Consider trade-offs:\n"
        for metric, best in best_models.items():
            message += f" - {metric}: {best}\n"

    return message, best_model


def vizBestModel(df,bestResult):
    print("\nModel Summary:")
    print(bestResult.summary())

    # Plot GARCH model results
    fig = bestResult.plot()

    # Customize line colors
    ax = fig.axes[0]  # Access the main plot axis
    for line in ax.get_lines():
        line.set_color('#389cfc')  # Change all lines to red

    return fig


def visVolatility(df,returns, garch_fit):
    # Get the conditional volatility (sigma_t) and standardized residuals
    conditional_volatility = garch_fit.conditional_volatility
    absolute_log_returns = np.abs(returns)

    # Combined plot with two subplots stacked vertically
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Log returns with shaded ±2 conditional standard deviations
    axs[0].plot(returns, label='Log Returns', color='#389cfc')
    upper_bound = 2 * conditional_volatility
    lower_bound = -2 * conditional_volatility
    axs[0].fill_between(df.index, lower_bound, upper_bound, color='red', alpha=0.2, label='±2 Conditional Std Dev')
    axs[0].set_title('Log Returns with Conditional Std Deviations (Shaded ±2σ)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Log Returns / Volatility')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Conditional Std Dev vs. Absolute Log Returns
    axs[1].plot(absolute_log_returns, label='|Log Returns|', color='gray', alpha=0.2)
    axs[1].plot(conditional_volatility, label='Conditional Std Dev (GARCH)', color='#389cfc')
    axs[1].set_title('Conditional Std Dev vs Absolute Log Returns')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Values')
    axs[1].legend()
    axs[1].grid(True)

    return fig, axs


def residualAnalysis(garch_fit):
    # Compute standardized residuals
    std_residuals = garch_fit.resid / garch_fit.conditional_volatility
    std_residuals = std_residuals.dropna()

    # Set up 2x2 residual plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Standardized residual time series
    sns.lineplot(x=std_residuals.index, y=std_residuals, ax=axes[0, 0], color='#389cfc')
    axes[0, 0].set_title("Standardized Residuals Time Series")
    axes[0, 0].set_ylabel("Standardized Residuals")
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.6)

    # 2. ACF of standardized residuals
    plot_acf(std_residuals, ax=axes[0, 1], lags=len(std_residuals) - 1)
    axes[0, 1].set_title("ACF of Standardized Residuals")

    # 3. ACF of squared standardized residuals
    plot_acf(std_residuals, ax=axes[1, 0], lags=len(std_residuals) - 1)
    axes[1, 0].set_title("ACF of Squared Standardized Residuals")

    # 4. QQ-plot with estimated t-distribution
    df_t, loc_t, scale_t = t.fit(std_residuals)
    x = np.linspace(min(std_residuals), max(std_residuals), 100)
    pdf_t = t.pdf(x, df_t, loc=loc_t, scale=scale_t)

    axes[1, 1].hist(std_residuals, bins=50, density=True, alpha=0.6, color='#389cfc', edgecolor='black', label="Residuals")
    axes[1, 1].plot(x, pdf_t, color='red', lw=2, label=f't-Dist (df={df_t:.2f})')
    axes[1, 1].set_title("Histogram of Standardized Residuals with t-Distribution")
    axes[1, 1].legend()

    return fig, axes


def VaR(returns,companyName):
  
    fCompany = f"{companyName.title()}"

    # Confidence levels for VaR
    global confidence_levels
    confidence_levels = np.linspace(0.975, 0.9999, 100)

    # Compute VaR for different confidence levels
    global VaR_hist, VaR_norm, VaR_t
    VaR_hist = [np.percentile(returns, (1 - alpha) * 100) for alpha in confidence_levels]
    VaR_norm = [norm.ppf(1 - alpha, loc=np.mean(returns), scale=np.std(returns)) for alpha in confidence_levels]
    global df_t, loc_t, scale_t
    df_t, loc_t, scale_t = t.fit(returns)
    VaR_t = [loc_t + scale_t * t.ppf(1 - alpha, df_t) for alpha in confidence_levels]

    # Plot VaR estimates as a function of confidence level
    fig = plt.figure(figsize=(10, 6))
    plt.plot(confidence_levels * 100, VaR_hist, label='Historical VaR', linestyle='dashed', color='black')
    plt.plot(confidence_levels * 100, VaR_norm, label='Normal VaR', linestyle='dotted', color='red')
    plt.plot(confidence_levels * 100, VaR_t, label='t-Distribution VaR', linestyle='solid', color='#389cfc')

    plt.xlabel("Confidence Level (%)")
    plt.ylabel("Log-Returns")
    plt.title(f"{fCompany} VaR Estimates Across Confidence Levels")
    plt.legend()
    plt.grid()
    plt.show()

    return fig, confidence_levels, VaR_hist, VaR_norm, VaR_t


def expectedShortfall(confidence_levels,companyName,returns):
    # Compute Expected Shortfall (ES) using the proper formulas
    mean_return, std_return = returns.mean(), returns.std()
    phi_norm = norm.pdf(norm.ppf(confidence_levels))
    ES_norm = mean_return - std_return * (phi_norm / (1 - confidence_levels))

    fCompany = f"{companyName.title()}"

    # Compute t-Distribution Expected Shortfall
    t_alpha = t.ppf(confidence_levels, df_t)
    t_pdf_alpha = t.pdf(t_alpha, df_t)
    ES_t = loc_t - scale_t * (t_pdf_alpha / (1 - confidence_levels)) * (df_t + t_alpha**2) / (df_t - 1)

    # Compute Historical Expected Shortfall (ES) directly from data
    ES_hist = [returns[returns <= VaR_hist[i]].mean() for i in range(len(confidence_levels))]

    # Plot Expected Shortfall (ES) estimates as a function of confidence level
    fig = plt.figure(figsize=(10, 6))
    plt.plot(confidence_levels * 100, ES_hist, label='Historical ES', linestyle='dashed', color='black')
    plt.plot(confidence_levels * 100, ES_norm, label='Normal ES', linestyle='dotted', color='red')
    plt.plot(confidence_levels * 100, ES_t, label='t-Distribution ES', linestyle='solid', color='#389cfc')
    plt.xlabel("Confidence Level (%)")
    plt.ylabel("Log-Returns")
    plt.title(f"{fCompany} Expected Shortfall (ES) Estimates Across Confidence Levels")
    plt.legend()
    plt.grid()
    plt.show()

    return fig


def dynamicRM(garch_fit,companyName,returns):
    # Extract conditional volatility and standardized residuals
    std_residuals = garch_fit.resid / garch_fit.conditional_volatility
    std_residuals = std_residuals.dropna()
    cond_volatility = garch_fit.conditional_volatility.dropna()

    # Fit t-distribution to standardized residuals
    df_t, loc_t, scale_t = t.fit(std_residuals)

    fCompany = f"{companyName.title()}"

    # Compute dynamic VaR at 95% and 99%
    VaR_95 = -scale_t * t.ppf(0.05, df_t) * cond_volatility
    VaR_99 = -scale_t * t.ppf(0.01, df_t) * cond_volatility

    # Plot time series of negative log-returns with dynamic VaR
    fig = plt.figure(figsize=(12, 6))
    plt.plot(-returns, label="Negative Log-Returns", color='blue', alpha=0.6)
    plt.plot(VaR_95, label="95% Dynamic VaR", linestyle='dashed', color='#389cfc')
    plt.plot(VaR_99, label="99% Dynamic VaR", linestyle='solid', color='black')

    plt.xlabel("Date")
    plt.ylabel("Log-Returns")
    plt.title(f"{fCompany} Negative Log-Returns with Dynamic VaR Estimates")
    plt.legend()
    plt.grid()
    plt.show()

    return fig