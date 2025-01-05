import pandas as pd 
import numpy as np  
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import argrelextrema
from scipy.fft import ifft
from scipy.optimize import curve_fit

equity = pd.read_excel("/Users/tangruiqi/Desktop/Investment/Macro asset data.xlsx")

# convert to monthly data
def convert_montly_data(data:pd.DataFrame):
    data = data.set_index("Date")
    data = data.resample("ME").last()
    data_pct = data.pct_change(periods=12)
    return data_pct.dropna()

def standard_pca(data):
    principal = None
    n_length = data.shape[1]
    scaling=StandardScaler()
    scaling.fit(data)
    Scaled_data=scaling.transform(data)
    principal=PCA(n_components=n_length)
    pca_compont = principal.fit_transform(Scaled_data)

    col = []

    for i in range(1,n_length+1):
        temp_col = str(i) + "_Component"
        col.append(temp_col)
    pca_compont = pd.DataFrame(pca_compont, columns=col,index=data.index)

    return pca_compont, principal

equity = pd.read_excel("/Users/tangruiqi/Desktop/Investment/Macro asset data.xlsx")
commodity = pd.read_excel("/Users/tangruiqi/Desktop/Investment/Macro asset data.xlsx", sheet_name="Commodity")
commodity = commodity.iloc[:,0:-2]
treasure = pd.read_excel("/Users/tangruiqi/Desktop/Investment/Macro asset data.xlsx", sheet_name="Treasure")
treasure = treasure.iloc[:,0:-3]
treasure.iloc[:,1:] = treasure.iloc[:,1:] + 0.0000001
currency = pd.read_excel("/Users/tangruiqi/Desktop/Investment/Macro asset data.xlsx", sheet_name="Sheet4")

equity_pct = convert_montly_data(data = equity)
commodity_pct = convert_montly_data(data = commodity)
treasure_pct = convert_montly_data(data = treasure)
currency_pct = convert_montly_data(data = currency)

equity_compont, equity_pca = standard_pca(data=equity_pct)
commodity_compont, commodity_pca = standard_pca(data=commodity_pct)
treasure_compont, treasure_pca = standard_pca(data = treasure_pct)
currency_compont, currency_pca = standard_pca(data=currency_pct)


plt.figure(num = 1, figsize = (20,6))
plt.plot(equity_compont.index, equity_compont["1_Component"], label =  "Equity")
plt.plot(commodity_compont.index, commodity_compont["1_Component"], label =  "Commodity")
plt.plot(treasure_compont.index, treasure_compont["1_Component"], label =  "Treasure")
plt.plot(currency_compont.index, -currency_compont["1_Component"], label =  "Currency")
plt.legend()
plt.grid()
plt.title("Asset Cycle")
plt.show()


# find cycle

equity_compont = equity_compont[equity_compont.index>="2009-07-31"]
commodity_compont = commodity_compont[commodity_compont.index>="2009-07-31"]
treasure_compont = treasure_compont[treasure_compont.index>="2009-07-31"]
currency_compont = currency_compont[currency_compont.index>="2009-07-31"]


def standard_prepare(data:pd.DataFrame, asset:str):
    scaling=StandardScaler()
    scaling.fit(data)
    Scaled_data=scaling.transform(data)
    output = pd.DataFrame(Scaled_data, columns=[asset], index=data.index)
    return output.reset_index()

equity_z = standard_prepare(data = equity_compont[["1_Component"]], asset="Equity")
commodity_z = standard_prepare(data = commodity_compont[["1_Component"]], asset="Commodity")
treasure_z = standard_prepare(data = treasure_compont[["1_Component"]], asset="Treasure")
currency_z = standard_prepare(data = currency_compont[["1_Component"]], asset="Currency")

df_cycle = equity_z.merge(commodity_z)
df_cycle = df_cycle.merge(treasure_z)
df_cycle = df_cycle.merge(currency_z)

df_cycle["Cycle"] = df_cycle.iloc[:,1:].mean(axis=1)

fig, ax1 = plt.subplots(num = 1, figsize = (12,6))
plt.grid(1)
ax1.plot(equity_compont.index, equity_compont["1_Component"], label =  "Equity", alpha = 0.5)
ax1.plot(commodity_compont.index, commodity_compont["1_Component"], label =  "Commodity", alpha = 0.5)
ax1.plot(treasure_compont.index, treasure_compont["1_Component"], label =  "Treasure", alpha = 0.5)
ax1.plot(currency_compont.index, -currency_compont["1_Component"], label =  "Currency", alpha = 0.5)
ax1.set_ylabel("Non-Standard")
ax2 = ax1.twinx()
ax2.plot(df_cycle["Date"], df_cycle["Cycle"], label = "Cycle")
ax2.set_ylabel("Standard")
ax1.set_xlabel("Date")
plt.title("Asset Cycle")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 +labels2, loc = "upper right")
plt.show()

# frequency 46
# fourior series
def multi_harmonic_model(t, A1, B1, f1, C):
    return (A1 * np.cos(2 * np.pi * f1 * t) + B1 * np.sin(2 * np.pi * f1 * t) + C)

t_values = np.arange(len(df_cycle["Cycle"]))
y_values = df_cycle["Cycle"].to_numpy()

# Perform Fourier Transform to estimate dominant frequencies
fft_values_best = fft(y_values)
frequencies_best = np.fft.fftfreq(len(fft_values_best))
positive_frequencies = frequencies_best[1:len(frequencies_best)//2]
amplitudes = np.abs(fft_values_best[1:len(fft_values_best)//2])
dominant_indices = np.argsort(amplitudes)[-2:]
dominant_frequencies = positive_frequencies[dominant_indices]

initial_guess_freq_based = [1, 1, dominant_frequencies[1], 0]
params_freq_based, params_covariance_freq_based = curve_fit(multi_harmonic_model, t_values, y_values, p0=initial_guess_freq_based)

# Extract the estimated parameters for the frequency-based model
A1_freq, B1_freq, f1_freq, C_freq = params_freq_based

# Generate the fitted model for the updated STL detrended series
fitted_values_freq_based = multi_harmonic_model(t_values, A1_freq, B1_freq, f1_freq, C_freq)

df_cycle["Fourior"] = fitted_values_freq_based

plt.figure(num = 1, figsize = (20,6))
plt.plot(df_cycle["Date"],df_cycle["Cycle"], label =  "Market")
plt.plot(df_cycle["Date"],df_cycle["Fourior"], label =  "Market")
plt.legend()
plt.grid()
plt.title("Asset Cycle")
plt.show()