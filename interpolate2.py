import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# reference: https://stackoverflow.com/questions/63097829/debugging-numpy-visibledeprecationwarning-ndarray-from-ragged-nested-sequences

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

data = pd.read_csv("clean_data/school_enrollment_tertiary_gross.csv")
indicator = "school_enrollment_tertiary"

# print(data.head(10))

# with open("interpolate_log.txt", "w") as f:
#     f.write("--- interpolation logs ---")

def interpolate(country_indx: str, log_transform=False):
    series = data.iloc[country_indx]

    values = list(series.drop(["Unnamed: 0", "Country Name", "Country Code", "Unnamed: 67"]))
    X, y = [], []

    for indx, val in enumerate(values):
        if pd.isna(val): continue
        if log_transform: y.append(np.log(val))
        else: y.append(val)
        X.append(indx)
    
    if np.min(y) <= 0:
        for i in range(len(y)):
            if y[i] <= 0: y[i] = 0.0001

    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)


    regressor = LinearRegression()
    regressor.fit(X, y)
    
    # coefficient of determination
    y_pred = regressor.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    
    to_predict = np.arange(0.0, 63.0, 1.0)[:, np.newaxis]

    prediction = regressor.predict(to_predict)
    
    # inverse log transform
    if log_transform:
        for i in range(len(prediction)):
            if prediction[i] <= 0:
                prediction[i] = 0.001
        prediction = np.exp(prediction)
    
    return prediction, r2, mae, mape

def merge_values(country_indx):
    merged_values = []
    series = data.iloc[country_indx]
    country_name = series.loc["Country Name"]
    values = list(series.drop(["Unnamed: 0", "Country Name", "Country Code", "Unnamed: 67"]))
    use_log = False
    
    pred, r2, mae, mape = interpolate(country_indx)
    pred_log, r2_log, mae_log, mape_log = interpolate(country_indx, True)
    
    # use prediction with better r2 score
    if r2 > r2_log:
        prediction = pred
        r2, mae, mape = r2_log, mae_log, mape_log
        use_log = True
    else:
        prediction = pred_log

    for indx, val in enumerate(values):
        if pd.isna(val):
            value = prediction[indx][0]
        else:
            value = val
        
        # print(value)
        if value < 0: value = 0.001
        merged_values.append(value)

    f, ax = plt.subplots()
    plt.plot(merged_values, color="orangered", label="Interpolierte Werte")
    plt.plot(values, color="cornflowerblue", label="Bestehende Werte")
    x_location, x_label = plt.xticks()
    x_label = [1950 + i*10 for i, _ in enumerate(x_label)]
    x_location = list(x_location)
    x_location.pop(0)
    x_label.pop(0)
    plt.xticks(x_location, x_label)
    plt.legend()
    # if log_transform: plt.text(0.1, 0.9, "using log transform", family="sans-serif", transform=ax.transAxes)
    # plt.title(f"GDP per capita (current US$) of {country_name}")
    plt.title(f"School Enrollment (tertiary) of {country_name}") # change
    plt.savefig(f"interpolate_plots2/{indicator}/{indicator}_{country_indx}_{country_name}.png", bbox_inches="tight")
    plt.close()
    
    return list(merged_values), str(country_name), r2, mae, mape, use_log


full_data = {}
interpolation_logs = {}
r2s = []

for indx, row in data.iterrows():
    country_data, country_name, r2, mae, mape, use_log = merge_values(indx)
    full_data[country_name] = country_data
    interpolation_logs[country_name] = [r2, mae, mape, use_log]
    r2s.append(r2)

out = pd.DataFrame(data=full_data)
logs_out = pd.DataFrame(data=interpolation_logs)

print(pd.Series(r2s).mean())
print(pd.Series(r2s).median())
print(sum(r2s))

out.to_csv(f"interpolated_data/{indicator}.csv")
logs_out.to_csv(f"interpolated_data/logs-{indicator}.csv")