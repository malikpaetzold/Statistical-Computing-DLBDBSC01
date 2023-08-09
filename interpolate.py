import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# reference: https://stackoverflow.com/questions/63097829/debugging-numpy-visibledeprecationwarning-ndarray-from-ragged-nested-sequences

# from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

data = pd.read_csv("clean_data/school_enrollment_tertiary_gross.csv")
indicator = "school_enrollment_tertiary_gross"

print(data.head(10))

def interpolate(country_indx: str):
    interpolation_required = False
    series = data.iloc[country_indx]
    country_name = series.loc["Country Name"]

    values = list(series.drop(["Unnamed: 0", "Country Name", "Country Code", "Unnamed: 67"]))
    X, y = [], []

    for indx, val in enumerate(values):
        if pd.isna(val): continue
        y.append(val)
        X.append(indx)

    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)


    regressor = LinearRegression()
    # try:
    regressor.fit(X, y)
    # regressor2 = KNeighborsRegressor(n_neighbors=3)
    # regressor2.fit(X, y)
    
    to_predict = np.arange(0.0, 63.0, 1.0)[:, np.newaxis]

    prediction = regressor.predict(to_predict)
    # prediction2 = regressor2.predict(to_predict)

    # merge original & interpolated values
    merged_values = []

    for indx, val in enumerate(values):
        if pd.isna(val):
            value = prediction[indx][0]
            interpolation_required = True
        else:
            value = val
        
        if value < 0: value = -1
        merged_values.append(value)

    if interpolation_required:
        plt.plot(merged_values, color="red")
        plt.plot(values)
        x_location, x_label = plt.xticks()
        x_label = [1950 + i*10 for i, _ in enumerate(x_label)]
        x_location = list(x_location)
        x_location.pop(0)
        x_label.pop(0)
        plt.xticks(x_location, x_label)
        plt.title(f"GDP per capita (current US$) of {country_name}")
        plt.savefig(f"interpolate_plots/{indicator}_{country_indx}_{country_name}.png", bbox_inches="tight")
        plt.close()
    
    return list(merged_values), str(country_name)

    # except KeyboardInterrupt: quit()
    # except Exception as e:
    #     print(country_indx, country_name)
    #     print("X - ", X)
    #     print("y - ", y)
    #     print("Error: ", e)

full_data = {}

for indx, row in data.iterrows():
    country_data, country_name = interpolate(indx)
    full_data[country_name] = country_data

out = pd.DataFrame(data=full_data)
print(out)

out.to_csv(f"interpolated_data/{indicator}.csv")