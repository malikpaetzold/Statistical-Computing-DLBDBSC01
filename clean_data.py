import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "download2/API_SL.UEM.TOTL.FE.ZS_DS2_en_csv_v2_5874438.csv"
METADATA_PATH = "download2/Metadata_Country_API_SL.UEM.TOTL.FE.ZS_DS2_en_csv_v2_5874438.csv"
OUT_PATH = "clean_data/unemployment_female.csv"

data = pd.read_csv(DATA_PATH, skiprows=4)
indicator_name = OUT_PATH.split("/")[-1].replace(".csv", "")

data = data.drop(["Indicator Name", "Indicator Code"], axis=1)

print(data.dtypes)

# from prettytable import PrettyTable
# x = PrettyTable()
# x.add_column("Country Name", data["Country Name"])
# x.add_column("NaN values (abs)", data.isnull().sum(1))
# x.add_column("NaN values (rel)", round(data.isnull().sum(1) / (len(data.keys())-2), 2))
# print(x)

to_drop = []

for indx, row in data.iterrows():
    data_row = row.drop(["Country Name", "Country Code", "Unnamed: 67"])
    data_row = data_row.fillna(-1).to_list()
    if data_row.count(-1) / len(data_row) > 0.7:
        to_drop.append(row["Country Name"])

print("to drop:", to_drop)
print("number of countries to drop: ", len(to_drop))
with open(f"output/{indicator_name}_clean-dropped_countries.txt", "w") as f:
    for c in to_drop:
        f.write(c + "\n")

# remove countries with to many missing values
clean = data

to_drop_indx = []
for elem in to_drop:
    drop_indx = clean.loc[clean["Country Name"] == elem].index[0]
    to_drop_indx.append(drop_indx)

clean.drop(index=to_drop_indx, inplace=True)
# get aggregates & other groups
country_meta = pd.read_csv(METADATA_PATH)

non_countries = []

for indx, row in country_meta.iterrows():
    # NaN is float in Pandas
    if type(row.Region) is float:
        if row["Country Code"] in ["WLD", "LIC", "LMC", "LMY", "MIC", "UMC"]: continue
        non_countries.append([row["Country Code"], row["SpecialNotes"]])

# for nc in non_countries:
#     print(nc)

# remove non countris
to_drop_indx = []
for elem in non_countries:
    try:
        drop_indx = clean.loc[clean["Country Code"] == elem[0]].index[0]
        to_drop_indx.append(drop_indx)
    except Exception as e:
        print(indx, elem, e)

clean.drop(index=to_drop_indx, inplace=True)

# print(clean.head(10))

clean.to_csv(OUT_PATH)