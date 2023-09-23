import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "download2/API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_5729318.csv"
METADATA_PATH = "download2/Metadata_Country_API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_5729318.csv"
OUT_PATH = "clean_data/electricity.csv"

data = pd.read_csv(DATA_PATH, skiprows=4)

data = data.drop(["Indicator Name", "Indicator Code"], axis=1)

print(data.dtypes)

from prettytable import PrettyTable
x = PrettyTable()
x.add_column("Country Name", data["Country Name"])
x.add_column("NaN values (abs)", data.isnull().sum(1))
x.add_column("NaN values (rel)", round(data.isnull().sum(1) / (len(data.keys())-2), 2))
print(x)

to_drop = []

for indx, row in data.iterrows():
    data_row = row.drop(["Country Name", "Country Code", "Unnamed: 67"])
    data_row = data_row.fillna(-1).to_list()
    if data_row.count(-1) / len(data_row) > 0.6:
        to_drop.append(row["Country Name"])

print("to drop:", to_drop)

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
        non_countries.append([row["Country Code"], row["SpecialNotes"]])

# remove non countris
to_drop_indx = []
for elem in non_countries:
    try:
        drop_indx = clean.loc[clean["Country Code"] == elem[0]].index[0]
        to_drop_indx.append(drop_indx)
    except Exception as e:
        print(indx, elem, e)

clean.drop(index=to_drop_indx, inplace=True)

print(clean.head(10))

clean.to_csv(OUT_PATH)