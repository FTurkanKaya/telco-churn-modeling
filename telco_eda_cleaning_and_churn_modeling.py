import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

from skimage.feature import shape_index
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


a=os.getcwd()
path=os.path.join(a,"Python_FutureEngineering","Datasets")

telco = f"{path}/Telco-Customer-Churn.csv"
diabetes = f"{path}/diabetes.csv"


def load(dataset):
    data = pd.read_csv(dataset)
    return data

df = load(telco)
df.head()

#  *************************************
#  TAAK 1 - EDA
#  Numerieke en Categorische variabelen analyse
#  *************************************

#  We groeperen ‘tenure’ in categorieën omdat het een continue variabele is.
#  Door deze in duidelijke intervallen te verdelen, wordt het eenvoudiger en
#  inzichtelijker om het gemiddelde churnpercentage per groep te analyseren.

# Tenure
df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
)
df["tenure_group"] = df["tenure_group"].astype("object")


# TotalCharges
# eerst de spaties van TotalCharges vervangen door NaN
# daarna omzetten naar float
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].dtype


def col_names_grab(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(col_names_grab(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = col_names_grab(df)

num_cols = [col for col in num_cols if "customerID" not in col]
df.info()



#  *************************************
#  Analyse van Categorische Variabelen
#  *************************************

# We bekijken hier hoe vaak elke categorie voorkomt en welk percentage van het totaal dat is.
for col in cat_cols:
    print(f"---- {col} ----")
    print(df[col].value_counts())  # aantallen per categorie
    print(df[col].value_counts(normalize=True) * 100)  # percentages
    print("\n")


#  *************************************
#  Analyse van Numerieke Variabelen
#  *************************************

df[num_cols].describe().T

# Hier bekijken we de beschrijvende statistieken van de numerieke kolommen:
# gemiddelde, standaarddeviatie, minimum en maximum waarden, kwartielen.


#  *************************************
#  Relatie met Doelvariabele (Target)
#  *************************************

#  Als de doelvariabele bijvoorbeeld "Churn" heet:
#
# Gemiddelde churn per categorie:

df["Churn_numeric"] = df["Churn"].map({"Yes": 1, "No": 0})

for col in cat_cols:
    print(f"{col} vs Churn")
    print(pd.DataFrame({"Churn_Gemiddelde": df.groupby(col)["Churn_numeric"].mean()}))
    print("\n")


#  Gemiddelde numerieke waarden per Churn-categorie:


df.groupby("Churn")[num_cols].mean()

df.info()
#  Hier kijken we of er een verband is tussen
#  de categorische of numerieke variabelen en de kans op churn (uitstroom).


#  *********************
#  1. Categorische variabelen vs Churn
#  *********************
#  Voor elke categorische variabele kun je een gestapelde staafdiagram of percentage barplot maken
#  om het verschil tussen churn = yes/no per categorie te zien.

for col in cat_cols:
    plt.figure(figsize=(8,4))
    churn_ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
    churn_ct.plot(kind='bar', stacked=True, color=['green', 'red'])
    plt.title(f"Churn verdeling per categorie: {col}")
    plt.ylabel("Percentage")
    plt.xlabel(col)
    plt.legend(title="Churn")
    plt.show()

#  *********************
#  2. Numerieke variabelen vs Churn
#  *********************

# Hier kun je een boxplot of violinplot gebruiken om te zien of
# de verdelingen van bijvoorbeeld tenure, MonthlyCharges en TotalCharges verschillend zijn voor churn=0/1.
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f"{col} distributie per Churn")
    plt.show()

# Of een histogram/kde plot:
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df, x=col, hue="Churn", fill=True)
    plt.title(f"{col} verdeling per Churn")
    plt.show()

#  ********************************
#  Voer een analyse van Outliers uit.
#  ********************************

# Het detecteren van uitschieters
#  *********************

df.head()
df.info()

def thresholds_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def outlier_check(dataframe, col_name):
    low_limit, up_limit = thresholds_outlier(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, outlier_check(df, col))


#  ********************************
#  Ontbrekende waarden analyse
#  ********************************

# We definiëren een functie om de ontbrekende waarden in de dataset overzichtelijk weer te geven:
def table_missing_values(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

table_missing_values(df)

table_missing_values(df, True)

# Bij de analyse van ontbrekende waarden zien we dat alleen de kolom TotalCharges ontbrekende waarden bevat.
#
# Bij nadere inspectie blijkt dat deze ontbrekende waarden voorkomen bij klanten met een tenure van 0.
# Dit betekent dat deze klanten net begonnen zijn en nog geen totale kosten hebben opgebouwd.
#
# Daarom is besloten om de ontbrekende waarden in TotalCharges op te vullen met 0,
# omdat dit logisch is gegeven hun tenure en het de dataset niet nadelig beïnvloedt.
df[df.isnull().any(axis=1)]

df["TotalCharges"].fillna(0, inplace=True)

#  ********************************
#  Correlatieanalyse
#  ********************************

na_cols = table_missing_values(df, True)

# Churn : 1 ---  No_Churn: 0

def missing_VS_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 'NaN', 'Not_NaN')

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({f"{target}_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_VS_target(df, "Churn_numeric", na_cols)


# We hebben onderzocht of de ontbrekende waarden in de kolom TotalCharges een verband hebben met de targetvariabele Churn_numeric.
# Hiervoor hebben we een nieuwe indicator (TotalCharges_NA_FLAG) aangemaakt die aangeeft of
# de waarde ontbreekt (NaN) of niet (Not_NaN).
#
# De resultaten:
#
# Flag	Gemiddelde Churn	Aantal
# NaN	0.000	11
# Not_NaN	0.266	7032
#
#  Dit betekent dat de rijen met ontbrekende waarden voor TotalCharges geen enkele churn (0%) vertonen.
#  De overige klanten (Not_NaN) hebben een churn-percentage van ongeveer 26,6%.
#
# Conclusie: de ontbrekende waarden lijken niet willekeurig te zijn, maar eerder gekoppeld aan klanten met een tenure van 0,
# wat logisch is omdat deze klanten waarschijnlijk nog niets betaald hebben.


#  ********************************
#  FEATURE ENGINEERING
#  ********************************

# Nieuwe Variabelen Aanmaken
# *********************************

df.head()
df.describe().T

#  1. TotalServices – Aantal gebruikte diensten
#  *******************************************

# Het totale aantal verschillende diensten dat een klant gebruikt. Hoe meer diensten, hoe meer betrokkenheid.
df['TotalServices'] = (
    df[['PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies']]
    .apply(lambda row: sum(row == 'Yes'), axis=1)
)

#  2. AvgMonthlySpend – Gemiddelde maandelijkse uitgaven
#  *******************************************

# Berekent hoeveel een klant gemiddeld per maand uitgeeft. Nuttig om "heavy users" te identificeren.

df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'].replace(0, 1))


#  3. ContractLevel – Contractduur als numerieke waarde
#  *******************************************

# Zet de contractduur om naar een numerieke schaal.
# Lange contracten wijzen vaak op een lager risico op opzegging.

contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df['ContractLevel'] = df['Contract'].map(contract_map)


#  4. HasSecurity – Gebruikt de klant beveiligingsdiensten?
#  *******************************************

# Geeft aan of een klant minimaal één beveiligingsdienst gebruikt. Deze klanten zijn vaak loyaler.

df['HasSecurity'] = df[['OnlineSecurity', 'OnlineBackup', 'TechSupport']].apply(
    lambda row: any(row == 'Yes'), axis=1
)

#  5. PricePerService – Prijs per gebruikte dienst
#  *******************************************

# DBerekent hoeveel een klant gemiddeld betaalt per dienst.
# Hoog bedrag = mogelijk overbilled of weinig diensten.

df['PricePerService'] = df['MonthlyCharges'] / (df['TotalServices'].replace(0, 1))


#############################################
# Label Encoding
#############################################
for col in cat_cols:
    print(df[col].value_counts())


def encode_label(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns
               if df[col].dtype == 'O' and df[col].nunique() == 2]


for col in binary_cols:
    encode_label(df,col)

df.head()
df[binary_cols].head()
df.dtypes


#############################################
# Rare Encoding
#############################################
df = load(telco)
df.head()

cat_cols, num_cols, cat_but_car = col_names_grab(df)

def analyse_rare(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")



def encode_rare(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = encode_rare(df, 0.01)

analyse_rare(new_df, "Churn_numeric", cat_cols)



#############################################
# One-Hot Encoding
#############################################

for col in df.columns:
    print(col, df[col].unique())

# Omdat de kolomtypes bool zijn, geven ze True of False terug.
# We zetten deze waarden om naar 0 en 1 zodat ze numeriek verwerkt kunnen worden.
for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

df.head()
df.dtypes

def one_hot_encoding(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoding(df, ohe_cols)

for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

df.head()

#############################################
# Feature Scaling - Schalen van numerieke variabelen
#############################################

# Hier passen we standaardisatie toe op numerieke kolommen.
# Standaardisatie (z-score) zorgt ervoor dat elke kolom een gemiddelde van 0 en een standaarddeviatie van 1 krijgt.
# Dit is vooral belangrijk bij algoritmes die gevoelig zijn voor schaling (zoals KNN, Logistic Regression, SVM).

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Pas de standaardisatie toe op de numerieke kolommen
df[num_cols] = scaler.fit_transform(df[num_cols])

# Bekijk de eerste 5 rijen van de geschaalde gegevens
df[num_cols].head()


#############################################
# 8. Model
#############################################
# De originele 'Churn'-kolom is nu overbodig en wordt verwijderd
df.drop("Churn", axis=1, inplace=True)

df.shape
df.columns

df.head()
y = df["Churn_numeric"]
X = df.drop(["customerID","Churn_numeric"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
print(X_train.select_dtypes(include='object').columns)

y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

print(X.columns)

# Yeni ürettiğimiz değişkenler ile karsilastirmak

def importance_ploting(model, features, num=20, save=False):
    if len(model.feature_importances_) != len(features.columns):
        raise ValueError(f"feature_importances_ length ({len(model.feature_importances_)}) ve features sütun sayısı ({len(features.columns)}) eşleşmiyor!")
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features Importance')
    plt.tight_layout()
    if save:
        plt.savefig('importances.png')
    plt.show()



importance_ploting(rf_model, X_train)

print(len(rf_model.feature_importances_))
print(len(X_train))


print("Model feature_importances_ uzunluğu:", len(rf_model.feature_importances_))
print("X_train sütun sayısı:", len(X_train.columns))
print("X_train sütun isimleri:", list(X_train.columns))

