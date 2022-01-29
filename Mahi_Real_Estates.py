import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd

housing = pd.read_csv("data.csv")

housing.head()

housing.info()

housing['CHAS'].value_counts()

housing.describe()

housing1 = (housing.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]).copy()

housing1

housing1.describe()

housing1.info()

housing1.to_csv('data1.csv',index = False)



housing1.hist(bins=50, figsize=(20, 15))
# train-test spliting
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing1, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing1, housing1['CHAS']):
    strat_train_set = housing1.loc[train_index]
    strat_test_set = housing1.loc[test_index]

strat_test_set

strat_test_set['CHAS'].value_counts()

strat_train_set['CHAS'].value_counts()

housing1 = strat_train_set.copy()

#Looking for correlation

corr_matrix = housing1.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))

housing1.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)

# Trying our attribute combinations

housing1["TAXRM"] = housing1['TAX']/housing1['RM']

housing1.head()

corr_matrix = housing1.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

housing1.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)

housing1 = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set[("MEDV")].copy()

# Creating Pipeline
median = housing1["RM"].median()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing1)


imputer.statistics_

X = imputer.transform(housing1)

housing_tr = pd.DataFrame(X, columns=housing1.columns)

housing_tr.describe()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing1)

# Selecting a desired model for Mahi Real Estates
housing_num_tr

housing_num_tr.shape

# Selecting a desired model for Mahi Real Estates
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)

some_data = housing1.iloc[:5]

some_labels = housing_labels.iloc[:5]

prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

list(some_labels)

# Evaluating the model
import numpy as np
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

rmse

# Using better evaluation technique - Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

rmse_scores

def print_scores(scores):
    print("scores:",scores)
    print("mean: ", scores.mean())
    print("standard deviation: ", scores.std())

print_scores(rmse_scores)

# Saving the model
from joblib import dump, load
dump(model, 'Mahi.joblib')

X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_predictions, list(Y_test))


final_rmse

prepared_data[0]

from joblib import dump, load
import numpy as np
model = load('Mahi.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -6.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)
