import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
import os

train_data_dir=os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', '../data/raw/bank_full.csv'))

#%%

def get_data():

    # Import the csv data as pands's dataframe
    data = pd.read_csv(train_data_dir,delimiter=';')

    # Get the features from the dataframe
    features = data.iloc[:,0:20]
    #Get the labels from the dataset
    labels = data.iloc[:,20].values

    le = LabelEncoder()

    # Encode labels
    labels = le.fit_transform(labels)

    # After LableEncoder columns become of dtype float
    # for col in ['job','marital','education','contact','month','day_of_week','poutcome']:
    for col in ['education','contact','month','day_of_week']:
        features[col] = le.fit_transform(features[col])

    # get_dummies() works only on columns with dtype object
    features = pd.get_dummies(features,prefix_sep='_')

    #Create an instance of sklearn's MinMaxScalar and use it to map the feature values in the range [0,1]
    sc = MinMaxScaler()
    features = sc.fit_transform(features)

    print(features.shape)

    # Split the dataset into training and testset. The train/test/validation ratio is 70%/20%/10%
    
    # Split the data into train and test sets with the ratio of 70/30 %
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, shuffle=True)
    # Split the testset again into a new testset and a validation set 
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.333, random_state=42, shuffle=True)

    n1 = sum(y_test)
    n0 = len(y_test)
    print("Test set %d/%d avg %.3f" % (n1,n0,float(n1)/n0))

    return [x_train, y_train], [x_test, y_test], [x_val, y_val]

