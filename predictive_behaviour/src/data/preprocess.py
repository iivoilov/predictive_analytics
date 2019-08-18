import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
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

    # Encode labels
    labels = LabelEncoder().fit_transform(labels)

    education_map = {
        'illiterate': 0.0,
        'basic.4y' : 1.0,
        'basic.6y' : 2.0,
        'basic.9y' : 3.0,
        'high.school' : 4.0,
        'professional.course' : 5.0,
        'university.degree' : 5.0,
        'unknown' : 4.0
        }
    data['education'] = data['education'].map(education_map)

    col_t = ColumnTransformer(
        [("cat", OneHotEncoder(), ['job','marital','default','housing','loan','contact', 'month', 'day_of_week','poutcome'])],
        sparse_threshold=0)
    features = col_t.fit_transform(features)

    #Create an instance of sklearn's MinMaxScalar and use it to map the feature values in the range [0,1]
    sc = MinMaxScaler()
    features = sc.fit_transform(features)

    print(features.shape)

    # Split the dataset into training and testset. The train/test/validation ratio is 70%/20%/10%
    
    # Split the data into train and test sets with the ratio of 70/30 %
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, shuffle=True)
    # Split the testset again into a new testset and a validation set 
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.333, random_state=42, shuffle=True)

    return [x_train, y_train], [x_test, y_test], [x_val, y_val]

