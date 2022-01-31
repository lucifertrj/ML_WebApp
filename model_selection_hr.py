from sklearn.preprocessing import StandardScaler
from encoder import encode
import pandas as pd
import seaborn as sb

def plot():
    pass

def features(scale):
    data = pd.read_csv("./dataset/hr_job.csv")
    data.drop(["company_size","company_type"],axis=1,inplace=True)
    data = encode(data)
    feature_selection = ["relevent_experience","experience","training_hours","education_level","major_discipline"]
    X = data[feature_selection]
    if scale:
        sc = StandardScaler()
        X = sc.fit_transform(X)
    return X
