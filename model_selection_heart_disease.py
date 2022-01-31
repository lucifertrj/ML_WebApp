from encoder import encode
from sklearn.preprocessing import StandardScaler
import pandas as pd

def features(scale):
    data = pd.read_csv("./dataset/heart_disease.csv")
    data = encode(data)
    X = data.drop("target",axis=1)
    if scale:
        sc = StandardScaler()
        X = sc.fit_transform(X)
    return X