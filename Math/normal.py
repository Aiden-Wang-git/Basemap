import pandas as pd

coman = pd.read_csv('newdata.csv', names=range(0,43),header=0,)
# print(coman)
# Normalize the data
def regularit(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
            d = df[c]
            MAX = d.max()
            MIN = d.min()
            newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame
data = regularit(coman)
print(data)
data.to_csv('./c_norm.csv', index=False, header=0)