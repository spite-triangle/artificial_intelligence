import pandas as pd

datas = pd.DataFrame([[ 1,2,3,4,5 ],[1,2,3,4,5]])

datas.loc[0,:] = 0
datas *= 2
print(datas)