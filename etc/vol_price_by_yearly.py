import pandas as pd
import numpy as np


#files = ["./data/framp.csv", "./data/gnyned.csv", "./data/gwoomed.csv", 
#         "./data/hoilled.csv", "./data/plent.csv", "./data/throwsh.csv",
#         "./data/twerche.csv", "./data/veeme.csv"]



def na(files):
    
    result = []
    
    for i in files:
        
        result1= []
        
        df = pd.read_csv(i)
        
        df['date'] = pd.to_datetime(df['date'])

        df['year'] = df['date'].dt.strftime('%Y')

        max_price_index = list(df.groupby(['year'])['close'].idxmax())
        max_vol_index = list(df.groupby(['year'])['vol'].idxmax())

        max_vol_data = df[['date','vol','year']].iloc[max_vol_index].reset_index()
        max_vol_data.drop(['index'], inplace =True, axis = 1)

        max_price_data = df[['date','close','year']].iloc[max_price_index].reset_index()
        max_price_data.drop(['index'], inplace =True, axis = 1)


        for i in range(len(max_price_data)):

            for j in range(len(df)):                  

                if max_price_data['close'][i] == df['close'][j] and max_price_data['year'][i] == df['year'][j]: 

                    max_price_index.append(j)


        for i in range(len(max_vol_data)):

            for j in range(len(df)):                  

                if max_vol_data['vol'][i] == df['vol'][j] and max_vol_data['year'][i] == df['year'][j]: 

                    max_vol_index.append(j)


        max_price_index = list(set(max_price_index))
        max_vol_index = list(set(max_vol_index))


        max_vol_data = df[['date','vol']].iloc[max_vol_index].reset_index()
        max_vol_data.drop(['index'], inplace =True, axis = 1)

        max_price_data = df[['date','close']].iloc[max_price_index].reset_index()
        max_price_data.drop(['index'], inplace =True, axis = 1)


        max_price_data = max_price_data.sort_values("date").reset_index().head(6)
        max_price_data.drop(['index'], inplace =True, axis = 1)
        max_vol_data = max_vol_data.sort_values("date").reset_index().head(6)
        max_vol_data.drop(['index'], inplace =True, axis = 1)
        
        result1.append(max_vol_data)
        result1.append(max_price_data)
        
        result.append(result1)
        
    return result