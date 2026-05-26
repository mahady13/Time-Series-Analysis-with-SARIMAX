import pandas as pd
from statsmodels.tsa.stattools import adfuller

def preprocess(df):
    df['Sales']=df['UnitPrice']*df['Quantity']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.set_index(['InvoiceDate'],inplace=True)
    df_daily=df['Sales'].resample('D').sum().to_frame()
    return df_daily

def dicky_fuller_test(df):
        results=adfuller(df)
        labels=['ADF Test Statistic: ','P Value: ','Lags Used: ','Number of Observations used: ']
        return results,labels