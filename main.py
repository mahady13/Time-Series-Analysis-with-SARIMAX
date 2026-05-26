import streamlit as st
import pandas as pd
import preprocessor as pp
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import DateOffset

st.sidebar.title('Step by Step Process Of Time Series Analysis With SARIMAX Using Ecommerce Dataset(UK)')
decided_action=st.sidebar.radio(label='Choose Any',options=['Main Dataset','Data Exploration','ADFuller Test','Differencing & Validating Data','ACF & PACF','Performance','Forecasting'])
df=pd.read_csv("data.csv",encoding='ISO-8859-1')
df_daily=pp.preprocess(df)
new_df=df_daily.copy()

if decided_action=='Main Dataset':
    st.title("Original Dataset")
    st.dataframe(df)
    st.title('Processed Dataset')
    st.dataframe(df_daily)

elif decided_action=='ADFuller Test':
    st.title('First ADFuller Test')
    results,labels=pp.dicky_fuller_test(df_daily['Sales'])
    for value, label in zip(results, labels):
        st.text(label + str(value))
    if results[1] <= 0.05:
        st.text('Null Hypothesis Rejected, the dataset is stationary')
    elif results[1] > 0.05:
        st.text('Null Hypothesis Accepted, the dataset is non-stationary')

elif decided_action=='Data Exploration':
    st.title('Sales Plot to Understand Seasonal Trend')
    fig,ax1=plt.subplots()
    ax1=df_daily['Sales'].plot()
    st.pyplot(fig)

elif decided_action=='Differencing & Validating Data':
    st.title('Now We do Differencing for Linear Trends')
    new_df['Sales Difference']=new_df['Sales']-new_df['Sales'].shift(1)
    st.dataframe(new_df)

    fig, ax2 = plt.subplots()
    ax2 = new_df['Sales Difference'].plot()
    st.title('Sales Difference Plot to Understand Linear Trend')
    st.pyplot(fig)

    st.title('Now We do Differencing for Seasonal(Weekly) Trends')
    new_df['Seasonal Sales Difference']=new_df['Sales Difference']-new_df['Sales Difference'].shift(7)
    st.dataframe(new_df)

    fig, ax3 = plt.subplots()
    ax3 = new_df['Seasonal Sales Difference'].plot()
    st.title('Seasonal Sales Difference Plot to Understand Seasonal Trend')
    st.pyplot(fig)

    st.title('Dicky Fuller Test Result for Linear+Seasonal Differencing')
    results,labels=pp.dicky_fuller_test(new_df['Seasonal Sales Difference'].dropna())
    for value, label in zip(results, labels):
        st.text(label + str(value))
    if results[1] <= 0.05:
        st.text('Null Hypothesis Rejected, the dataset is stationary')
    elif results[1] > 0.05:
        st.text('Null Hypothesis Accepted, the dataset is non-stationary')


elif decided_action=='ACF & PACF':
    new_df['Sales Difference'] = new_df['Sales'] - new_df['Sales'].shift(1)
    new_df['Seasonal Sales Difference'] = new_df['Sales Difference'] - new_df['Sales Difference'].shift(7)
    st.title('Partial Correlation Function for AR(AutoRegression=p)')
    fig,ax3=plt.subplots()
    fig=plot_pacf(new_df['Seasonal Sales Difference'].iloc[8:], lags=40,ax=ax3)
    st.pyplot(fig)

    st.title('Auto Correlation Function for MA(Moving Averages=q)')
    fig2,ax4=plt.subplots()
    fig2=plot_acf(new_df['Seasonal Sales Difference'].iloc[8:],ax=ax4)
    st.pyplot(fig2)

elif decided_action == 'Performance':
    model1=SARIMAX(new_df['Sales'],order=(1,1,1),seasonal_order=(1,1,1,7))
    result1=model1.fit()
    st.title('SARIMAX Result1 For (p,d,q)=(1,1,1)')
    st.text(result1.summary())

    model2=SARIMAX(new_df['Sales'],order=(2,1,1),seasonal_order=(2,1,1,7))
    result2=model2.fit()
    st.title('SARIMAX Result2 For (p,d,q)=(2,1,1)')
    st.text(result2.summary())

    st.subheader('As AIC & BIC of the 1st model(1,1,1) is less,it means it works better on forecasting. So we choose (p,d,q)=(1,1,1) for our final model')

elif decided_action=='Forecasting':
    new_df['Sales Difference'] = new_df['Sales'] - new_df['Sales'].shift(1)
    new_df['Seasonal Sales Difference'] = new_df['Sales Difference'] - new_df['Sales Difference'].shift(7)
    model1 = SARIMAX(new_df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    result1 = model1.fit()
    fig,ax5=plt.subplots()
    st.title('Model Forecasting Comparing with Old Data')
    prediction=result1.get_prediction(start=new_df['Sales'].index[-30],end=new_df['Sales'].index[-1],dynamic=True)
    new_df['Forecast']=prediction.predicted_mean
    pred_ci=prediction.conf_int()
    pred_ci.plot(color='green',alpha=0.3,ax=ax5)
    new_df['Sales'].plot(alpha=0.7,ax=ax5)
    new_df['Forecast'].plot(color='black',ax=ax5)
    plt.legend()
    st.pyplot(fig)

    st.sidebar.header('How many days do you want to see the forecast of')
    y=st.sidebar.number_input('Enter number of days:',7)
    y=int(y)
    if st.sidebar.button('Model Forecast'):
        st.title('Sales Forecast For Next '+str(y)+' Days')
        future_dates=[new_df.index[-1]+DateOffset(days=x) for x in range(0,y-1)]
        future_dates=pd.DataFrame(index=future_dates[1:],columns=new_df.columns)
        predict=result1.get_prediction(start=future_dates.index[0],end=future_dates.index[-1],dynamic=True)
        future_dates['Forecast']=predict.predicted_mean
        pred_ci2=predict.conf_int()


        fig,ax6=plt.subplots()
        new_df['Sales'].iloc[-60:].plot(color='green',alpha=0.6,ax=ax6)
        future_dates['Forecast'].plot(color='orange',ax=ax6)
        ax6.fill_between(pred_ci2.index,pred_ci2.iloc[:,0],pred_ci2.iloc[:,1],color='orange',alpha=0.2)
        st.pyplot(plt.gcf())