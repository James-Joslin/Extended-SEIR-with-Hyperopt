import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller as adf
from pmdarima.arima import auto_arima

r_df = pd.read_csv("./Past_R.csv")
r_df['r_mean'] = (r_df['LowerR'] + r_df['UpperR'])/2
r_df['Date'] = pd.to_datetime(r_df['Date'], dayfirst=True)
r_df.index = r_df['Date']
r_df = r_df.drop(["Date", "UpperR", "LowerR"], axis = 1)
print(r_df)

output = (adf(r_df))
dfoutput = pd.Series(output[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in output[4].items():
        dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

split_pct = 0.85
train = r_df[:int(split_pct*(len(r_df)))]
valid = r_df[int(split_pct*(len(r_df))):]
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True,seasonal=True,m=18,D=1)
model.fit(train)

num_weeks = 48
forecast = model.predict(n_periods=num_weeks)
forecast_dates = pd.date_range(start=valid.index[0], periods=num_weeks, freq='W')
forecast = pd.DataFrame(forecast,index = forecast_dates,columns=['Prediction'])

#plot the predictions for validation set
rcParams['figure.figsize'] = (15, 6)
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.legend(loc='best')
plt.show()
forecast["Date"] = forecast.index
forecast.to_csv("./R_Forecast.csv", index = False)