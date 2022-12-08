
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from fbprophet import Prophet


Con_Case = pd.read_csv("C:\Users\Nishanth Mohankumar\OneDrive\Desktop\BBMP_Corporators_office\Cases_in_Blore.csv")
Con_Death = pd.read_csv("C:\Users\Nishanth Mohankumar\OneDrive\Desktop\BBMP_Corporators_office\Deaths_in_Blore.csv")


df =pd.DataFrame()
df['Date']= pd.to_datetime(Con_Case['Dates'][1:])
df=df.set_index("Date")
df['Cases']= pd.to_numeric(Con_Case['Cases'][1:])
df['Deaths']= pd.to_numeric(Con_Death['Deaths'][1:])

plt.style.use("ggplot")

df.Cases.plot(title="Daily Covid19 Cases in Bangalore",marker="*",figsize=(10,5),label="Cases Daily")
df.Cases.rolling(window=5).mean().plot(figsize=(11,6),label="MeanOver5")
plt.legend()
plt.ylabel("Cases in Bangalore")
plt.show()

df.Deaths.plot(title="Daily Covid19 Deaths in Bangalore",marker="*",figsize=(10,5),label="Deaths Daily")
df.Deaths.rolling(window=5).mean().plot(figsize=(11,6),label="MeanOver5")
plt.legend()
plt.ylabel("Deaths in Bangalore")
plt.show()

class FaceBookProphet(object):
    def Fit_model(self,Data):
        
        self.Data  = Data
        self.model = Prophet(weekly_seasonality=True,daily_seasonality=False,yearly_seasonality=False)
        self.model.fit(self.Data)
    
    def Forecast_model(self,per,freq):
        
        self.future = self.model.make_future_dataframe(per=per,freq=freq)
        self.df_forecast = self.model.predict(self.future)
        
    def R2_score(self):
        return r2_score(self.Data.y, self.df_forecast.yhat[:len(df)])   
    
    def plot(self,xlabel="Years",ylabel="Values"):
        
        self.model.plot(self.df_forecast,xlabel=xlabel,ylabel=ylabel,figsize=(10,5))
        self.model.plot_components(self.df_forecast,figsize=(10,7))
        
    
        
df_train  = pd.DataFrame({"Date":[],"Y":[]})
df_train["Date"] = pd.to_datetime(df.index)
df_train["Y"]  = df.iloc[:,0].values

model = FaceBookProphet()
model.Fit_model(df_train)

num_of_days = int(input('Number of days to predict?\n'))

model.Forecast_model(num_of_days,"D")
model.R2_score()

FC = model.Forecast_model[["ds","yhat_lower","yhat_upper","yhat"]].tail(num_of_days).reset_index().set_index("ds").drop("index",axis=1)
FC["yhat"].plot(marker=".",figsize=(11,6))
plt.fill_between(x=FC.index, y1=FC["yhat_lower"], y2=FC["yhat_upper"],color="blue")
plt.title("Forecasting of Next No. of Chosen Days Cases")
plt.legend(["Fore Cast","Bounding"],loc="upper right")

plt.show()