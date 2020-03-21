df_apt = pd.read_csv("./drive/My Drive/Colab Notebooks/eval_data.csv")


timese_df = df_apt[df_apt["room_id"]==3513]

timese_df["transaction_year_month"] = timese_df["transaction_year_month"].astype(str)

timese_df.head(1)

timese_df["trade_year"] = timese_df["transaction_year_month"].apply(lambda x : x[:4])
timese_df["trade_month"] = timese_df["transaction_year_month"].apply(lambda x : x[4:])

timese_df.dtypes

timese_df["transaction_day"] = timese_df["transaction_day"].astype(str)

timese_df["trade_date"] = timese_df[["trade_year","trade_month","transaction_day"]].apply(lambda row : "{}-{}-{}".format(row[0], row[1], row[2]), axis=1)

timese_df["trade_date"] = pd.to_datetime(timese_df["trade_date"])

timese_df=timese_df[["trade_date", "trade_price"]]

timese_df.head()

timese_df_set_index = timese_df.set_index('trade_date')

timese_df_set_index.index

y =timese_df_set_index.resample('MS').mean()["trade_price"]

y.head()

from fbprophet import Prophet
prophet_input_df = pd.DataFrame(y)
prophet_input_df.reset_index(drop=False,inplace=True)
prophet_input_df.rename(columns={"trade_date":"ds","trade_price":"y"},inplace=True)


m = Prophet()
m.fit(prophet_input_df)
future = m.make_future_dataframe(periods=12,freq="MS")
future.tail()

forecast = m.predict(future)
forecast.tail()

fig = m.plot_components(forecast)

forecast[["ds","yhat"]][-12:]










