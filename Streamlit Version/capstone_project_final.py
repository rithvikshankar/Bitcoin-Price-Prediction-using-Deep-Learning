import streamlit as st
from streamlit_option_menu import option_menu

import tensorflow as tf
import numpy as np
import pandas as pd
import math

import keras

from sklearn.metrics import mean_squared_error
import random

import yfinance as yf

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.layers import Activation, Flatten, Conv1D, MaxPooling1D

from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, LeakyReLU

import matplotlib.pyplot as plt
import plotly.express as px

from os.path import exists

from textblob import TextBlob

# mape_lstm_1, mape_lstm_2, mape_lstm_3 = 0, 0, 0
# results = {}
# flattened_results = []

data = yf.download(tickers="BTC-USD", end="2023-05-19")
# data=yf.download(tickers='BTC-USD',start='2014-09-17',end='2023-02-28')

data.to_csv("livebitcoindata.csv")

data = pd.read_csv("livebitcoindata.csv")  # loading data

nullval=data.isnull().values.sum()
print('Number of null values present:',nullval) #check for any null values
if(nullval>0):
  data.fillna(method='ffill', inplace=True) #dealing with null values if any are present 
  print("null values dealt with")

print("Starting Date: ", data.iloc[0][0])
print("Ending Date: ", data.iloc[-1][0])

newdf = data

newdf.drop(newdf[["Adj Close", "Volume"]], axis=1)  # remove unnecessary columns

plotBitcoinTrends = px.line(
    newdf,
    x=newdf.Date,
    y=[newdf["Open"], newdf["Close"], newdf["High"], newdf["Low"]],
    labels={"Date": "Date", "value": "Bitcoin Prices"},
)

selected = option_menu(
    menu_title=None,
    options=["Home", "About", "Contact"],
    icons=["house", "info-circle", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


close_data = data[["Date", "Close"]]  # only close price is needed
df = close_data

scaler = MinMaxScaler()

df = df["Close"].values.reshape(-1, 1)

df = scaler.fit_transform(df)  # performing normalization of data

# splitting training and testing data
split = int(len(df) * 0.80)
train = df[0:split]  
test = df[split : len(df), :1]
print("Training data: ", train.shape)
print("Testing data: ", test.shape)


def transform(data, step):  # transform into sequences
    x = []
    y = []
    for i in range(len(data) - step - 1):
        x.append(data[i : (i + step), 0])
        y.append(data[i + step, 0])
    return np.array(x), np.array(y)


step = 20  # window size
xtraining, ytraining = transform(train, step)
xtesting, ytesting = transform(test, step)

print("x training data: ", xtraining.shape)
print("y training data: ", ytraining.shape)
print("x testing data: ", xtesting.shape)
print("y testing data", ytesting.shape)

xtraining = xtraining.reshape(
    xtraining.shape[0], xtraining.shape[1], 1
)  # reshape the data into required shape for input to model
xtesting = xtesting.reshape(xtesting.shape[0], xtesting.shape[1], 1)

print("X training data: ", xtraining.shape)
print("X testing data: ", xtesting.shape)


ytraining = ytraining.reshape(ytraining.shape[0], 1)
ytesting = ytesting.reshape(ytesting.shape[0], 1)
print("Y training data: ", ytraining.shape)
print("Y testing data: ", ytesting.shape)

dates = close_data["Date"]
dates = dates[int(len(df) * 0.80) + step : -1]  # separating the dates for test data
print(dates)

y_inverse = scaler.inverse_transform(ytesting)
y_inverse = y_inverse.reshape(y_inverse.shape[0])


def modelpred(model):  # predicting using the model
    ypredict = model.predict(xtesting)
    yp_inverse = scaler.inverse_transform(ypredict)
    yp_inverse = yp_inverse.reshape(yp_inverse.shape[0])
    return yp_inverse


def plotseries(dates, actual, predicted):  # plot actual price vs predicted price
    plotdf = pd.DataFrame()
    plotdf["Date"] = dates
    plotdf["Actual Price"] = actual
    plotdf["Predicted Price"] = predicted

    plot = px.line(
        plotdf,
        x="Date",
        y=["Actual Price", "Predicted Price"],
        labels={"x": "Date", "value": "Bitcoin Prices"},
    )

    st.plotly_chart(plot, theme="streamlit", use_container_width=True)


if selected == "Home":
    st.title("Bitcoin Price Trends Over the Last Decade")

    st.plotly_chart(plotBitcoinTrends, theme=None, use_container_width=True)

    st.title("Select model to be used")

    option = st.selectbox("Model", ("Select Model", "CNN", "LSTM", "GRU"))

    if option != "Select Model":
        if option == "CNN":
            noOfLayers = st.select_slider(
                "Select the number of layers for CNN", options=["Select:", "2", "3"]
            )

            keras.backend.clear_session()
            tf.random.set_seed(40)
            np.random.seed(40)
            random.seed(40)

            if noOfLayers == "2":
                if not exists("checkpoints/cnn_checkpoint1.h5"):
                    cnnmodel = Sequential()
                    cnnmodel.add(
                        Conv1D(
                            64,
                            kernel_size=3,
                            activation="relu",
                            input_shape=(xtraining.shape[1], 1),
                        )
                    )
                    cnnmodel.add(MaxPooling1D(pool_size=2))
                    cnnmodel.add(
                        Conv1D(128, kernel_size=3, activation="relu", padding="same")
                    )
                    cnnmodel.add(MaxPooling1D(pool_size=2))
                    cnnmodel.add(Flatten())
                    cnnmodel.add(Dense(64, activation="relu"))
                    cnnmodel.add(Dense(1, activation="linear"))
                    cnnmodel.compile(loss=keras.losses.Huber(), optimizer="adam")
                    checkpoint = keras.callbacks.ModelCheckpoint(
                        "checkpoints/cnn_checkpoint1.h5", save_best_only=True
                    )
                    earlystop = keras.callbacks.EarlyStopping(patience=100)
                    cnnmodel.fit(
                        xtraining,
                        ytraining,
                        epochs=300,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[earlystop, checkpoint],
                        shuffle=False,
                    )
                    loss = min(cnnmodel.history.history["val_loss"])
                    print("Lowest value of validation loss:", f"{loss:.6f}")
                cnnmodel = keras.models.load_model("checkpoints/cnn_checkpoint1.h5")
                yp1 = modelpred(cnnmodel)
                plotseries(dates, y_inverse, yp1)
                mae_value1 = keras.metrics.mean_absolute_error(y_inverse, yp1).numpy()
                st.write("MAE: ", mae_value1)
                rmse_value1 = math.sqrt(
                    keras.metrics.mean_squared_error(y_inverse, yp1).numpy()
                )
                st.write("RMSE: ", round(rmse_value1, 4))
                mape_value1 = np.mean(np.abs((y_inverse - yp1) / y_inverse)) * 100
                st.write("MAPE: ", round(mape_value1, 2))

            elif noOfLayers == "3":
                if not exists("checkpoints/cnn_checkpoint2.h5"):
                    cnnmodel2 = Sequential()
                    cnnmodel2.add(
                        Conv1D(
                            32,
                            kernel_size=3,
                            activation="relu",
                            input_shape=(xtraining.shape[1], 1),
                        )
                    )
                    cnnmodel2.add(MaxPooling1D(pool_size=2))
                    cnnmodel2.add(
                        Conv1D(64, kernel_size=3, activation="relu", padding="same")
                    )
                    cnnmodel2.add(MaxPooling1D(pool_size=2))
                    cnnmodel2.add(
                        Conv1D(64, kernel_size=2, activation="relu", padding="same")
                    )
                    cnnmodel2.add(MaxPooling1D(pool_size=2))
                    cnnmodel2.add(Flatten())
                    cnnmodel2.add(Dense(64, activation="relu"))
                    cnnmodel2.add(Dense(1, activation="linear"))
                    cnnmodel2.compile(loss=keras.losses.Huber(), optimizer="adam")
                    checkpoint = keras.callbacks.ModelCheckpoint(
                        "checkpoints/cnn_checkpoint2.h5", save_best_only=True
                    )
                    earlystop = keras.callbacks.EarlyStopping(patience=100)
                    cnnmodel2.fit(
                        xtraining,
                        ytraining,
                        epochs=300,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[earlystop, checkpoint],
                        shuffle=False,
                    )
                    loss = min(cnnmodel2.history.history["val_loss"])
                    print("Lowest value of validation loss:", f"{loss:.6f}")
                cnnmodel2 = keras.models.load_model("checkpoints/cnn_checkpoint2.h5")
                yp2 = modelpred(cnnmodel2)
                plotseries(dates, y_inverse, yp2)
                mae_value1 = keras.metrics.mean_absolute_error(y_inverse, yp2).numpy()
                st.write("MAE: ", mae_value1)
                rmse_value1 = math.sqrt(
                    keras.metrics.mean_squared_error(y_inverse, yp2).numpy()
                )
                st.write("RMSE: ", round(rmse_value1, 4))
                mape_value1 = np.mean(np.abs((y_inverse - yp2) / y_inverse)) * 100
                st.write("MAPE: ", round(mape_value1, 2))

        elif option == "LSTM":
            noOfLayers = st.select_slider(
                "Select the number of layers for LSTM",
                options=["Select:", "1", "2", "3"],
            )

            keras.backend.clear_session()
            tf.random.set_seed(40)
            np.random.seed(40)
            random.seed(40)

            if noOfLayers == "1":
                if not exists("checkpoints/lstm_checkpoint3.h5"):
                    lstm1 = Sequential()
                    lstm1.add(
                        LSTM(
                            128,
                            activation="tanh",
                            input_shape=(xtraining.shape[1], 1),
                            return_sequences=False,
                        )
                    )

                    lstm1.add(Dense(1))
                    lstm1.add(LeakyReLU())
                    lstm1.compile(loss=keras.losses.Huber(), optimizer="adam")

                    checkpoint = keras.callbacks.ModelCheckpoint(
                        "checkpoints/lstm_checkpoint3.h5", save_best_only=True
                    )
                    earlystop = keras.callbacks.EarlyStopping(patience=50)
                    lstm1.fit(
                        xtraining,
                        ytraining,
                        epochs=300,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[earlystop, checkpoint],
                    )

                    loss = min(lstm1.history.history["val_loss"])
                    print("Lowest value of validation loss:", f"{loss:.6f}")

                lstm1 = keras.models.load_model("checkpoints/lstm_checkpoint3.h5")
                yp5 = modelpred(lstm1)
                plotseries(dates, y_inverse, yp5)

                st.write(
                    "MAE: ", keras.metrics.mean_absolute_error(y_inverse, yp5).numpy()
                )

                m = keras.metrics.mean_squared_error(y_inverse, yp5).numpy()

                st.write("RMSE: ", round(math.sqrt(m), 4))

                mape_lstm_1 = np.mean(np.abs((y_inverse - yp5) / y_inverse)) * 100
                st.write("MAPE: ", round(mape_lstm_1, 2))

                y_inverse = y_inverse.reshape(-1, 1)
                yp5 = yp5.reshape(-1, 1)

                # Compute the scalar values for metrics
                mae_value = keras.metrics.mean_absolute_error(
                    y_inverse, scaler.inverse_transform(yp5)
                ).numpy()
                mse_value = keras.metrics.mean_squared_error(
                    y_inverse, scaler.inverse_transform(yp5)
                ).numpy()

            elif noOfLayers == "2":
                if not exists("checkpoints/lstm_checkpoint2.h5"):
                    lstm2 = Sequential()
                    lstm2.add(
                        LSTM(
                            64,
                            return_sequences=True,
                            activation="tanh",
                            input_shape=(xtraining.shape[1], 1),
                        )
                    )

                    lstm2.add(LSTM(128, activation="tanh"))

                    lstm2.add(Dense(1))
                    lstm2.add(LeakyReLU())

                    lstm2.compile(loss=keras.losses.Huber(), optimizer="adam")
                    checkpoint = keras.callbacks.ModelCheckpoint(
                        "checkpoints/lstm_checkpoint2.h5", save_best_only=True
                    )
                    earlystop = keras.callbacks.EarlyStopping(patience=50)
                    lstm2.fit(
                        xtraining,
                        ytraining,
                        epochs=300,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[earlystop, checkpoint],
                    )

                    loss = min(lstm2.history.history["val_loss"])
                    print("Lowest value of validation loss:", f"{loss:.6f}")

                lstm2 = keras.models.load_model("checkpoints/lstm_checkpoint2.h5")
                yp4 = modelpred(lstm2)
                plotseries(dates, y_inverse, yp4)
                st.write(
                    "MAE: ", keras.metrics.mean_absolute_error(y_inverse, yp4).numpy()
                )

                m = keras.metrics.mean_squared_error(y_inverse, yp4).numpy()

                st.write("RMSE: ", round(math.sqrt(m), 4))

                mape_lstm_2 = np.mean(np.abs((y_inverse - yp4) / y_inverse)) * 100
                st.write("MAPE: ", round(mape_lstm_2, 2))

                y_inverse = y_inverse.reshape(-1, 1)
                yp4 = yp4.reshape(-1, 1)

            elif noOfLayers == "3":
                if not exists("checkpoints/lstm_checkpoint1.h5"):
                    lstm3 = Sequential()
                    lstm3.add(
                        LSTM(
                            32,
                            return_sequences=True,
                            activation="tanh",
                            input_shape=(xtraining.shape[1], 1),
                        )
                    )

                    lstm3.add(LSTM(64, return_sequences=True, activation="tanh"))

                    # lstm1.add(Dropout(0.1))

                    lstm3.add(LSTM(128, activation="tanh"))

                    lstm3.add(Dense(1))
                    lstm3.add(LeakyReLU())

                    lstm3.compile(loss=keras.losses.Huber(), optimizer="adam")
                    checkpoint = keras.callbacks.ModelCheckpoint(
                        "checkpoints/lstm_checkpoint1.h5", save_best_only=True
                    )
                    earlystop = keras.callbacks.EarlyStopping(patience=50)
                    lstm3.fit(
                        xtraining,
                        ytraining,
                        epochs=300,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[earlystop, checkpoint],
                    )

                    loss = min(lstm3.history.history["val_loss"])
                    print("Lowest value of validation loss:", f"{loss:.6f}")

                lstm3 = keras.models.load_model("checkpoints/lstm_checkpoint1.h5")
                yp3 = modelpred(lstm3)
                plotseries(dates, y_inverse, yp3)
                st.write(
                    "MAE: ", keras.metrics.mean_absolute_error(y_inverse, yp3).numpy()
                )

                m = keras.metrics.mean_squared_error(y_inverse, yp3).numpy()

                st.write("RMSE: ", round(math.sqrt(m), 4))

                mape_lstm_3 = np.mean(np.abs((y_inverse - yp3) / y_inverse)) * 100
                st.write("MAPE: ", round(mape_lstm_3, 2))

            # def build_lstm_model(num_layers):
            #     model = Sequential()
            #     for i in range(num_layers):
            #         if i == 0:
            #             model.add(LSTM(32, return_sequences=True, activation='tanh', input_shape=(xtraining.shape[1],1)))
            #         elif i == num_layers - 1:
            #             model.add(LSTM(128, activation='tanh'))
            #         else:
            #             model.add(LSTM(64, return_sequences=True, activation='tanh'))
            #         model.add(LeakyReLU())
            #     model.add(Dense(1))
            #     model.compile(loss=keras.losses.Huber(), optimizer='adam')
            #     return model

            # no_of_layers = st.select_slider("Select the number of layers for LSTM", options=[1, 2, 3])

            # keras.backend.clear_session()
            # tf.random.set_seed(40)
            # np.random.seed(40)
            # random.seed(40)

            # checkpoint = keras.callbacks.ModelCheckpoint("lstm_checkpoint.h5", save_best_only=True)
            # earlystop = keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)

            # if not exists("lstm_checkpoint.h5"):
            #     lstm = build_lstm_model(no_of_layers)
            #     lstm.fit(xtraining, ytraining, epochs=300, batch_size=64, validation_split=0.1, callbacks=[earlystop, checkpoint])
            # else:
            #     lstm = keras.models.load_model("lstm_checkpoint.h5")

            # yp = modelpred(lstm)
            # plotseries(dates, y_inverse, yp)
            # st.write("MAE: ", keras.metrics.mean_absolute_error(y_inverse, yp).numpy())
            # m = keras.metrics.mean_squared_error(y_inverse, yp).numpy()
            # st.write("RMSE: ", round(math.sqrt(m), 4))
            # mape = np.mean(np.abs((y_inverse - yp) / y_inverse)) * 100
            # st.write("MAPE: ", round(mape, 2))

        elif option == "GRU":
            noOfLayers = st.select_slider(
                "Select the number of layers for GRU",
                options=["Select:", "1", "2", "3"],
            )

            keras.backend.clear_session()
            tf.random.set_seed(40)
            np.random.seed(40)
            random.seed(40)

            if noOfLayers == "1":
                if not exists("checkpoints/gru_checkpoint3.h5"):
                    gru3 = Sequential()

                    gru3.add(
                        GRU(
                            128,
                            input_shape=(xtraining.shape[1], 1),
                            activation="tanh",
                            return_sequences=False,
                        )
                    )

                    gru3.add(Dense(units=1))

                    gru3.compile(loss=keras.losses.Huber(), optimizer="adam")

                    checkpoint = keras.callbacks.ModelCheckpoint(
                        "checkpoints/gru_checkpoint3.h5", save_best_only=True
                    )
                    earlystop = keras.callbacks.EarlyStopping(patience=50)
                    gru3.fit(
                        xtraining,
                        ytraining,
                        epochs=300,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[earlystop, checkpoint],
                    )

                    loss = min(gru3.history.history["val_loss"])
                    print("Lowest value of validation loss:", f"{loss:.6f}")

                gru3 = keras.models.load_model("checkpoints/gru_checkpoint3.h5")

                yp8 = modelpred(gru3)

                plotseries(dates, y_inverse, yp8)
                st.write(
                    "MAE: ", keras.metrics.mean_absolute_error(y_inverse, yp8).numpy()
                )

                m = keras.metrics.mean_squared_error(y_inverse, yp8).numpy()

                st.write("RMSE: ", round(math.sqrt(m), 4))

                mape = np.mean(np.abs((y_inverse - yp8) / y_inverse)) * 100
                st.write("MAPE: ", round(mape, 2))

            elif noOfLayers == "2":
                if not exists("checkpoints/gru_checkpoint2.h5"):
                    gru2 = Sequential()

                    gru2.add(
                        GRU(
                            64,
                            return_sequences=True,
                            input_shape=(xtraining.shape[1], 1),
                            activation="tanh",
                        )
                    )

                    gru2.add(GRU(128, activation="tanh"))

                    gru2.add(Dense(1))

                    gru2.compile(loss=keras.losses.Huber(), optimizer="adam")

                    checkpoint = keras.callbacks.ModelCheckpoint(
                        "checkpoints/gru_checkpoint2.h5", save_best_only=True
                    )
                    earlystop = keras.callbacks.EarlyStopping(patience=50)
                    gru2.fit(
                        xtraining,
                        ytraining,
                        epochs=300,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[earlystop, checkpoint],
                    )

                    loss = min(gru2.history.history["val_loss"])
                    print("Lowest value of validation loss:", f"{loss:.6f}")

                gru2 = keras.models.load_model("checkpoints/gru_checkpoint2.h5")
                yp7 = modelpred(gru2)
                plotseries(dates, y_inverse, yp7)
                st.write(
                    "MAE: ", keras.metrics.mean_absolute_error(y_inverse, yp7).numpy()
                )

                m = keras.metrics.mean_squared_error(y_inverse, yp7).numpy()

                st.write("RMSE: ", round(math.sqrt(m), 4))

                mape = np.mean(np.abs((y_inverse - yp7) / y_inverse)) * 100
                st.write("MAPE: ", round(mape, 2))

            elif noOfLayers == "3":
                if not exists("checkpoints/gru_checkpoint1.h5"):
                    gru1 = Sequential()

                    gru1.add(
                        GRU(
                            64,
                            return_sequences=True,
                            input_shape=(xtraining.shape[1], 1),
                            activation="tanh",
                        )
                    )

                    gru1.add(GRU(64, return_sequences=True, activation="tanh"))

                    gru1.add(GRU(128, activation="tanh"))

                    gru1.add(Dense(units=1))

                    gru1.compile(loss=keras.losses.Huber(), optimizer="adam")

                    checkpoint = keras.callbacks.ModelCheckpoint(
                        "checkpoints/gru_checkpoint1.h5", save_best_only=True
                    )
                    earlystop = keras.callbacks.EarlyStopping(patience=50)
                    gru1.fit(
                        xtraining,
                        ytraining,
                        epochs=300,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[earlystop, checkpoint],
                    )

                    loss = min(gru1.history.history["val_loss"])
                    print("Lowest value of validation loss:", f"{loss:.6f}")

                gru1 = keras.models.load_model("checkpoints/gru_checkpoint1.h5")
                yp6 = modelpred(gru1)
                plotseries(dates, y_inverse, yp6)
                st.write(
                    "MAE: ", keras.metrics.mean_absolute_error(y_inverse, yp6).numpy()
                )

                m = keras.metrics.mean_squared_error(y_inverse, yp6).numpy()

                st.write("RMSE: ", round(math.sqrt(m), 4))

                mape = np.mean(np.abs((y_inverse - yp6) / y_inverse)) * 100
                st.write("MAPE: ", round(mape, 2))

elif selected == "About":
    st.title("About Bitcoin ")
    st.write(
        """Bitcoin is a decentralized digital currency that operates on a peer-to-peer network known as the blockchain.
          Created in 2009 by an anonymous person or group of people using the pseudonym Satoshi Nakamoto, Bitcoin
            revolutionized the concept of money by introducing a secure, transparent, and censorship-resistant form of digital currency.

Unlike traditional currencies issued by central banks, Bitcoin is not controlled by any government or financial institution.
 It utilizes cryptographic techniques to secure transactions and regulate the creation of new units. Bitcoin transactions are
   recorded on the blockchain, a public ledger that ensures transparency and prevents double-spending.

Bitcoin's value is determined by market forces and can be highly volatile. It has gained popularity as an alternative investment
 and a means of transferring value across borders quickly and relatively cheaply. Bitcoin's limited supply and decentralized
   nature have also made it a store of value and a potential hedge against inflation.

Overall, Bitcoin represents a groundbreaking innovation in the world of finance, offering a digital currency system that operates
 independently of traditional financial intermediaries."""
    )
    st.title("Why use Deep Learning?")
    st.write(
        """Deep learning can be a powerful tool for predicting Bitcoin prices due to its ability to uncover complex patterns
      and relationships within vast amounts of data. The cryptocurrency market is highly volatile and influenced by various factors,
        such as market sentiment, economic indicators, and global events. Deep learning models can analyze historical price data,
          news articles, social media trends, and other relevant information to identify meaningful patterns that can help forecast
            future price movements. 
            
By leveraging the computational power of deep learning algorithms, we can potentially gain insights
    and make more informed decisions in the dynamic and unpredictable world of Bitcoin trading. However, it's important to
    note that cryptocurrency markets are inherently risky, and accurate price predictions are challenging, as they can be
        influenced by numerous unpredictable factors."""
    )
    st.title("FAQ")
    with st.expander("How accurate are the Bitcoin price forecasts on your website?"):
        st.write(
            "Our Bitcoin price forecasts are based on deep learning algorithms and historical data. While we strive for accuracy, please note that all forecasts carry inherent uncertainty and should not be solely relied upon for investment decisions."
        )
    with st.expander(
        "What methodology or algorithms do you use for your Bitcoin price predictions?"
    ):
        st.write(
            "The user can choose from a varierty of deep learning models to use for predicting the bitcoin prices. These models are CNN, LSTM and GRU and the number of layers can be selected too."
        )
    with st.expander("How frequently are the Bitcoin price forecasts updated?"):
        st.write("The dataset is updated daily and the model is trained each day too.")
    with st.expander(
        "Do you provide forecasts for other cryptocurrencies besides Bitcoin?"
    ):
        st.write(
            "At the moment we have only trained the model and tested using past exchange prices for Bitcoin, but we plan to train the model using other cryptocurrencies too and plan to allow the user to select which cryptocurrency's price they want the forecast for."
        )
    with st.expander(
        "Can I access historical Bitcoin price predictions on your website?"
    ):
        st.write(
            "We only store the current predictions. We plan to implement this in the future."
        )
    with st.expander(
        "Are there any disclaimers or limitations I should be aware of when using your Bitcoin price forecasts?"
    ):
        st.write(
            "Our Bitcoin price forecasts come with disclaimers and limitations. They should be used as informational tools and not as financial advice. Please read our terms and conditions for more details."
        )
    # with st.expander("Can I use your Bitcoin price forecasts for commercial purposes or investment advice?")
    # with st.expander("How can I interpret the forecasted Bitcoin prices provided on your website?")
    # with st.expander("Do you offer any educational resources or guides to help users understand your forecasting methodology?")
    # with st.expander("How can I contact you if I have further questions or need assistance with the Bitcoin price forecasts?")
    # with st.expander("Are there any subscription or membership options available for accessing premium features or more detailed forecasts?")
    # with st.expander("What measures do you take to ensure the security and privacy of user information on your website?")

    # Rithvik fill this if u can/want. You can remove questions if you want.

elif selected == "Contact":
    st.title("© The Neural Innovators")
    st.write("2019-2023")
    st.header("\nFor more queries, contact us at:")
    st.subheader("E-Mail")
    st.write("Anuj Suresh: anuj.suresh2019@vitstudent.ac.in")
    st.write("Rithvik Shankar: rithvik.shankar2019@vitstudent.ac.in")
    st.subheader("P.O. Box")
    st.write(
        """VIT,
Vellore Campus,
Vellore - 632 014"""
    )
    # st.header("Terms and Conditions")
    # st.markdown(
    #     "By using this webste, you are agreeing to the following [teams and conditions]()"
    # )
