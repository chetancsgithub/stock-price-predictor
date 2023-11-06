import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

#hello 

st.markdown("""
<style>
    [data-testid="stAppViewContainer"]{
        background-image: url("https://images.pexels.com/photos/187041/pexels-photo-187041.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2");
        background-size: cover;
    }
</style>
""",
unsafe_allow_html=True
)

start = '2015-01-01'
today = date.today().strftime("%Y-%m-%d")

custom_color = "#FFFFFF"
st.markdown(f"<h1 style='color:{custom_color}'>Stock Prediction App</h1>", unsafe_allow_html=True)

stocks = ("AAPL","GOOG","MSFT","GME","TCS.NS","TATAMOTORS.NS","TATASTEEL.NS","TATAPOWER.NS","TSLA","TSLA.NE","PNB.NS","ADANIENT.NS","ADANIPOWER.NS","ADANIPORTS.NS","ADANIGREEN.NS","AMZN","SBI","YESBANK.NS","UNIONBANK.NS")
st.markdown(f"<style>.stSelectbox label {{color:{custom_color};}}</style>", unsafe_allow_html=True)
selected_stocks = st.selectbox("Select dataset for prediction",stocks)


st.markdown(
    f"""
    <style>
        .stSlider > div > div > div > div {{
            background-color: {custom_color} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(f"<style>.stSlider label {{color:{custom_color};}}</style>", unsafe_allow_html=True)
n_years = st.slider( "Years of Prediction", 1, 7)
period = n_years * 365

def load_data(ticker):
    data = yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data

data_load_state =st.markdown(f"<p style='color: white;'>Loading data...</p>", unsafe_allow_html=True)
data = load_data(selected_stocks)
data_load_state.text(st.markdown(f"<p style='color: white;'>Loading data...Done!</p>", unsafe_allow_html=True))


st.markdown(f"<h1 style='color:{custom_color}'>Raw Data</h1>", unsafe_allow_html=True)


st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_Close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.markdown(f"<h1 style='color:{custom_color}'>Forecast data</h1>", unsafe_allow_html=True)
st.write(forecast.tail())

st.markdown(f"<p style='color:{custom_color}'>forecast data</p>", unsafe_allow_html=True)
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.markdown(f"<p style='color:{custom_color}'>forecast components</p>", unsafe_allow_html=True)
fig2 = m.plot_components(forecast)
st.write(fig2)


