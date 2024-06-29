# Import các thư viện cần thiết để sử dụng
# streamlit để tạo giao diện web/app 
import streamlit as st
# datetime để lấy ra ngày giờ hiện tại
from datetime import date
# yfinance để tải dữ liệu tài chính từ BTC-USD, ETH-USD hoặc là ADA-USD 
import yfinance as yf
# prophet để dự báo thời gian và giá cả 
from prophet import Prophet
from prophet.plot import plot_plotly
# plotly để vẽ biểu đồ chart 
from plotly import graph_objs as go

# Thiết lập ngày bắt đầu lấy dữ liệu và ngày hiện tại để giới hạn khoảng thời gian dữ liệu
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Giao diện web/app sử dụng Streamlit để hiện thị , một title, một select box chọn cặp tiền dự đoán và 1 slider để chọn số năm dự đoán

# Tạo một đối tượng stock_title để hiện thị theo cặp tiền dự đoán
stock_titles = {
    'BTC-USD' : 'Bitcoin Price Prediction Model',
    'ETH-USD' : 'Ethereum Price Prediction Model',
    'ADA-USD' : 'Cardano Price Prediction Model'
}

# Select pair 
stocks = ('BTC-USD', 'ETH-USD','ADA-USD')
selected_stock = st.selectbox('Select the currency pair to predict', stocks)

# Lấy tiêu đề tương ứng từ từ điển
st.title(stock_titles[selected_stock])

# Số năm dự đoán 
n_years = st.slider('Number of years to predict', 1, 4)
period = n_years * 365

# Hàm tải dữ liệu từ trên yfinance dựa trên cặp tiền dự đoán, ngày bắt đầu và ngày hiện tại để lấy dữ liệu
# Tải và lưu trữ nó trong bộ nhớ cache để tăng tốc độ tải dữ liệu
@st.cache_data 
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Hiển thị thông báo đang tải dữ liệu và sau khi hoàn thành sẽ hiện thị thông tin của cặp tiền đó trong 5 ngày gần nhất	
stock_names = {
    'BTC-USD' : 'Bitcoin',
    'ETH-USD' : 'Ethereum',
    'ADA-USD' : 'Cardano'
}
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text(f'{stock_names[selected_stock]} price data for the last 5 days')
st.write(data.tail())

# Vẽ biểu đồ giá mở và giá đóng của dữ liệu bằng Plotly 
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Opening price"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Closing price"))
	fig.layout.update(title_text='Time Series Data ', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Chuẩn bị dữ liệu cho mô hình dự đoán : Bằng cách chọn cột `Date` và `Close` và đổi tên chúng để phù hợp với yêu cầu của Prophet
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train.head()

# Tạo mô hình Prophet, huấn luyện nó với dữ liệu , tạo khung dữ liệu tương lai và dự báo 
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Hiển thị kết quả dự báo , biểu đồ dự báo và các thành phần của dự báo như trend, weekly, yearly 
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Components")
fig2 = m.plot_components(forecast)
st.write(fig2)