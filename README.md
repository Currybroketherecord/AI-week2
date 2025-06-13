# **Climate Risk Prediction Using LSTM & PyTorch**  
**Author:** Micky  
**Date:** June 13, 2025  

## **📌 Project Overview**  
This project uses **Long Short-Term Memory (LSTM) neural networks** to analyze historical **climate change data** and predict future temperature trends. The model integrates **real-time climate data** from APIs, ensuring dynamic forecasting.  

---

## **📂 Dataset Information**  
The dataset, **climate_change_indicators.csv**, contains historical temperature records from **1961 to 2022** across multiple countries.  

### **Key Features:**  
✔ **Country Info** – ISO codes, region data  
✔ **Temperature Changes** – Annual variations (F1961 - F2022)  
✔ **Climate Indicators** – Baseline climatology, rolling averages  

---

## **⚙ Machine Learning Model Details**  
### **1️⃣ Data Preprocessing**  
✅ Convert non-numeric columns to numbers  
✅ Fill missing values using column means  
✅ Compute **Rolling Average Temperature** and **Temperature Deviation**  

### **2️⃣ Model Architecture**  
**Neural Network:** LSTM-based deep learning model  
**Layers:** 2 LSTM layers + Fully Connected output  
**Loss Function:** MSE (Mean Squared Error)  
**Optimizer:** Adam (learning rate: 0.001)  

### **3️⃣ Training & Evaluation**  
✅ **R² Score** for variance explanation  
✅ **MAPE (Mean Absolute Percentage Error)** for accuracy  
✅ **Epochs:** 100  

---

## **🔮 Forecasting Future Climate Trends**  
The trained LSTM model forecasts **temperature changes for the next 10 years** using test data.

---

## **🌍 Real-Time Data Integration**  
✔ Live temperature data fetched via **Open-Meteo API**  
✔ Integrated into the dataset dynamically  
✔ Ensures **real-time climate predictions**  

```python
url = "https://api.open-meteo.com/v1/forecast?latitude=-1.29&longitude=36.82&current=temperature_2m,wind_speed_10m"
response = requests.get(url)
data = response.json()
real_time_temp = data["current"]["temperature_2m"]
```

---

## **🚀 Deployment**  
### **1️⃣ Flask API** – Enables real-time predictions  
Run the Flask server:  
```sh
python app.py
```
### **2️⃣ Streamlit Web App** – Interactive visualization  
Launch the web app:  
```sh
streamlit run app.py
```

---

## **📢 Next Steps**  
✔ Optimize model performance (hyperparameter tuning)  
✔ Expand dataset with real-time climate updates  
✔ Deploy on **Heroku or Render** for global access  

---

## **💡 Usage**  
1️⃣ Clone the repository  
2️⃣ Install dependencies (`pip install -r requirements.txt`)  
3️⃣ Run `app.py` for API  
4️⃣ Use `streamlit run app.py` for interactive dashboard  

---

## **🔗 References**  
- Open-Meteo API: [https://open-meteo.com/](https://open-meteo.com/)  
- UN Climate Data: [https://sdg.data.gov](https://sdg.data.gov)  
