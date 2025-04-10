# senlytics — Sensor Data, Smarter Analytics

**`senlytics`** is a lightweight Python toolkit for fetching, cleaning, and analyzing data from low-cost environmental sensors — with a focus on air quality.

Whether you're running community science projects, research deployments, or smart city trials, `senlytics` gives you a streamlined way to work with sensor networks like **AirGradient**, and easily generate insights from raw data.

---

## 🔧 Features

- 📡 **Batch + parallel fetching** from AirGradient API (`raw` or `past` mode)
- 📊 **Automated cleaning** and resampling
- 🧪 **Built-in performance metrics** (bias, RMSE, FAC2, R², and more)
- 🖼️ **Interactive and static plotting** (Holoviews & Matplotlib)
- 📈 **Diurnal, weekday/weekend, and comparative analytics**
- 🛠️ **Ready-to-deploy pipelines** for research and real-world deployment

---

## 🔍 Example Use Cases

- Academic research on urban air pollution
- Citizen science sensor networks
- Evaluating sensor calibration against reference instruments
- Generating rapid time-series summaries and plots

---

## 🧪 Supported Inputs

- AirGradient sensors (CO₂, PM1/2.5/10, TVOC, Temp/RH, NOx index)
- Any dataset in similar tabular format with timestamps and pollutant columns
