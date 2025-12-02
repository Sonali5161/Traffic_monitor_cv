import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time
import numpy as np
import pandas as pd
import altair as alt
from fpdf import FPDF
import requests

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="Smart Traffic Monitoring System", page_icon="🚦", layout="wide")

st.title("🚦 Smart Traffic Monitoring Dashboard")
st.markdown("### YOLOv8 + OpenCV + Streamlit")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
source = st.sidebar.radio("Select Source", ["Webcam", "Upload Video"])
confidence = st.sidebar.slider("Detection Confidence", 0.25, 1.0, 0.5)
api_key = "212f49dc03a89da1074900f3c9497f56"

city = st.sidebar.text_input("🏙️ Enter City Name for Weather", "Mumbai")

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Initialize global stats
vehicle_counts = []
time_stamps = []
total_count = 0
peak_density = "Smooth"

# Utility function
def get_density_label(count):
    if count < 5:
        return "🟢 Smooth"
    elif count < 15:
        return "🟠 Moderate"
    else:
        return "🔴 Heavy"

# --------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Detection", "📊 Analytics", "🧾 Reports", "🌦️ Weather"])

# --------------- TAB 1: Live Detection ----------------
with tab1:
    col1, col2, col3 = st.columns(3)
    total_placeholder = col1.metric("🛻 Vehicles", 0)
    density_placeholder = col2.metric("🚦 Traffic Density", "Smooth")
    fps_placeholder = col3.metric("⚡ FPS", "0")

    stframe = st.empty()
    st.markdown("#### 🚨 Violation Detection (Crossing Red Line)")
    st.info("If a vehicle crosses the red line, it will be marked as a violation.")

    if source == "Upload Video":
        video_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            vid = cv2.VideoCapture(tfile.name)
            prev_time = 0

            while True:
                success, frame = vid.read()
                if not success:
                    break

                # Draw red line for rule violation
                height, width, _ = frame.shape
                line_y = int(height * 0.7)
                cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)

                results = model.predict(frame, conf=confidence, stream=True, verbose=False)
                count = 0
                violation = False

                for r in results:
                    frame = r.plot()
                    boxes = r.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        count += 1
                        # Violation check
                        if cy > line_y:
                            violation = True
                            cv2.putText(frame, "🚨 Violation!", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                total_count = count
                vehicle_counts.append(count)
                time_stamps.append(time.strftime("%H:%M:%S"))

                # FPS calculation
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
                prev_time = current_time

                # Density
                density_label = get_density_label(total_count)
                peak_density = density_label

                # Update metrics
                total_placeholder.metric("🛻 Vehicles", total_count)
                density_placeholder.metric("🚦 Traffic Density", density_label)
                fps_placeholder.metric("⚡ FPS", f"{fps:.1f}")

                if density_label == "🔴 Heavy":
                    st.warning("🚨 Heavy Traffic Alert! Consider alternate route.")

                if violation:
                    st.error("🛑 Rule Violation Detected!")

                stframe.image(frame, channels="BGR", use_container_width=True)

            vid.release()

    elif source == "Webcam":
        run = st.checkbox("Start Webcam Detection")
        if run:
            cap = cv2.VideoCapture(0)
            prev_time = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break
                results = model(frame, conf=confidence, verbose=False)
                frame = results[0].plot()
                count = len(results[0].boxes.cls)

                vehicle_counts.append(count)
                time_stamps.append(time.strftime("%H:%M:%S"))

                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
                prev_time = current_time

                density_label = get_density_label(count)
                total_placeholder.metric("🛻 Vehicles", count)
                density_placeholder.metric("🚦 Traffic Density", density_label)
                fps_placeholder.metric("⚡ FPS", f"{fps:.1f}")

                if density_label == "🔴 Heavy":
                    st.warning("🚨 Heavy Traffic Alert!")

                stframe.image(frame, channels="BGR", use_container_width=True)
            cap.release()

# --------------- TAB 2: Analytics ----------------
with tab2:
    st.subheader("📊 Real-Time Vehicle Analytics")
    if len(vehicle_counts) > 0:
        df = pd.DataFrame({"Time": time_stamps, "Vehicle_Count": vehicle_counts})
        chart = alt.Chart(df).mark_line(point=True).encode(x="Time", y="Vehicle_Count")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Run detection first to see analytics.")

# --------------- TAB 3: Reports ----------------
with tab3:
    st.subheader("🧾 Generate Traffic Report (PDF)")
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="Smart Traffic Monitoring Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Total Vehicles Detected: {total_count}", ln=True)
        pdf.cell(200, 10, txt=f"Peak Traffic Status: {peak_density}", ln=True)
        pdf.cell(200, 10, txt=f"Report Generated At: {time.strftime('%H:%M:%S')}", ln=True)
        pdf.output("Traffic_Report.pdf")

        with open("Traffic_Report.pdf", "rb") as file:
            st.download_button("📄 Download Daily Report", data=file, file_name="Traffic_Report.pdf")

# --------------- TAB 4: Weather ----------------
with tab4:
    st.subheader("🌦️ Weather-Aware Traffic Prediction")
    if api_key:
        try:
            res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric").json()
            temp = res["main"]["temp"]
            condition = res["weather"][0]["main"]
            st.metric("🌡️ Temperature (°C)", temp)
            st.metric("☁️ Condition", condition)

            if "rain" in condition.lower():
                st.warning("🌧️ Rainy weather may cause heavier traffic.")
            elif "clear" in condition.lower():
                st.success("☀️ Clear weather — smooth traffic expected.")
            else:
                st.info("🌤️ Moderate traffic conditions possible.")
        except Exception as e:
            st.error(f"⚠️ Unable to fetch weather data. Error: {e}")
