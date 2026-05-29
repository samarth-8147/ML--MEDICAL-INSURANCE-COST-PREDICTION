import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import io

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Insurance Prediction", layout="centered")

# -----------------------------
# ANIMATED STYLE
# -----------------------------
st.markdown("""
<style>

/* Animated Hospital Gradient */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #1c92d2, #2c5364, #00c6ff);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}

/* Smooth Gradient Animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Heading Colors */
h1, h2, h3 {
    color: black;
    text-shadow: 0px 0px 10px rgba(0,255,255,0.6);
}

/* Button Styling */
div.stButton > button {
    background-color: #00e5ff;
    color: black;
    border-radius: 10px;
    transition: 0.3s;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #00bcd4;
    transform: scale(1.05);
}

/* Card Effect for Sections */
.section-card {
    background: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
    box-shadow: 0px 0px 15px rgba(0,255,255,0.3);
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# HEADER SECTION
# -----------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🏥 Medical Insurance Cost Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Machine Learning Based Insurance Prediction System</p>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# LOAD DATA & MODEL
# -----------------------------
data = pd.read_csv("insurance.csv")
model = pickle.load(open("model.pkl", "rb"))

region_map = {
    "southeast": 0,
    "southwest": 1,
    "northeast": 2,
    "northwest": 3
}

# -----------------------------
# INPUT SECTION
# -----------------------------
st.markdown('<div class="section-animate">', unsafe_allow_html=True)

st.header("📝 Enter Patient Details")

age = st.number_input("Age", 1, 100, 25)
sex = st.selectbox("Gender", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Number of Children", 0, 10, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", list(region_map.keys()))

st.markdown('</div>', unsafe_allow_html=True)

# Encoding
sex_encoded = 0 if sex == "male" else 1
smoker_encoded = 1 if smoker == "yes" else 0
region_encoded = region_map[region]

prediction_value = None

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Insurance Cost"):
    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    prediction = model.predict(input_data)[0]
    prediction_value = prediction
    st.success(f"💰 Predicted Insurance Cost: ₹ {prediction:,.2f}")

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.markdown('<div class="section-animate">', unsafe_allow_html=True)

data_encoded = data.copy()
data_encoded['sex'] = data_encoded['sex'].map({'male': 0, 'female': 1})
data_encoded['smoker'] = data_encoded['smoker'].map({'yes': 1, 'no': 0})
data_encoded['region'] = data_encoded['region'].map(region_map)

X = data_encoded.drop("charges", axis=1)
y = data_encoded["charges"]
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

st.header("📈 Model Performance")
st.write(f"R² Score: {r2:.3f}")
st.write(f"RMSE: {rmse:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# DATA VISUALIZATION
# -----------------------------
st.markdown('<div class="section-animate">', unsafe_allow_html=True)

st.header("📊 Data Visualization")

fig, ax = plt.subplots()
sns.histplot(data['charges'], kde=True, ax=ax)
ax.set_title("Charges Distribution")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# PDF GENERATION
# -----------------------------
if prediction_value is not None:

    def generate_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("<b>Medical Insurance Prediction Report</b>", styles["Title"]))
        elements.append(Spacer(1, 20))

        report = f"""
        Age: {age}<br/>
        Gender: {sex}<br/>
        BMI: {bmi}<br/>
        Children: {children}<br/>
        Smoker: {smoker}<br/>
        Region: {region}<br/><br/>
        <b>Predicted Insurance Cost: ₹ {prediction_value:,.2f}</b><br/><br/>
        Model Used: Random Forest Regressor<br/>
        R2 Score: {r2:.3f}<br/>
        RMSE: {rmse:.2f}
        """

        elements.append(Paragraph(report, styles["Normal"]))
        elements.append(Spacer(1, 20))

        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        elements.append(Image(img_buffer, 5*inch, 3*inch))
        doc.build(elements)

        buffer.seek(0)
        return buffer

    pdf_file = generate_pdf()

    st.download_button(
        label="📄 Download Full Report (PDF with Graph)",
        data=pdf_file,
        file_name="Insurance_Report.pdf",
        mime="application/pdf"
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Machine Learning Project")