import streamlit as st
import pandas as pd
import joblib
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Public Transport Delay Predictor",
    page_icon="🚌",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "transport_delay_model.pkl"
DATA_PATH = BASE_DIR / "transport_data.csv"


# ============================================================
# LOAD MODEL AND DATASET
# ============================================================

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

df["Bus_Number"] = df["Bus_Number"].astype(str)
df["Route"] = df["Route"].astype(str)


# ============================================================
# TITLE
# ============================================================

st.title("🚌 Real-Time Public Transport Delay Prediction System")

st.markdown(
    """
    **Machine Learning Based Public Transport Delay Prediction**
    
    Predict bus delays using traffic, weather, previous delay,
    distance and peak-hour conditions.
    """
)

st.divider()


# ============================================================
# CURRENT SYSTEM STATUS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🚌 Total Bus Routes",
        df["Bus_Number"].nunique()
    )

with col2:
    st.metric(
        "📊 Dataset Records",
        len(df)
    )

with col3:
    st.metric(
        "⏱️ Average Delay",
        f"{df['Actual_Delay_Min'].mean():.1f} min"
    )

with col4:
    high_traffic = len(df[df["Traffic"] == "High"])
    st.metric(
        "🚦 High Traffic Records",
        high_traffic
    )


st.divider()


# ============================================================
# LIVE PUBLIC TRANSPORT STATUS
# ============================================================

st.header("🔴 Live Public Transport Status")

st.caption(
    "Live status is simulated using the trained ML model."
)

if st.button("🔄 UPDATE LIVE STATUS"):

    live_data = df.sample(
        min(10, len(df)),
        random_state=random.randint(1, 10000)
    ).copy()

    predictions = model.predict(
        live_data[
            [
                "Route",
                "Traffic",
                "Weather",
                "Distance_KM",
                "Previous_Delay_Min",
                "Peak_Hour"
            ]
        ]
    )

    live_data["Predicted_Delay"] = (
        predictions + 
        [random.uniform(-1.5, 1.5) for _ in range(len(predictions))]
    ).round(1)

    def get_status(delay):

        if delay <= 5:
            return "🟢 ON TIME"

        elif delay <= 15:
            return "🟡 MODERATE DELAY"

        else:
            return "🔴 HIGH DELAY"

    live_data["Status"] = live_data["Predicted_Delay"].apply(
        get_status
    )

    display_data = live_data[
        [
            "Bus_Number",
            "Route",
            "Traffic",
            "Weather",
            "Predicted_Delay",
            "Status"
        ]
    ]

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# LIVE BUS TRACKING MAP
# ============================================================

st.header("🗺️ Live Bus Tracking")

st.caption(
    "Demo map using simulated bus locations around Coimbatore."
)

bus_locations = pd.DataFrame(
    {
        "Bus_Number": ["1", "101", "102", "102A", "103", "104"],
        "latitude": [
            11.0168,
            11.0200,
            11.0150,
            11.0250,
            11.0100,
            11.0300
        ],
        "longitude": [
            76.9558,
            76.9600,
            76.9500,
            76.9650,
            76.9580,
            76.9520
        ]
    }
)

st.map(
    bus_locations[
        ["latitude", "longitude"]
    ]
)


st.divider()


# ============================================================
# DELAY PREDICTION
# ============================================================

st.header("🤖 Bus Delay Prediction")

left, right = st.columns(2)


# ============================================================
# LEFT SIDE - BUS INFORMATION
# ============================================================

with left:

    bus_numbers = sorted(
        df["Bus_Number"].unique()
    )

    bus_number = st.selectbox(
        "🚌 Select Bus Number",
        bus_numbers
    )

    bus_routes = df[
        df["Bus_Number"] == bus_number
    ]["Route"].unique()

    route = st.selectbox(
        "🛣️ Select Route",
        bus_routes
    )

    traffic = st.selectbox(
        "🚦 Traffic Condition",
        ["Low", "Medium", "High"]
    )

    weather = st.selectbox(
        "🌦️ Weather Condition",
        ["Clear", "Cloudy", "Rain"]
    )


# ============================================================
# RIGHT SIDE - OTHER FEATURES
# ============================================================

with right:

    route_distance = df[
        (df["Bus_Number"] == bus_number) &
        (df["Route"] == route)
    ]["Distance_KM"].iloc[0]

    distance = st.number_input(
        "📏 Distance (KM)",
        min_value=1.0,
        max_value=1000.0,
        value=float(route_distance),
        step=1.0
    )

    previous_delay = st.number_input(
        "⏱️ Previous Delay (Minutes)",
        min_value=0.0,
        max_value=120.0,
        value=5.0,
        step=1.0
    )

    peak_hour = st.selectbox(
        "🕐 Peak Hour",
        ["Yes", "No"]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 PREDICT BUS DELAY",
    use_container_width=True
):

    input_data = pd.DataFrame(
        {
            "Route": [route],
            "Traffic": [traffic],
            "Weather": [weather],
            "Distance_KM": [distance],
            "Previous_Delay_Min": [previous_delay],
            "Peak_Hour": [peak_hour]
        }
    )

    prediction = model.predict(input_data)[0]

    prediction = max(
        0,
        round(float(prediction), 1)
    )

    st.success(
        f"🚌 Bus {bus_number} predicted delay: "
        f"**{prediction} minutes**"
    )

    if prediction <= 5:

        st.info(
            "🟢 Expected to arrive approximately on time."
        )

    elif prediction <= 15:

        st.warning(
            "🟡 Moderate delay expected."
        )

    else:

        st.error(
            "🔴 High delay expected. Please plan accordingly."
        )


st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📊 ML Model Performance")

try:

    X = df[
        [
            "Route",
            "Traffic",
            "Weather",
            "Distance_KM",
            "Previous_Delay_Min",
            "Peak_Hour"
        ]
    ]

    y = df["Actual_Delay_Min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    r2 = r2_score(
        y_test,
        test_predictions
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "🌳 Algorithm",
            "Random Forest"
        )

    with m2:
        st.metric(
            "📉 MAE",
            f"{mae:.2f} min"
        )

    with m3:
        st.metric(
            "🎯 R² Score",
            f"{r2:.2f}"
        )

    performance_df = pd.DataFrame(
        {
            "Actual Delay": y_test.values,
            "Predicted Delay": test_predictions
        }
    ).reset_index(drop=True)

    st.subheader(
        "Actual vs Predicted Delay"
    )

    st.line_chart(
        performance_df,
        use_container_width=True
    )

except Exception as e:

    st.error(
        f"Model performance calculation error: {e}"
    )


st.divider()


# ============================================================
# ROUTE-WISE DELAY ANALYSIS
# ============================================================

st.header("📈 Route-wise Average Delay")

chart_data = (
    df.groupby(
        "Route",
        as_index=False
    )["Actual_Delay_Min"]
    .mean()
    .sort_values(
        "Actual_Delay_Min",
        ascending=False
    )
    .head(20)
)

chart_data = chart_data.set_index("Route")

st.bar_chart(
    chart_data,
    y="Actual_Delay_Min",
    use_container_width=True
)


st.divider()


# ============================================================
# DATASET VIEWER
# ============================================================

st.header("📋 Transport Dataset")

with st.expander(
    "Click here to view dataset"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚌 Real-Time Public Transport Delay Prediction System | "
    "Random Forest + Streamlit"
)

st.caption(
    "Note: Traffic, weather, delay values and map locations "
    "are simulated/project data and are not live GPS or traffic feeds."
)
