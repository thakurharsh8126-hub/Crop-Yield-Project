import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import CropYieldModel, DATA_PATH



st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown(
    """
    <style>
        .stApp {
            background-color: #0f172a;
            color: #e2e8f0;
        }

        section[data-testid="stSidebar"] {
            background-color: #1e293b;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-bottom: 0;
        }

        .subtitle {
            color: #94a3b8;
            font-size: 1rem;
            margin-bottom: 25px;
        }

        .metric-card {
            background-color: #1e293b;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #334155;
            text-align: center;
        }

        .result-card {
            background-color: #0f2e1a;
            padding: 25px;
            border-radius: 12px;
            border: 2px solid #22c55e;
            margin-top: 15px;
        }

        .result-value {
            font-size: 3rem;
            font-weight: 700;
            color: #22c55e;
        }

        .result-label {
            color: #94a3b8;
            font-size: 0.9rem;
            font-weight: 600;
        }

        .result-message {
            color: #e2e8f0;
            margin-top: 10px;
        }

        div[data-testid="stMetric"] {
            background-color: #1e293b;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #334155;
        }
    </style>
    """,
    unsafe_allow_html=True
)




if not os.path.exists(DATA_PATH):
    import generate_data  # noqa: F401



@st.cache_resource
def load_model():
    model = CropYieldModel()
    metrics = model.train()
    return model, metrics


try:
    cy_model, metrics = load_model()
    df = cy_model.df

except Exception as e:
    st.error("Unable to load the crop yield model.")
    st.exception(e)
    st.stop()



st.markdown(
    '<div class="main-title">🌾 Crop Yield Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based crop yield prediction and analysis</div>',
    unsafe_allow_html=True
)



col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model R²",
        f"{metrics['r2']:.3f}"
    )

with col2:
    st.metric(
        "MAE",
        f"{metrics['mae']:.2f} t/ha"
    )

with col3:
    st.metric(
        "RMSE",
        f"{metrics['rmse']:.2f} t/ha"
    )


st.divider()



predict_tab, dashboard_tab, model_tab = st.tabs(
    [
        "🔮 Predict Yield",
        "📊 Data Dashboard",
        "🧠 Model Insights"
    ]
)



with predict_tab:

    st.subheader("Enter Field Conditions")

    left, right = st.columns([1, 1.6])

    # -----------------------------------------------------
    # Input Form
    # -----------------------------------------------------
    with left:

        rainfall = st.slider(
            "Rainfall (mm)",
            min_value=200.0,
            max_value=3000.0,
            value=1200.0,
            step=10.0
        )

        temperature = st.slider(
            "Temperature (°C)",
            min_value=10.0,
            max_value=45.0,
            value=25.0,
            step=0.5
        )

        humidity = st.slider(
            "Humidity (%)",
            min_value=20.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

        fertilizer = st.slider(
            "Fertilizer Usage (kg/ha)",
            min_value=0.0,
            max_value=300.0,
            value=100.0,
            step=5.0
        )

        soil_types = cy_model.get_soil_types()

        crop_types = cy_model.get_crop_types()

        soil = st.selectbox(
            "Soil Type",
            soil_types
        )

        crop = st.selectbox(
            "Crop Type",
            crop_types
        )

        predict_clicked = st.button(
            "🌾 Predict Yield",
            type="primary",
            use_container_width=True
        )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    with right:

        if predict_clicked:

            try:

                pred = cy_model.predict(
                    rainfall,
                    temperature,
                    humidity,
                    soil,
                    fertilizer,
                    crop
                )

                crop_yields = df[
                    df["Crop_Type"] == crop
                ]["Yield_tons_per_ha"]

                avg = crop_yields.mean()

                percentile = (
                    crop_yields < pred
                ).mean() * 100

                # -----------------------------------------
                # Determine quality
                # -----------------------------------------
                if percentile >= 75:

                    level = "Excellent"

                    msg = (
                        f"This is an excellent yield — better than "
                        f"{percentile:.0f}% of {crop} results in the "
                        f"dataset. Current rainfall, temperature, "
                        f"humidity, fertilizer and soil conditions "
                        f"are working well together."
                    )

                elif percentile >= 50:

                    level = "Good"

                    msg = (
                        f"This is a good yield — above the average "
                        f"of {avg:.2f} t/ha for {crop} and better "
                        f"than {percentile:.0f}% of comparable results. "
                        f"There's still some room to optimize."
                    )

                elif percentile >= 25:

                    level = "Below Average"

                    msg = (
                        f"This yield is below the {crop} average "
                        f"of {avg:.2f} t/ha (higher than only "
                        f"{percentile:.0f}% of results). Consider "
                        f"adjusting fertilizer usage or checking if "
                        f"soil type suits {crop}."
                    )

                else:

                    level = "Poor"

                    msg = (
                        f"This yield is significantly below the "
                        f"{crop} average of {avg:.2f} t/ha — only "
                        f"{percentile:.0f}% of results are this low "
                        f"or lower. Rainfall, temperature or "
                        f"fertilizer levels may be unfavorable for "
                        f"{crop} under these conditions."
                    )

                # -----------------------------------------
                # Result Card
                # -----------------------------------------
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            PREDICTED YIELD — {crop.upper()}
                        </div>

                        <div class="result-value">
                            {pred:.2f}
                        </div>

                        <div class="result-label">
                            tons / hectare
                        </div>

                        <br>

                        <strong>Assessment: {level}</strong>

                        <div class="result-message">
                            {msg}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                # -----------------------------------------
                # Prediction vs Benchmarks
                # -----------------------------------------
                mx = crop_yields.max()

                chart_df = pd.DataFrame(
                    {
                        "Category": [
                            "Your Prediction",
                            f"{crop} Average",
                            f"{crop} Maximum"
                        ],
                        "Yield": [
                            pred,
                            avg,
                            mx
                        ]
                    }
                )

                fig, ax = plt.subplots(figsize=(8, 4))

                ax.bar(
                    chart_df["Category"],
                    chart_df["Yield"]
                )

                ax.set_ylabel("tons/hectare")
                ax.set_title(
                    f"Predicted Yield vs {crop} Benchmarks"
                )

                ax.tick_params(axis="x", rotation=15)

                for i, value in enumerate(chart_df["Yield"]):
                    ax.text(
                        i,
                        value,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom"
                    )

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

                # -----------------------------------------
                # Feature Importance
                # -----------------------------------------
                st.subheader(
                    "Relative Influence of Each Factor"
                )

                fi = cy_model.feature_importance.sort_values(
                    ascending=True
                )

                fig2, ax2 = plt.subplots(figsize=(8, 4))

                ax2.barh(
                    fi.index,
                    fi.values
                )

                ax2.set_xlabel(
                    "Importance Score"
                )

                ax2.set_title(
                    "Feature Importance"
                )

                fig2.tight_layout()

                st.pyplot(
                    fig2,
                    use_container_width=True
                )

                plt.close(fig2)

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)

        else:

            st.info(
                "Enter the field conditions and click "
                "**Predict Yield** to see the result."
            )

            # ---------------------------------------------
            # Default charts
            # ---------------------------------------------
            st.subheader(
                "Model-wide Feature Importance"
            )

            fi = cy_model.feature_importance.sort_values(
                ascending=True
            )

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.barh(
                fi.index,
                fi.values
            )

            ax.set_xlabel(
                "Importance"
            )

            ax.set_title(
                "Feature Importance"
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


# =========================================================
# DATA DASHBOARD TAB
# =========================================================
with dashboard_tab:

    st.subheader("📊 Training Data Dashboard")

    # -----------------------------------------------------
    # Dataset overview
    # -----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with c2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with c3:
        st.metric(
            "Crop Types",
            df["Crop_Type"].nunique()
        )

    with c4:
        st.metric(
            "Soil Types",
            df["Soil_Type"].nunique()
        )

    st.divider()

    # -----------------------------------------------------
    # Charts
    # -----------------------------------------------------
    col1, col2 = st.columns(2)

    # Soil distribution
    with col1:

        st.markdown("### Soil Type Distribution")

        soil_counts = df[
            "Soil_Type"
        ].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            soil_counts.values,
            labels=soil_counts.index,
            autopct="%1.0f%%"
        )

        ax.set_title(
            "Soil Type Distribution"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # Average crop yield
    with col2:

        st.markdown("### Average Yield by Crop")

        avg_yield = (
            df.groupby("Crop_Type")[
                "Yield_tons_per_ha"
            ]
            .mean()
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.barh(
            avg_yield.index,
            avg_yield.values
        )

        ax.set_xlabel(
            "tons/ha"
        )

        ax.set_title(
            "Average Yield by Crop"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # Rainfall vs yield
    col3, col4 = st.columns(2)

    with col3:

        st.markdown("### Rainfall vs Yield")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(
            df["Rainfall_mm"],
            df["Yield_tons_per_ha"],
            s=8,
            alpha=0.4
        )

        ax.set_xlabel(
            "Rainfall (mm)"
        )

        ax.set_ylabel(
            "Yield (t/ha)"
        )

        ax.set_title(
            "Rainfall vs Yield"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # Temperature vs yield
    with col4:

        st.markdown("### Temperature vs Yield")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(
            df["Temperature_C"],
            df["Yield_tons_per_ha"],
            s=8,
            alpha=0.4
        )

        ax.set_xlabel(
            "Temperature (°C)"
        )

        ax.set_ylabel(
            "Yield (t/ha)"
        )

        ax.set_title(
            "Temperature vs Yield"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # Soil yield
    col5, col6 = st.columns(2)

    with col5:

        st.markdown("### Average Yield by Soil Type")

        soil_yield = (
            df.groupby("Soil_Type")[
                "Yield_tons_per_ha"
            ]
            .mean()
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            soil_yield.index,
            soil_yield.values
        )

        ax.set_ylabel(
            "tons/ha"
        )

        ax.set_title(
            "Average Yield by Soil Type"
        )

        ax.tick_params(
            axis="x",
            rotation=30
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # Yield distribution
    with col6:

        st.markdown("### Yield Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(
            df["Yield_tons_per_ha"],
            bins=25
        )

        ax.set_xlabel(
            "Yield (tons/ha)"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.set_title(
            "Yield Distribution"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # -----------------------------------------------------
    # Raw data
    # -----------------------------------------------------
    with st.expander("View Training Dataset"):

        st.dataframe(
            df,
            use_container_width=True
        )


# =========================================================
# MODEL INSIGHTS TAB
# =========================================================
with model_tab:

    st.subheader("🧠 Model Insights")

    st.markdown(
        """
        ### Random Forest Regressor

        The crop yield prediction system uses a
        **Random Forest Regression** model.

        The model configuration used by the existing
        application is:

        - **Trees:** 200
        - **Maximum Depth:** 12
        """
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Test R²",
            f"{metrics['r2']:.3f}"
        )

    with c2:
        st.metric(
            "MAE",
            f"{metrics['mae']:.2f}"
        )

    with c3:
        st.metric(
            "RMSE",
            f"{metrics['rmse']:.2f}"
        )

    st.divider()

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Feature Importance")

        fi = (
            cy_model.feature_importance
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.barh(
            fi.index,
            fi.values
        )

        ax.set_xlabel(
            "Importance Score"
        )

        ax.set_title(
            "Feature Importance"
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # -----------------------------------------------------
    # Actual vs Predicted
    # -----------------------------------------------------
    with col2:

        st.markdown(
            "### Predicted vs Actual"
        )

        y_test = np.asarray(
            cy_model.y_test
        )

        preds = np.asarray(
            cy_model.preds
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            y_test,
            preds,
            s=10,
            alpha=0.5
        )

        max_value = max(
            y_test.max(),
            preds.max()
        ) + 0.5

        ax.plot(
            [0, max_value],
            [0, max_value],
            linestyle="--",
            linewidth=1.5
        )

        ax.set_xlim(
            0,
            max_value
        )

        ax.set_ylim(
            0,
            max_value
        )

        ax.set_xlabel(
            "Actual Yield (t/ha)"
        )

        ax.set_ylabel(
            "Predicted Yield (t/ha)"
        )

        ax.set_title(
            "Predicted vs Actual (Test Set)"
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)
