
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="🍫",
    layout="wide"
)


# ============================================================
# FACTORY COORDINATES FROM PROJECT DOCUMENTATION
# ============================================================
FACTORIES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107)
}

FACTORY_NAMES = list(FACTORIES.keys())


# ============================================================
# PRODUCT -> CURRENT FACTORY
# The project documentation explicitly shows these mappings.
# For products not listed there, the fallback is used.
# ============================================================
PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",

    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",

    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",

    "Everlasting Gobstopper": "Secret Factory"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_column(df, names):
    """Find a column ignoring case and surrounding spaces."""
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.strip().lower() in normalized:
            return normalized[name.strip().lower()]
    return None


def assign_current_factory(row):
    """Use documented product mapping; use a transparent fallback."""
    product = str(row["Product Name"]).strip()

    if product in PRODUCT_FACTORY_MAP:
        return PRODUCT_FACTORY_MAP[product]

    # The project screenshots do not show mappings for every product.
    # Use division as a fallback so the app remains executable.
    if row["Division"] == "Chocolate":
        return "Lot's O' Nuts"
    if row["Division"] in ["Sugar", "Other"]:
        return "Sugar Shack"

    return "The Other Factory"


def prepare_data(raw):
    df = raw.copy()

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")

    order_col = find_column(df, ["Order Date"])
    ship_col = find_column(df, ["Ship Date"])

    if order_col is None or ship_col is None:
        raise ValueError("Order Date and Ship Date columns are required.")

    # The uploaded dataset uses DD-MM-YYYY dates.
    df["Order Date"] = pd.to_datetime(
        df[order_col], dayfirst=True, errors="coerce"
    )
    df["Ship Date"] = pd.to_datetime(
        df[ship_col], dayfirst=True, errors="coerce"
    )

    df["Lead Time"] = (
        df["Ship Date"] - df["Order Date"]
    ).dt.days

    # Required categorical columns
    for col in ["Product Name", "Division", "Region", "Ship Mode"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].astype(str).str.strip()

    # Numeric columns
    for col in ["Sales", "Units", "Gross Profit", "Cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Current factory used by the optimization simulation.
    df["Current Factory"] = df.apply(assign_current_factory, axis=1)

    # Remove invalid lead times only.
    df = df.dropna(subset=["Lead Time"]).copy()
    df = df[df["Lead Time"] >= 0].copy()

    return df


def remove_extreme_outliers(df):
    """Remove only extreme lead-time values for model training."""
    if len(df) < 20:
        return df.copy()

    low = df["Lead Time"].quantile(0.01)
    high = df["Lead Time"].quantile(0.99)

    return df[
        (df["Lead Time"] >= low) &
        (df["Lead Time"] <= high)
    ].copy()


@st.cache_resource
def train_models(data):
    """
    Train the three models requested by the project:
    Linear Regression, Random Forest, Gradient Boosting.
    """
    model_data = remove_extreme_outliers(data)

    features = [
        "Product Name",
        "Division",
        "Region",
        "Ship Mode",
        "Current Factory",
        "Sales",
        "Units",
        "Cost"
    ]

    features = [c for c in features if c in model_data.columns]

    X = model_data[features].copy()
    y = model_data["Lead Time"].astype(float)

    categorical = [
        c for c in features
        if X[c].dtype == "object"
    ]

    numerical = [
        c for c in features
        if c not in categorical
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ))
                ]),
                categorical
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numerical
            )
        ],
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    results = []
    fitted_models = {}

    for name, model in models.items():

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        fitted_models[name] = pipe

    results_df = pd.DataFrame(results).sort_values("RMSE")

    # Lowest RMSE = best predictive model.
    best_name = results_df.iloc[0]["Model"]

    return fitted_models, results_df, best_name, features


def prediction_frame(product, division, region, ship_mode, factory,
                      avg_sales, avg_units, avg_cost):
    return pd.DataFrame([{
        "Product Name": product,
        "Division": division,
        "Region": region,
        "Ship Mode": ship_mode,
        "Current Factory": factory,
        "Sales": avg_sales,
        "Units": avg_units,
        "Cost": avg_cost
    }])


def simulate_factories(
    models,
    product,
    division,
    region,
    ship_mode,
    current_factory,
    avg_sales,
    avg_units,
    avg_cost
):
    rows = []

    for factory in FACTORY_NAMES:

        X_new = prediction_frame(
            product,
            division,
            region,
            ship_mode,
            factory,
            avg_sales,
            avg_units,
            avg_cost
        )

        for model_name, model in models.items():
            predicted = float(model.predict(X_new)[0])

            rows.append({
                "Factory": factory,
                "Model": model_name,
                "Predicted Lead Time": max(0, predicted)
            })

    sim = pd.DataFrame(rows)

    # Use Random Forest for simulation when available.
    preferred = (
        "Random Forest"
        if "Random Forest" in models
        else list(models.keys())[0]
    )

    sim_best = sim[sim["Model"] == preferred].copy()
    sim_best = sim_best.drop(columns=["Model"])

    current_time = float(
        sim_best.loc[
            sim_best["Factory"] == current_factory,
            "Predicted Lead Time"
        ].iloc[0]
    )

    sim_best["Lead Time Reduction"] = (
        current_time - sim_best["Predicted Lead Time"]
    )

    sim_best["Lead Time Reduction %"] = np.where(
        current_time != 0,
        sim_best["Lead Time Reduction"] / current_time * 100,
        0
    )

    sim_best["Risk Score"] = (
        sim_best["Predicted Lead Time"] /
        max(sim_best["Predicted Lead Time"].max(), 1)
    ) * 100

    sim_best["Historical Gross Profit"] = avg_sales - avg_cost

    sim_best["Factory Change"] = np.where(
        sim_best["Factory"] == current_factory,
        "Current",
        "Alternative"
    )

    return sim_best.sort_values(
        "Predicted Lead Time"
    ).reset_index(drop=True)


# ============================================================
# HEADER
# ============================================================
st.title("🍫 Factory Reallocation & Shipping Optimization")
st.caption(
    "Nassau Candy Distributor — Predictive Modeling + What-If Simulation + Recommendations"
)


# ============================================================
# FILE UPLOAD
# ============================================================
st.sidebar.header("📁 Dataset")

uploaded = st.sidebar.file_uploader(
    "Upload Nassau Candy CSV",
    type=["csv"]
)

if uploaded is None:
    st.info("Upload **Nassau Candy Distributor-1.csv** from the sidebar to start.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
    df = prepare_data(raw_df)
except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()


# ============================================================
# DATA SUMMARY
# ============================================================
st.success(f"Dataset loaded successfully: {len(df):,} valid rows")


# ============================================================
# TRAIN MODELS
# ============================================================
with st.spinner("Training Linear Regression, Random Forest and Gradient Boosting..."):
    models, results_df, best_model, feature_columns = train_models(df)


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("🎛️ Filters")

product_options = sorted(df["Product Name"].unique())
region_options = sorted(df["Region"].unique())
ship_options = sorted(df["Ship Mode"].unique())

selected_product = st.sidebar.selectbox(
    "Product",
    product_options
)

selected_region = st.sidebar.selectbox(
    "Destination Region",
    region_options
)

selected_ship_mode = st.sidebar.selectbox(
    "Ship Mode",
    ship_options
)

priority = st.sidebar.slider(
    "Optimization Priority: Speed ↔ Profit",
    min_value=0,
    max_value=100,
    value=70,
    help="0 = more profit stability, 100 = more emphasis on shipping speed."
)


# ============================================================
# SELECTED PRODUCT INFORMATION
# ============================================================
product_rows = df[
    df["Product Name"] == selected_product
].copy()

division = product_rows["Division"].mode().iloc[0]

current_factory = (
    product_rows["Current Factory"].mode().iloc[0]
)

avg_sales = float(product_rows["Sales"].mean())
avg_units = float(product_rows["Units"].mean())
avg_cost = float(product_rows["Cost"].mean())
avg_profit = float(product_rows["Gross Profit"].mean())


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔎 EDA",
    "🤖 Prediction",
    "🏭 Factory Simulation",
    "💡 Recommendations"
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:

    st.header("Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Orders",
            f"{df['Order ID'].nunique():,}"
            if "Order ID" in df.columns
            else f"{len(df):,}"
        )

    with c2:
        st.metric(
            "Average Lead Time",
            f"{df['Lead Time'].mean():.2f} days"
        )

    with c3:
        st.metric(
            "Total Sales",
            f"${df['Sales'].sum():,.2f}"
        )

    with c4:
        st.metric(
            "Average Gross Profit",
            f"${df['Gross Profit'].mean():,.2f}"
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Lead Time by Region")

        region_chart = (
            df.groupby("Region")["Lead Time"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(region_chart)

    with right:
        st.subheader("Sales by Division")

        division_chart = (
            df.groupby("Division")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(division_chart)

    st.subheader("Model Evaluation")

    display_results = results_df.copy()

    display_results["RMSE"] = display_results["RMSE"].round(2)
    display_results["MAE"] = display_results["MAE"].round(2)
    display_results["R2"] = display_results["R2"].round(3)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        f"Selected predictive model based on lowest RMSE: **{best_model}**"
    )


# ============================================================
# TAB 2 — EDA
# ============================================================
with tab2:

    st.header("🔎 Exploratory Data Analysis")

    e1, e2 = st.columns(2)

    with e1:
        st.subheader("Orders by Ship Mode")
        st.bar_chart(df["Ship Mode"].value_counts())

    with e2:
        st.subheader("Orders by Region")
        st.bar_chart(df["Region"].value_counts())

    st.subheader("Average Lead Time by Ship Mode")

    ship_lead = (
        df.groupby("Ship Mode")["Lead Time"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(ship_lead)

    st.subheader("Product Performance")

    product_summary = (
        df.groupby("Product Name")
        .agg(
            Orders=("Order ID", "nunique")
            if "Order ID" in df.columns
            else ("Product Name", "count"),
            Sales=("Sales", "sum"),
            Units=("Units", "sum"),
            Gross_Profit=("Gross Profit", "sum"),
            Avg_Lead_Time=("Lead Time", "mean")
        )
        .sort_values("Sales", ascending=False)
        .reset_index()
    )

    st.dataframe(
        product_summary.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Missing Values")

    missing = (
        df.isna()
        .sum()
        .reset_index()
    )

    missing.columns = ["Column", "Missing Values"]

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 3 — PREDICTION
# ============================================================
with tab3:

    st.header("🤖 Shipping Lead-Time Prediction")

    st.write(
        f"""
        **Product:** {selected_product}  
        **Division:** {division}  
        **Destination Region:** {selected_region}  
        **Ship Mode:** {selected_ship_mode}  
        **Current Factory:** {current_factory}
        """
    )

    X_selected = prediction_frame(
        selected_product,
        division,
        selected_region,
        selected_ship_mode,
        current_factory,
        avg_sales,
        avg_units,
        avg_cost
    )

    prediction_rows = []

    for model_name, model in models.items():

        value = float(
            model.predict(X_selected)[0]
        )

        prediction_rows.append({
            "Model": model_name,
            "Predicted Lead Time (days)": max(0, value)
        })

    prediction_df = pd.DataFrame(prediction_rows)

    st.dataframe(
        prediction_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    best_prediction = float(
        prediction_df.loc[
            prediction_df["Model"] == best_model,
            "Predicted Lead Time (days)"
        ].iloc[0]
    )

    st.metric(
        f"{best_model} Prediction",
        f"{best_prediction:.2f} days"
    )


# ============================================================
# TAB 4 — FACTORY SIMULATION
# ============================================================
with tab4:

    st.header("🏭 Factory Optimization Simulator")

    st.write(
        "Compare the selected product across all available factories."
    )

    simulation = simulate_factories(
        models,
        selected_product,
        division,
        selected_region,
        selected_ship_mode,
        current_factory,
        avg_sales,
        avg_units,
        avg_cost
    )

    st.subheader("Current Assignment")

    st.metric(
        "Current Factory",
        current_factory
    )

    current_row = simulation[
        simulation["Factory"] == current_factory
    ].iloc[0]

    st.metric(
        "Predicted Current Lead Time",
        f"{current_row['Predicted Lead Time']:.2f} days"
    )

    st.divider()

    st.subheader("Factory Comparison")

    simulation_display = simulation.copy()

    for col in [
        "Predicted Lead Time",
        "Lead Time Reduction",
        "Lead Time Reduction %",
        "Risk Score",
        "Historical Gross Profit"
    ]:
        simulation_display[col] = (
            simulation_display[col].round(2)
        )

    st.dataframe(
        simulation_display,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Predicted Lead Time by Factory")

    chart_df = simulation.set_index("Factory")[
        ["Predicted Lead Time"]
    ]

    st.bar_chart(chart_df)


# ============================================================
# TAB 5 — RECOMMENDATIONS
# ============================================================
with tab5:

    st.header("💡 Factory Reallocation Recommendations")

    simulation = simulate_factories(
        models,
        selected_product,
        division,
        selected_region,
        selected_ship_mode,
        current_factory,
        avg_sales,
        avg_units,
        avg_cost
    )

    # Normalize speed score.
    max_time = simulation["Predicted Lead Time"].max()
    min_time = simulation["Predicted Lead Time"].min()

    if max_time == min_time:
        simulation["Speed Score"] = 100
    else:
        simulation["Speed Score"] = (
            (max_time - simulation["Predicted Lead Time"]) /
            (max_time - min_time)
        ) * 100

    # Profit score.
    # The dataset does not contain factory-specific profit.
    # Therefore historical product gross profit is used as a
    # stability indicator rather than claiming an actual factory profit.
    simulation["Profit Score"] = 100

    speed_weight = priority / 100
    profit_weight = 1 - speed_weight

    simulation["Recommendation Score"] = (
        simulation["Speed Score"] * speed_weight
        + simulation["Profit Score"] * profit_weight
    )

    recommendations = (
        simulation
        .sort_values(
            "Recommendation Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    top_n = min(5, len(recommendations))

    st.subheader(f"Top {top_n} Factory Options")

    rec_display = recommendations.head(top_n).copy()

    rec_display["Predicted Lead Time"] = (
        rec_display["Predicted Lead Time"].round(2)
    )

    rec_display["Lead Time Reduction"] = (
        rec_display["Lead Time Reduction"].round(2)
    )

    rec_display["Lead Time Reduction %"] = (
        rec_display["Lead Time Reduction %"].round(2)
    )

    rec_display["Recommendation Score"] = (
        rec_display["Recommendation Score"].round(2)
    )

    st.dataframe(
        rec_display[
            [
                "Factory",
                "Factory Change",
                "Predicted Lead Time",
                "Lead Time Reduction",
                "Lead Time Reduction %",
                "Risk Score",
                "Recommendation Score"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    best_option = recommendations.iloc[0]

    st.success(
        f"""
        Simulation result for **{selected_product}**:

        Current factory: **{current_factory}**

        Lowest-scored simulation option: **{best_option['Factory']}**

        Predicted lead time: **{best_option['Predicted Lead Time']:.2f} days**

        Lead-time change vs current: **{best_option['Lead Time Reduction']:.2f} days**
        """
    )

    st.warning(
        "Important: factory-specific shipping cost/profit is not present in the uploaded CSV. "
        "Therefore the profit component is treated as a stability indicator, not an actual "
        "incremental profit forecast."
    )

    st.subheader("Optimization Logic")

    st.write(
        f"""
        **Speed weight:** {priority}%  
        **Profit-stability weight:** {100 - priority}%

        The recommendation score combines predicted shipping speed and
        profit stability. The speed component is based on predicted lead
        time across factories.
        """
    )


# ============================================================
# FACTORY MAP DATA
# ============================================================
st.divider()

st.header("📍 Factory Coordinates")

factory_df = pd.DataFrame([
    {
        "Factory": name,
        "Latitude": lat,
        "Longitude": lon
    }
    for name, (lat, lon) in FACTORIES.items()
])

st.dataframe(
    factory_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RAW DATA
# ============================================================
with st.expander("📄 View Cleaned Dataset"):

    st.dataframe(
        df.head(500),
        use_container_width=True,
        hide_index=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Cleaned CSV",
        data=csv,
        file_name="nassau_candy_cleaned.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================
st.caption(
    "Nassau Candy Distributor | Factory Reallocation & Shipping Optimization Recommendation System"
)
