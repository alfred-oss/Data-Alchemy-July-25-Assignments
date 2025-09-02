import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

st.set_page_config(page_title="Real Estate Explorer", page_icon="🏠", layout="wide")
st.title("🏠 Real Estate Explorer")

DATA_PATH = r"dsc.csv"

@st.cache_data
def load_csv(path, nrows=None):
    return pd.read_csv(path, nrows=nrows, encoding="utf-8-sig", low_memory=False)

if not os.path.exists(DATA_PATH):
    st.error(f"File not found: {DATA_PATH}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Options")
    sample_rows = st.number_input("Rows to read (0 = all)", 0, 1_000_000, 0, step=50_000)
    page = st.radio("📂 Navigation", ["Home", "EDA", "Maps", "Insights"], index=0)

df = load_csv(DATA_PATH, nrows=None if sample_rows == 0 else int(sample_rows))

def numeric_cols(d): return [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
def has_cols(d, cols): return all(c in d.columns for c in cols)

def page_home():
    st.subheader("🧩 Problem statement")
    st.write(
        "**Goal:** Identify the best place to buy a house using `price_per_sqft` "
        "and contextual features (e.g., `state`, `crime_rate`, `flood_risk`)."
    )

    st.divider()

    # Why this matters
    st.subheader("Why this matters")
    st.markdown("""
- Housing is typically the largest purchase most people make.
- Prices vary widely with fundamentals (bed/bath, size, lot) and with **context** (crime, income, flood risk).
- Using **price_per_sqft** lets us compare locations more fairly than raw prices.
- Adding state-level context (crime rate groups, median income, flood risk) helps set expectations, even if it’s an indirect indicator.
    """)

    st.divider()

        # Data Source & Limitations
    st.subheader("Data Source & Limitations")
    st.markdown("""
**Source:** USA Real Estate Dataset from `kaggle.com`.

**Coverage:** Multiple US states. 

**What was missing:**
- `price_per_sqft`, `crime_rate`, `median_income`, `flood_risk` — had to search and add manually for more context and better decision making.
- `zip_code` / `street` contained missing values; some `house_size`, `acre_lot`, `bed`, and `bath` had zeros and extreme outliers.
    """)

    st.divider()

    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")

    st.write("**Head**")
    st.dataframe(df.head(6), use_container_width=True)

    st.write("**Describe (numeric)**")
    num = numeric_cols(df)
    if num:
        st.dataframe(df[num].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns found.")
    
def page_eda():
    st.subheader("🔎 EDA")
    num = numeric_cols(df)
    if not num:
        st.info("No numeric columns.")
        return

    st.markdown("**Histogram**")
    col = st.selectbox("Column", num)
    bins = st.slider("Bins", 10, 100, 40)
    if HAS_PLOTLY:
        st.plotly_chart(px.histogram(df, x=col, nbins=bins), use_container_width=True)
    else:
        st.bar_chart(df[col].dropna())

    st.markdown("**Scatter plot**")
    c1, c2 = st.columns(2)
    xcol = c1.selectbox("X", num, key="x")
    ycol = c2.selectbox("Y", num, key="y")
    if HAS_PLOTLY:
        st.plotly_chart(px.scatter(df, x=xcol, y=ycol, opacity=0.5), use_container_width=True)
    else:
        st.line_chart(df[[xcol, ycol]].dropna().sort_values(xcol))

    st.markdown("**Correlation Heatmap (numeric)**")

    # compute correlation
    corr = df[num].corr(numeric_only=True)

    # plot heatmap with seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True)

    # show in Streamlit
    st.pyplot(fig)

STATE_ABBREV = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT',
    'Delaware':'DE','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
    'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI',
    'Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH',
    'Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
    'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV',
    'Wisconsin':'WI','Wyoming':'WY','District of Columbia':'DC'
}
def page_maps():
    st.subheader("🗺️ Maps")
    # Which metric to map?
    metric = st.selectbox(
        "Choose a metric to map:",
        ["price_per_sqft", "crime_rate", "median_income", "flood_risk"],
        index=0
    )

    # Pick a nice color scale per metric
    scales = {
        "price_per_sqft": "Viridis",
        "crime_rate": "Reds",
        "median_income": "Blues",
        "flood_risk": "Plasma",
    }
    labels = {
        "price_per_sqft": "Avg. $/sqft",
        "crime_rate": "Avg. crime rate",
        "median_income": "Avg. median income",
        "flood_risk": "Avg. flood risk",
    }

    # Aggregate by state
    g = df.groupby("state", as_index=False)[metric].mean()
    g["code"] = g["state"].map(STATE_ABBREV)
    g = g.dropna(subset=["code"])  # keep only valid US states in our map

    # Build the choropleth
    fig = px.choropleth(
        g,
        locations="code",
        locationmode="USA-states",
        color=metric,
        scope="usa",
        color_continuous_scale=scales[metric],
        labels={metric: labels[metric]},
        title=f"Average {labels[metric]} by State"
    )
    st.plotly_chart(fig, use_container_width=True)

 # ----- Regional Comparisons -------------------------------------------------
    st.markdown("---")
    st.subheader("🏁 Regional Comparisons")

    # pick states to compare
    all_states = sorted(df["state"].dropna().unique().tolist())
    compare_states = st.multiselect(
        "Pick 2–5 states to compare (e.g., New York, Georgia):",
        options=all_states,
        default=["New York", "Georgia"],
        max_selections=5,
    )

    # pick metrics to compare
    metrics = ["price_per_sqft", "crime_rate", "median_income", "flood_risk"]
    pick_metrics = st.multiselect(
        "Pick metrics to compare:",
        options=metrics,
        default=["price_per_sqft", "median_income", "crime_rate"],
    )

    norm = st.checkbox("Normalize selected metrics to [0, 1] for fair visual comparison")

    if len(compare_states) >= 2 and len(pick_metrics) >= 1:
        # aggregate means per state
        agg = df.groupby("state", as_index=False)[pick_metrics].mean()

        # restrict to chosen states
        view = agg[agg["state"].isin(compare_states)].copy()

        # optional normalization across chosen states
        if norm:
            # min-max per metric (only over chosen states so bars remain intuitive for the comparison)
            for m in pick_metrics:
                mmin, mmax = view[m].min(), view[m].max()
                if mmax > mmin:
                    view[m] = (view[m] - mmin) / (mmax - mmin)
                else:
                    # avoid divide-by-zero if all values equal
                    view[m] = 0.5

        # long format for grouped bars
        long = view.melt(id_vars="state", value_vars=pick_metrics,
                         var_name="metric", value_name="value")

        fig_cmp = px.bar(
            long,
            x="state", y="value", color="metric",
            barmode="group",
            labels={"value": "value (normalized)" if norm else "value"},
            title="State comparison" + (" (normalized)" if norm else ""),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # small numeric table under the chart
        with st.expander("Show numeric table"):
            st.dataframe(view.set_index("state"))
    else:
        st.info("Pick at least two states and one metric to see the comparison.")
            
def page_insights():
    st.subheader("💡 Insights & Next steps")

    # --- Prep: state-level aggregates
    cols_needed = ["state", "price_per_sqft", "crime_rate", "median_income", "flood_risk"]
    d = df[cols_needed].dropna().copy()

    state_agg = (
        d.groupby("state", as_index=False)
         .agg(price_per_sqft=("price_per_sqft", "mean"),
              crime_rate=("crime_rate", "mean"),
              median_income=("median_income", "mean"),
              flood_risk=("flood_risk", "mean"),
              listings=("price_per_sqft", "size"))
         .sort_values("price_per_sqft")
         .reset_index(drop=True)
    )

    # --- 1) Where is it cheaper / more expensive?
    st.markdown("### 1) Where is it cheaper / more expensive?")
    k = st.slider("How many states to show (each list)", 3, 10, 5, key="insights_topk")

    cheap = state_agg.nsmallest(k, "price_per_sqft")[["state", "price_per_sqft", "listings"]]
    expensive = state_agg.nlargest(k, "price_per_sqft")[["state", "price_per_sqft", "listings"]]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cheapest (by average price_per_sqft)**")
        st.dataframe(cheap.set_index("state"), use_container_width=True)
    with c2:
        st.markdown("**Most expensive (by average price_per_sqft)**")
        st.dataframe(expensive.set_index("state"), use_container_width=True)

    # optional bars
    fig_bar = px.bar(state_agg.sort_values("price_per_sqft").head(15),
                     x="state", y="price_per_sqft",
                     title="Lowest 15 price_per_sqft (avg by state)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- 2) What factors influence price_per_sqft?
    st.subheader("2) What influences price_per_sqft?")
    st.markdown("""
    **Key signals from correlations:**

    - **Price** ↗️ **strong** (+): higher listing price tends to go with higher `price_per_sqft` (~0.76).
    - **House size** ↘️ **weak–moderate** (−): bigger homes are cheaper per sqft (about −0.17).
    - **Beds/Baths**: very weak links (≈0 to −0.11). Count itself isn’t a strong driver per sqft.
    - **Median income** ↗️ **moderate** (+): wealthier areas tend to have higher `price_per_sqft` (~0.34).
    - **Crime rate** ↗️ **weak–moderate** (+): pricier-per-sqft places can still have higher recorded crime (~0.23).
    - **Flood risk** ↗️ **weak** (+): small positive association (~0.22).

    **Reading:** `price_per_sqft` is driven mainly by location and market demand (overall price levels and income). Home size or the number of beds/baths matter much less once you normalize by square footage.
    """)

    # --- 3) Trade-offs (low price, high risk)
    st.markdown("### 3) Trade-offs")
    st.caption("States that are **cheap** (bottom quartile of price_per_sqft) but **riskier** (top quartile of crime_rate or flood_risk).")
    q_price_low = state_agg["price_per_sqft"].quantile(0.25)
    q_crime_high = state_agg["crime_rate"].quantile(0.75)
    q_flood_high = state_agg["flood_risk"].quantile(0.75)

    tradeoffs = state_agg.query(
        "(price_per_sqft <= @q_price_low) and (crime_rate >= @q_crime_high or flood_risk >= @q_flood_high)"
    )[["state", "price_per_sqft", "crime_rate", "flood_risk"]].sort_values("price_per_sqft")

    if tradeoffs.empty:
        st.info("No strong trade-off states under the chosen quartile rule.")
    else:
        st.dataframe(tradeoffs.set_index("state"), use_container_width=True)

    # --- 4) “Best to buy” vs “Expensive” (balanced shortlists)
    st.markdown("### 4) Best states to buy vs expensive states")
    st.caption("Balanced score: low price_per_sqft, higher median_income, lower crime_rate, lower flood_risk (min–max normalized).")

    # normalize 0–1 (higher is better after sign flips)
    norm = state_agg.copy()
    def mm(x): 
        x = x.astype(float)
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    norm["p_inv"] = 1 - mm(norm["price_per_sqft"])
    norm["income"] = mm(norm["median_income"])
    norm["crime_inv"] = 1 - mm(norm["crime_rate"])
    norm["flood_inv"] = 1 - mm(norm["flood_risk"])
    norm["score"] = (
        0.40*norm["p_inv"] + 
        0.25*norm["income"] + 
        0.20*norm["crime_inv"] + 
        0.15*norm["flood_inv"]
    )
    shortlist = norm.sort_values("score", ascending=False).head(k)[
        ["state", "score", "price_per_sqft", "median_income", "crime_rate", "flood_risk"]
    ]
    pricey = state_agg.nlargest(k, "price_per_sqft")[["state", "price_per_sqft", "median_income", "crime_rate", "flood_risk"]]

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Best states to buy (multi-factor score)**")
        st.dataframe(shortlist.set_index("state"), use_container_width=True)
    with cc2:
        st.markdown("**Expensive states (by price_per_sqft)**")
        st.dataframe(pricey.set_index("state"), use_container_width=True)

    # --- 5) Future Enhancements
    st.markdown("### 🔮 Future Enhancements")
    st.markdown(
        """
- **Add richer features**
  - School quality (ratings, distance to top schools)  
  - Job market (unemployment, new jobs, industry mix)  
  - Transportation & accessibility (transit score, commute time)  
  - Amenities (parks, hospitals, grocery density), environmental risks beyond floods  
  - Property specifics (year built, HOA fees, parking/garage, renovation status)
        """
    )

    # --- Summary bullets (fast talking points)
    st.markdown("### Quick summary:")
    st.markdown(
        f"""
- **Cheapest now:** {', '.join(cheap['state'].head(3))} &nbsp;&nbsp;|&nbsp;&nbsp; **Most expensive:** {', '.join(expensive['state'].head(3))}
- **Drivers (corr):** higher **median_income** → higher price/sqft; higher **crime_rate**/**flood_risk** → tends to lower appeal.
- **Trade-offs:** Some low-price states show higher **crime/flood risk**—screen area-level safety before deciding.
- **Best shortlist (multi-factor):** {', '.join(shortlist['state'].head(5))}
        """
    )

if page == "Home":
    page_home()
elif page == "EDA":
    page_eda()
elif page == "Maps":
    page_maps()
else:
    page_insights()
    