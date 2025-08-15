import pandas as pd
from shapely.geometry import Point, Polygon
from plotly.colors import qualitative
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
import re
import io

# =======================
# --- Layout
# =======================
st.set_page_config(layout="wide", page_title="Pennsylvania Orphan Wells")

# Global font tweaks (bolder + bigger) + bigger tab button text
st.markdown(
    """
    <style>
      html, body, [class*="block-container"] { font-size: 17px !important; }
      p, span, label, div, .stText, .stCaption { font-weight: 600 !important; }
      h1, h2, h3 { font-weight: 800 !important; }
      .stSelectbox label, .stMultiSelect label, .stRadio label { font-weight: 700 !important; }
      .stButton>button, .stDownloadButton>button { font-weight: 700 !important; }

      /* --- Bigger tab labels --- */
      /* Streamlit tabs render as buttons; cover common selectors across versions */
      div[role="tablist"] button[role="tab"] { 
        font-size: 1.05rem !important; 
        font-weight: 800 !important; 
        padding: 8px 16px !important;
      }
      .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        padding: 8px 16px !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ======= Title ABOVE tabs + author line =======
st.title("🛢️ Pennsylvania Orphan Wells")
st.markdown(
    "<p style='margin-top:-8px; font-size:15px; font-weight:700;'>Authored by SN, AKD, RD</p>",
    unsafe_allow_html=True
)

# =======================
# --- Data (cached)
# =======================
@st.cache_data
def load_data(path: str):
    df = pd.read_excel(path)

    def clean_binary_column(series: pd.Series):
        s = series.astype("string").str.lower().str.strip()
        s = s.replace({
            "false": "no", "true": "yes",
            "na": "unknown", "n/a": "unknown", "": "unknown",
            "none found": "no", "none": "no",
            "0": "no", "1": "yes"
        })
        s = s.fillna("unknown")
        s = s.map({"yes": "Yes", "no": "No", "unknown": "Unknown"}).fillna("Unknown")
        return s

    bin_cols = [
        "Well within 200 feet of occupied building or water supply well",
        "Well within 100 feet of stream",
        "Oil/brine in occupied structure believed to be associated with well?",
        "Soil gas within 200 feet of structure believed to be associated with well?",
        "Liquids (oil/brine) to stream or wetland from well?",
        "Gas present outside surface casing/present in stream or liquid flow to surface",
        "Water Contamination",
    ]
    for c in bin_cols:
        if c in df.columns:
            df[c] = clean_binary_column(df[c])

    def clean_gas_value(val):
        if pd.isna(val): return "Unknown"
        s = re.sub(r"\s+", " ", str(val).strip().lower())
        if "methane" in s or s in {"1","yes","y"}: return "Yes"
        if s in {"0","no","n"}: return "No"
        if s in {"na","n/a","none",""}: return "Unknown"
        return "Unknown"

    if "Leaking Gas" in df.columns:
        df["Leaking Gas"] = df["Leaking Gas"].apply(clean_gas_value)

    if "Water Contamination" in df.columns:
        df["Water Contamination"] = (
            df["Water Contamination"].astype("string").str.strip().str.title()
            .where(df["Water Contamination"].isin(["Yes","No","Unknown"]), "Unknown")
        )

    spec_col = "Gas present outside surface casing/present in stream or liquid flow to surface"
    if spec_col in df.columns:
        df[spec_col] = (
            df[spec_col].astype("string").str.lower().str.strip()
            .replace({"0":"no","1":"yes"})
            .replace({"false":"no","true":"yes","na":"unknown","n/a":"unknown","":"unknown"})
        )
        df[spec_col] = df[spec_col].map({"yes":"Yes","no":"No","unknown":"Unknown"}).fillna("Unknown")

    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("Data must include 'Latitude' and 'Longitude'.")
    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df

df = load_data("PA data set 2K records (1).xlsx")

# =======================
# --- Hover fields
# =======================
HOVER_FIELDS = [
    "GlobalID","Inspection ID","Inspector","Date","API No.","Well Type",
    "Producing Formation","Well Status","County","Latitude","Longitude",
    "Contaminated Soil Suspected","Contaminated Soil Confirmed",
    "Surface Water Impact Suspected","Surface Water Impact Confirmed",
    "Ground Water Impact Suspected","Ground Water Impact Confirmed",
    "Existing Contamination Total",
    "Gas in occupied structure with similar isotopic signature or believed to be associated with well?",
    "Oil/brine in occupied structure believed to be associated with well?",
    "Soil gas within 200 feet of structure believed to be associated with well?",
    "Gas present outside surface casing/present in stream or liquid flow to surface",
    "Measurable annular flow of gas","Leaking Gas","Well Depth","Well Operator",
    "Describe the condition of the well, including any oil, gas, brine discharge.",
]

# =======================
# --- Shared controls
# =======================
preferred = [
    "Leaking Gas",
    "Well within 200 feet of occupied building or water supply well",
    "Well within 100 feet of stream",
    "Oil/brine in occupied structure believed to be associated with well?",
    "Soil gas within 200 feet of structure believed to be associated with well?",
    "Liquids (oil/brine) to stream or wetland from well?",
    "Gas present outside surface casing/present in stream or liquid flow to surface",
    "Water Contamination",
]
available = [c for c in preferred if c in df.columns]
if not available:
    available = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=False) <= 50
    ][:12]
if not available:
    st.error("No suitable columns found to color by.")
    st.stop()

# =======================
# --- Colors & helpers
# =======================
default_cat_map = {"No": "#1205c2", "Yes": "black", "Unknown": "#9e9e9e"}
leaking_map = {"No": "#1205c2", "Yes": "black", "Unknown": "#9e9e9e"}
compact_map_cols = {
    "Well within 200 feet of occupied building or water supply well": default_cat_map,
    "Well within 100 feet of stream": default_cat_map,
    "Oil/brine in occupied structure believed to be associated with well?": default_cat_map,
    "Soil gas within 200 feet of structure believed to be associated with well?": default_cat_map,
    "Liquids (oil/brine) to stream or wetland from well?": default_cat_map,
    "Gas present outside surface casing/present in stream or liquid flow to surface": default_cat_map,
    "Water Contamination": default_cat_map,
    "Leaking Gas": leaking_map,
}
BIG = qualitative.Alphabet + qualitative.Dark24 + qualitative.Light24 + qualitative.Set3 + qualitative.Plotly

RISK_SCALE_DARKPURP = [
    "#ffffff", "#f5f0ff", "#e5d7ff", "#c9afff",
    "#a67aff", "#8a4dff", "#722bd7", "#5a19b3",
    "#3e0f82", "#23084d"
]

def bw_map_style():
    return "carto-positron"

def pick_well_name_column(df_like: pd.DataFrame):
    for c in ["Well name", "Well Name", "Well", "API No."]:
        if c in df_like.columns:
            return c
    return None

def pick_well_status_column(df_like: pd.DataFrame):
    for c in ["Well Status", "Well Statues", "Status"]:
        if c in df_like.columns:
            return c
    return None

# =====================================================
# ===============  TABS (Point first)  =================
# =====================================================
tab_map_table, tab_prio = st.tabs(["🗺️ Point Map + 📃 Data Table", "📌 Prioritization Map"])

# =======================
# --- TAB 1: Point Map + Data Table
# =======================
with tab_map_table:
    st.subheader("🗺️ Point Map")

    valid_ll = df["Latitude"].notna() & df["Longitude"].notna()
    dview = df.loc[valid_ll].copy()
    hover_data = {c: True for c in dict.fromkeys(HOVER_FIELDS) if c in dview.columns}
    wn = pick_well_name_column(dview)

    category_options = ["Default value: All Wells"] + available
    color_by = st.selectbox("Categories:", category_options, index=0, key="color_by_point")
    simple_mode = (color_by == "Default value: All Wells")

    if simple_mode:
        center_lat = dview["Latitude"].mean() if not dview.empty else 40.9
        center_lon = dview["Longitude"].mean() if not dview.empty else -77.7

        # Filled GREEN markers
        fig = px.scatter_mapbox(
            dview,
            lat="Latitude", lon="Longitude",
            hover_name=wn if wn else None,
            hover_data=hover_data,
            zoom=6, height=700, center=dict(lat=center_lat, lon=center_lon),
            mapbox_style=bw_map_style(),
        )
        fig.update_traces(marker=dict(size=10, opacity=0.95, color="#2ecc71", symbol="circle"),
                          showlegend=False)
        fig.update_layout(margin=dict(r=0, t=40, l=0, b=0),
                          title={"text": "All Wells", "font": {"size": 18}},
                          font=dict(size=16))
        st.plotly_chart(fig, use_container_width=True)

    else:
        if color_by in compact_map_cols:
            dview[color_by] = dview[color_by].astype("string").fillna("Unknown").str.strip().str.title()
            category_orders = {color_by: list(compact_map_cols[color_by].keys())}
            color_map = compact_map_cols[color_by]
        else:
            dview[color_by] = dview[color_by].astype("string").fillna("Unknown").str.strip()
            order = list(dview[color_by].value_counts(dropna=False).index)
            category_orders = {color_by: order}
            color_map = {c: BIG[i % len(BIG)] for i, c in enumerate(order)}

        fig = px.scatter_mapbox(
            dview,
            lat="Latitude", lon="Longitude",
            color=color_by,
            hover_name=wn if wn else None,
            color_discrete_map=color_map, category_orders=category_orders,
            hover_data=hover_data,
            zoom=6, height=700,
            center=dict(lat=dview["Latitude"].mean() if not dview.empty else 40.9,
                        lon=dview["Longitude"].mean() if not dview.empty else -77.7),
            mapbox_style=bw_map_style(),
        )
        fig.update_traces(marker=dict(size=9, opacity=0.9))
        # Bigger legend text (map info)
        fig.update_layout(
            margin=dict(r=0, t=40, l=0, b=0),
            title={"text": str(color_by), "font": {"size": 18}},
            legend=dict(font=dict(size=18)),
            legend_title=dict(font=dict(size=18)),
            font=dict(size=16)
        )
        st.plotly_chart(fig, use_container_width=True)

        # YES-only table under the map
        dtable = dview.copy()
        dtable[color_by] = dtable[color_by].astype("string").fillna("Unknown").str.strip().str.title()
        d_yes = dtable[dtable[color_by] == "Yes"].copy()

        well_name_col   = pick_well_name_column(d_yes)
        well_status_col = pick_well_status_column(d_yes)

        desired_cols = [c for c in [
            well_name_col,
            "County" if "County" in d_yes.columns else None,
            "Inspector" if "Inspector" in d_yes.columns else None,
            "Well Type" if "Well Type" in d_yes.columns else None,
            well_status_col,
        ] if c is not None]

        st.subheader(f"Contaminated (Yes) — {color_by}")
        if d_yes.empty or not desired_cols:
            st.caption("No 'Yes' records for the selected category (or required columns not present).")
        else:
            st.caption(f"Showing {len(d_yes)} records with “Yes”.")
            st.dataframe(d_yes[desired_cols], use_container_width=True)
            csvp = d_yes[desired_cols].to_csv(index=False).encode("utf-8")
            fnamep = "yes_records_" + re.sub(r"[^A-Za-z0-9_-]+", "_", str(color_by).lower()) + ".csv"
            st.download_button(
                label="📥 Download 'Yes' table as CSV",
                data=csvp,
                file_name=fnamep,
                mime="text/csv",
                key="yes_only_csv_point_map"
            )

    # ---------- Data Table ----------
    st.markdown("---")
    st.subheader("📃 Data Table")

    category_options_table = ["Default value: All Wells"] + available
    color_by_table = st.selectbox("Categories (for table):", category_options_table, index=0, key="color_by_table")
    simple_mode_table = (color_by_table == "Default value: All Wells")

    if simple_mode_table:
        st.info("Choose a category above to view/filter the data table.")
    else:
        if "data_value_choice" not in st.session_state:
            st.session_state.data_value_choice = "Yes"

        value_choice = st.radio(
            f"Show rows where **{color_by_table}** is:",
            options=["Yes", "Unknown", "No", "All"],
            horizontal=True,
            key="data_value_choice"
        )

        dtab = df.copy()
        dtab[color_by_table] = dtab[color_by_table].astype("string").fillna("Unknown").str.strip().str.title()
        if value_choice != "All":
            dtab = dtab[dtab[color_by_table] == value_choice]

        well_name_col_t   = pick_well_name_column(dtab)
        well_status_col_t = pick_well_status_column(dtab)

        desired_cols_t = [c for c in [
            well_name_col_t,
            "County" if "County" in dtab.columns else None,
            "Inspector" if "Inspector" in dtab.columns else None,
            "Well Type" if "Well Type" in dtab.columns else None,
            well_status_col_t,
        ] if c is not None]

        if not desired_cols_t:
            st.caption("None of the desired columns were found. Showing a preview of available data instead.")
            preview_df = dtab.head(100)
            st.dataframe(preview_df, use_container_width=True)

            csv = preview_df.to_csv(index=False)
            st.download_button(
                label="📥 Download preview as CSV",
                data=csv.encode("utf-8"),
                file_name="contamination_preview.csv",
                mime="text/csv"
            )
        else:
            st.caption(f"Showing {len(dtab)} row(s).")
            shown_df = dtab[desired_cols_t]
            st.dataframe(shown_df, use_container_width=True)

            csv = shown_df.to_csv(index=False)
            st.download_button(
                label="📥 Download table as CSV",
                data=csv.encode("utf-8"),
                file_name="contamination_data.csv",
                mime="text/csv"
            )

# =======================
# --- TAB 2: Prioritization Map
# =======================
with tab_prio:
    st.subheader("📌 Prioritization Map")

    cols_ctrl = st.columns([1, 2])
    with cols_ctrl[0]:
        # renamed control
        n_top = st.selectbox("Limit to top by priority score:", [10, 25, 50, 100, "All"], index=0, key="prio_top_n_main")
    with cols_ctrl[1]:
        selectable_features = [c for c in preferred if c in df.columns]
        selected_features = st.multiselect(
            "Select features to prioritize:",
            options=selectable_features,
            default=selectable_features,
            key="prio_features_main"
        )

    if not selected_features:
        st.warning("Please select at least one feature to prioritize.")
    else:
        score_col = "Contamination Risk Score"
        df[score_col] = df[selected_features].apply(lambda row: int(sum(row == "Yes")), axis=1)

        valid_ll = df["Latitude"].notna() & df["Longitude"].notna()
        df_all = df.loc[valid_ll].copy()
        df_score = df_all.sort_values(by=[score_col], ascending=False)
        if n_top != "All":
            df_score = df_score.head(int(n_top))

        center_lat = df_score["Latitude"].mean() if not df_score.empty else 40.9
        center_lon = df_score["Longitude"].mean() if not df_score.empty else -77.7
        max_score = max(1, len(selected_features))

        hover_cols1 = [score_col] + selected_features
        for extra in ("County", "Well Type"):
            if extra in df_score.columns:
                hover_cols1.append(extra)
        wn1 = pick_well_name_column(df_score)
        if wn1:
            hover_cols1.insert(0, wn1)

        fig1 = px.scatter_mapbox(
            df_score,
            lat="Latitude", lon="Longitude",
            color=score_col,
            size=score_col, size_max=22,
            hover_name=wn1 if wn1 else None,
            hover_data={c: True for c in hover_cols1},
            zoom=6, height=640,
            center=dict(lat=center_lat, lon=center_lon),
            mapbox_style=bw_map_style(),
            color_continuous_scale=RISK_SCALE_DARKPURP,
            range_color=[0, max_score],
        )
        fig1.update_traces(marker=dict(opacity=0.98, symbol="circle"))
        fig1.update_layout(
            margin=dict(r=0, t=30, l=0, b=0),
            title={"text": "Contamination Risk Score (higher = more concerning)", "font": {"size": 18}},
            coloraxis_colorbar=dict(
                title=dict(text="Contamination Risk Score", font=dict(size=18)),
                tickfont=dict(size=16)
            ),
            font=dict(size=16)
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Top Priority Sites table
        st.subheader("📋 Top Priority Sites")
        st.caption(f"Showing top {len(df_score)} site(s) by {score_col}.")
        well_name_col   = pick_well_name_column(df_score)
        well_status_col = pick_well_status_column(df_score)

        display_cols = [c for c in [well_name_col, "County", "Well Type", well_status_col] if c is not None]
        display_cols = [score_col] + display_cols + selected_features

        shown_df = df_score[display_cols]
        st.dataframe(shown_df, use_container_width=True)

        csv_bytes = shown_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download prioritization table as CSV",
            data=csv_bytes,
            file_name="priority_wells.csv",
            mime="text/csv",
            key="prio_csv_btn_main"
        )
