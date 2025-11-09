import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data
from round_robin import generate_double_round_robin
from greedy_scheduler import greedy_schedule
from csp_scheduler import run_basic_csp_with_homeaway


# -------------------------------------------------------
# ✅ Global Page Setup + Custom Theme
# -------------------------------------------------------
st.set_page_config(
    page_title="Premier League Scheduler",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Custom CSS (Modern + High Contrast)
st.markdown("""
<style>

html, body {
    font-size: 22px !important;
    background-color: white !important;
    color: black !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #000000 !important;
}

[data-testid="stSidebar"] * {
    color: white !important;
    font-size: 20px !important;
}

/* Headings */
h1 {
    font-size: 46px !important;
    font-weight: 800 !important;
}
h2 {
    font-size: 36px !important;
    font-weight: 700 !important;
}
h3 {
    font-size: 30px !important;
}

/* Buttons */
.stButton>button {
    background-color: #000000;
    color: white;
    padding: 12px 24px;
    border-radius: 10px;
    font-size: 22px;
}
.stButton>button:hover {
    background-color: #444444;
}

/* Dataframe */
.dataframe tbody tr td {
    font-size: 18px !important;
    color: black !important;
}
.dataframe thead tr th {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: black !important;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# ✅ Load CSV Data
# -------------------------------------------------------
teams_df, dist_df, slots_df = load_data()
team_names = teams_df["team"].tolist()
weeks_sorted = sorted(slots_df["week"].astype(int).unique())

# -------------------------------------------------------
# ✅ Generate Round-Robin Fixtures
# -------------------------------------------------------
rounds = generate_double_round_robin(team_names)

# -------------------------------------------------------
# ✅ Sidebar Navigation
# -------------------------------------------------------
st.sidebar.title("⚽ Navigation")

PAGES = [
    "Home",
    "Teams",
    "Map",
    "Options",
    "Greedy",
    "CSP",
    "Club Fixtures",
    "Download",
    "About"
]

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

def nav(label):
    if st.sidebar.button(label, use_container_width=True):
        st.session_state["page"] = label

for p in PAGES:
    nav(p)

page = st.session_state["page"]

# -------------------------------------------------------
# ✅ Store Options
# -------------------------------------------------------
if "opts" not in st.session_state:
    st.session_state["opts"] = {
        "max_home": 2,
        "max_away": 2,
        "min_rest": 48,
        "weekly": True,
    }


# -------------------------------------------------------
# ✅ PAGES
# -------------------------------------------------------

# ---------------- Home -----------------
if page == "Home":
    st.title("🏆 Premier League Match Scheduler — CSP + Greedy")
    st.markdown("""
Welcome to the **Premier League Scheduling System**, powered by:
- ✅ **Greedy Heuristics** (Very Fast)
- ✅ **CSP (Constraint Satisfaction Problem)** using CP-SAT  
- ✅ **Home/Away Streak Rules**
- ✅ **Weekly Match Constraints**
- ✅ **UEFA Block Adjustments**
- ✅ **Real-world Calendar Slots (UK & IST)**

Use the left navigation to explore scheduling features step-by-step.
""")

# ---------------- Teams -----------------
elif page == "Teams":
    st.title("📋 Premier League Teams")
    st.dataframe(teams_df)

# ---------------- Map -----------------
elif page == "Map":
    st.title("🗺 Stadium Map (Fully Connected)")
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.scatter(teams_df["lon"], teams_df["lat"])
    for i in range(len(teams_df)):
        xi, yi = teams_df.loc[i, "lon"], teams_df.loc[i, "lat"]
        ax.text(xi + 0.06, yi + 0.06, teams_df.loc[i, "team"], fontsize=10)
        for j in range(i + 1, len(teams_df)):
            xj, yj = teams_df.loc[j, "lon"], teams_df.loc[j, "lat"]
            ax.plot([xi, xj], [yi, yj], lw=0.2, alpha=0.3)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ---------------- Options -----------------
if page == "Options":
    st.title("Scheduling Options & Constraints")
    cfg = st.session_state["opts"]

    # Minimum Rest Hours
    min_rest = st.slider(
        "Minimum Rest Gap Between Two Matches (hours)",
        24, 96, cfg.get("min_rest", 48), step=12,
        help="48 hours = 2 days gap, 72 hours = 3 days gap"
    )

    # Home/Away Streaks
    max_home = st.slider("Max consecutive HOME", 2, 4, cfg["max_home"])
    max_away = st.slider("Max consecutive AWAY", 2, 4, cfg["max_away"])

    # Weekly Rule Toggle
    weekly = st.checkbox(
        "Restrict teams to ONE match per week (Optional)",
        value=cfg["weekly"],
        help="If OFF, teams may play more than once per week if rest gap is satisfied"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Save Settings", use_container_width=True):
            st.session_state["opts"] = {
                "max_home": int(max_home),
                "max_away": int(max_away),
                "min_rest": int(min_rest),
                "weekly": bool(weekly),
            }
            st.success("Settings saved.")

    with col2:
        if st.button("⚡ Apply Recommended Rules", use_container_width=True):
            st.session_state["opts"] = {
                "max_home": 2,
                "max_away": 2,
                "min_rest": 48,
                "weekly": True,
            }
            st.success("Recommended rules applied.")


# ---------------- Greedy Scheduler -----------------
elif page == "Greedy":
    st.title("⚡ Generate Schedule (Greedy Algorithm)")

    if st.button("Run Greedy Scheduler"):
        with st.spinner("Scheduling..."):
            df = pd.DataFrame(greedy_schedule(rounds, slots_df, team_names, st.session_state["opts"]))
            st.session_state["greedy"] = df
        st.success("✅ Greedy scheduling done!")

    if "greedy" in st.session_state:
        df = st.session_state["greedy"]
        st.write(f"✅ Assigned: {df['slot_id'].notna().sum()} / 380")
        st.dataframe(df)

# ---------------- CSP Scheduler -----------------
elif page == "CSP":
    st.title("🧠 Generate Schedule (CSP - CP-SAT)")

    colA, colB, colC = st.columns(3)
    with colA:
        wk_window = st.number_input("Round→Week Window (±)", 1, 3, 1)
    with colB:
        time_limit = st.number_input("Time Limit (s)", 5, 60, 10)
    with colC:
        workers = st.number_input("Threads", 1, 16, 8)

    with st.expander("UEFA Blocks"):
        t_sel = st.selectbox("Team to block", team_names)
        w_sel = st.selectbox("Week to block", weeks_sorted)
        if st.button("➕ Add block"):
            st.session_state.setdefault("uefa_blocks", set()).add((t_sel, int(w_sel)))
        st.write("Current:", st.session_state.get("uefa_blocks", set()))

    blocks = {}
    for (t, w) in st.session_state.get("uefa_blocks", set()):
        blocks.setdefault(t, set()).add(w)

    if st.button("Run CSP Scheduler"):
        with st.spinner("Solving CSP..."):
            df_csp, status = run_basic_csp_with_homeaway(
                rounds,
                slots_df,
                opts=st.session_state["opts"],
                week_window=int(wk_window),
                time_limit=int(time_limit),
                uefa_blocks=blocks,
                workers=int(workers)
            )
            st.session_state["csp"] = df_csp
            st.session_state["csp_status"] = status
        st.success(f"✅ CSP finished — Status: {status}")

    if "csp" in st.session_state:
        st.dataframe(st.session_state["csp"])

# ---------------- Club Fixtures Viewer -----------------
elif page == "Club Fixtures":
    st.title("🔎 Club-Specific Fixtures")

    # FIX: Safe schedule selection
    schedule = None
    if "csp" in st.session_state and isinstance(st.session_state.get("csp"), pd.DataFrame) and len(st.session_state["csp"]) > 0:
        schedule = st.session_state["csp"]
        st.info("✅ Showing fixtures from **CSP Scheduler**")
    elif "greedy" in st.session_state and isinstance(st.session_state.get("greedy"), pd.DataFrame) and len(st.session_state["greedy"]) > 0:
        schedule = st.session_state["greedy"]
        st.info("✅ Showing fixtures from **Greedy Scheduler**")

    # No schedule found
    if schedule is None:
        st.warning("⚠ Please run Greedy or CSP scheduler first.")
    else:
        team = st.selectbox("Select a Team", team_names)

        # Extract matches involving selected team
        df = schedule.copy()
        df["type"] = df.apply(
            lambda r: "HOME" if r["home"] == team else ("AWAY" if r["away"] == team else ""),
            axis=1
        )
        df = df[df["type"] != ""]

        # Opponent identification
        df["opponent"] = df.apply(
            lambda r: r["away"] if r["home"] == team else r["home"],
            axis=1
        )

        # Sort safely by time if available
        try:
            df["sort_dt"] = pd.to_datetime(
                df["time_uk"].str.replace("BST", "").str.replace("GMT", "").str.strip(),
                errors="coerce"
            )
            df = df.sort_values("sort_dt")
        except:
            df = df.sort_values(["week", "round"])

        # Display fixtures
        st.subheader(f"📅 Fixtures for **{team}**")
        st.dataframe(df[["round", "week", "type", "opponent", "date", "time_uk", "time_ist", "label"]])

        # Summary
        st.subheader("📊 Summary")
        st.write(f"🏠 Home Matches: { (df['type'] == 'HOME').sum() }")
        st.write(f"🚌 Away Matches: { (df['type'] == 'AWAY').sum() }")


# ---------------- Download -----------------
elif page == "Download":
    st.title("⬇ Download Data")

    st.download_button("teams.csv", teams_df.to_csv(index=False), "teams.csv")
    st.download_button("slots.csv", slots_df.to_csv(index=False), "slots.csv")

    if "greedy" in st.session_state:
        st.download_button("greedy_schedule.csv", st.session_state["greedy"].to_csv(index=False))

    if "csp" in st.session_state:
        st.download_button("csp_schedule.csv", st.session_state["csp"].to_csv(index=False))

# ---------------- About -----------------
elif page == "About":
    st.title("ℹ About This Project")
    st.markdown("""
✅ **Premier League Scheduler using AI + CSP**  
✅ Greedy + CP-SAT + Heuristic Search  
✅ Weekly constraints, home/away streaks  
✅ UEFA rescheduling support  
✅ UI built with Streamlit  
""")
