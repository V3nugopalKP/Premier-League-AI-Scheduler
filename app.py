import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

st.set_page_config(page_title="Premier League Scheduler (CSP)", layout="wide")

st.markdown("""
<style>
/* Increase all font sizes */
html, body, [class*="css"]  {
    font-size: 20px !important;
}

/* Bigger titles */
h1 { font-size: 42px !important; }
h2 { font-size: 34px !important; }
h3 { font-size: 28px !important; }

/* Sidebar buttons */
.sidebar .stButton>button {
    font-size: 20px !important;
    padding: 12px;
    border-radius: 8px;
}

/* Main buttons */
.stButton>button {
    font-size: 20px !important;
    padding: 14px 20px;
    border-radius: 8px;
}

/* Sliders */
.stSlider label {
    font-size: 20px !important;
}
.stSlider div {
    font-size: 20px !important;
}

/* DataFrame / tables */
.dataframe tbody tr td {
    font-size: 18px !important;
}
.dataframe thead tr th {
    font-size: 20px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------
# Data Loading
# ---------------------
@st.cache_data
def load_data():
    teams = pd.read_csv("teams.csv")
    dists = pd.read_csv("distances.csv")
    slots = pd.read_csv("slots.csv")

    # Normalize datetime strings like "2026-08-15 15:00 BST"
    def parse_dt(s):
        try:
            # remove trailing TZ token; keep naive local time for comparisons
            parts = s.rsplit(" ", 1)
            return datetime.strptime(parts[0], "%Y-%m-%d %H:%M")
        except Exception:
            return None

    slots["dt_uk"] = slots["time_uk"].apply(parse_dt)
    slots["dt_ist"] = slots["time_ist"].apply(parse_dt)
    slots.sort_values(["dt_uk", "parallel_index"], inplace=True, ignore_index=True)
    return teams, dists, slots

teams_df, dist_df, slots_df = load_data()

st.sidebar.title("Premier League Scheduler")

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

def nav_button(label, target):
    if st.sidebar.button(label, use_container_width=True):
        st.session_state["page"] = target

nav_button("🏠 Home", "Home")
nav_button("📋 Teams", "Teams")
nav_button("🗺 Stadium Map", "Stadium Map")
nav_button("⚙ Scheduling Options", "Scheduling Options")
nav_button("📅 Generate Schedule", "Generate Schedule")
nav_button("🔄 Rescheduling", "Rescheduling")
nav_button("⬇ Downloads", "Downloads")
nav_button("ℹ About", "About")

page = st.session_state["page"]


# ---------------------
# Helper: Round-robin fixtures
# ---------------------
def generate_double_round_robin(teams):
    """
    Circle method for even number of teams.
    Returns list of rounds; each round is list of (home, away) for first half.
    Second half mirrors with swapped home/away.
    """
    n = len(teams)
    assert n % 2 == 0, "Number of teams must be even"
    team_list = list(teams)
    rounds = []
    for r in range(n - 1):
        pairings = []
        for i in range(n // 2):
            t1 = team_list[i]
            t2 = team_list[n - 1 - i]
            # alternate home/away across rounds for variety
            pairings.append((t1, t2) if r % 2 == 0 else (t2, t1))
        rounds.append(pairings)
        # rotate
        team_list = [team_list[0]] + [team_list[-1]] + team_list[1:-1]
    rounds2 = [[(b, a) for (a, b) in rnd] for rnd in rounds]
    return rounds + rounds2  # 38 rounds

team_names = teams_df["team"].tolist()
if len(team_names) != 20:
    st.warning(f"Expected 20 teams. Found: {len(team_names)}")

rounds = generate_double_round_robin(team_names)  # 38 rounds, 10 matches each

# ---------------------
# Home Page
# ---------------------
if page == "Home":
    st.title("⚽ Premier League Match Scheduler — AI + CSP System")

    st.markdown("""
Welcome to the **Premier League Fixture Scheduling System**, an AI-assisted tool designed to 
generate a complete season schedule for the Premier League using **Constraint Satisfaction**, 
**real-world calendar data**, and **custom rule controls**.

This system intelligently assigns 380 matches into available time slots while ensuring 
fairness, realism, and adherence to tournament constraints.

---

## ✅ What This System Does
- Generates a **full 38-game season** for all 20 clubs  
- Uses a real-date calendar (**Aug 2026 → May 2027**)  
- Considers **DST (GMT/BST)** and automatically shows **IST equivalent**  
- Applies multiple scheduling constraints to ensure fairness  
- Allows **club-specific rescheduling** (UEFA match week conflicts)  
- Provides a dynamic and interactive **Streamlit UI**

---

## ✅ Key Features
### 📅 Real Calendar Slot System  
- Weekend: **2 parallel matches @ 15:00 UK**  
- Weekdays: **20:00 UK** (Wed & Fri double slots)  
- Handles **off weeks** (FIFA breaks) & **off days** (Dec 25, Jan 1)

### 🔧 Configurable Scheduling Rules  
- Max consecutive **home** matches  
- Max consecutive **away** matches  
- Minimum **rest hours** between matches  
- Weekly match limit per team  
- All rules adjustable in **Scheduling Options**

### 🔄 Intelligent Rescheduling  
- Move a specific club's match to the next feasible slot  
- Ensures other matches remain unaffected  
- Useful for European competitions (UEFA)

### 🌍 Stadium & Distance Data  
- Stadium coordinates  
- Club-to-club distances (used for future optimization)

---

## ✅ Architecture Overview
- **Greedy rule-based scheduler** for fast assignment  
- **Slot feasibility engine** using constraints  
- **Future-ready design** for OR-Tools CP-SAT integration  
- **Streamlit UI** with persistent settings & modular pages  

---

## ✅ Start Scheduling
Click below to begin generating the Premier League season:

### 👉 [Go to Generate Schedule](#)
""")


# ---------------------
# Teams
# ---------------------
if page == "Teams":
    st.title("Teams & Stadiums")
    st.dataframe(teams_df)

# ---------------------
# Stadium Map
# ---------------------
if page == "Stadium Map":
    st.title("Club Location Graph (All Clubs Connected)")
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.scatter(teams_df["lon"], teams_df["lat"])
    for i in range(len(teams_df)):
        xi, yi = teams_df.loc[i, "lon"], teams_df.loc[i, "lat"]
        ax.text(xi + 0.07, yi + 0.07, teams_df.loc[i, "team"], fontsize=7)
        for j in range(i + 1, len(teams_df)):
            xj, yj = teams_df.loc[j, "lon"], teams_df.loc[j, "lat"]
            ax.plot([xi, xj], [yi, yj], lw=0.2, alpha=0.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Premier League Clubs — Complete Graph")
    ax.grid(alpha=0.2)
    st.pyplot(fig)

# ---------------------
# Scheduling Options
# ---------------------
if page == "Scheduling Options":
    st.title("Scheduling Options & Constraints")

    # Load current saved values OR defaults
    if "opts" not in st.session_state:
        st.session_state["opts"] = {
            "max_cons_home": 2,
            "max_cons_away": 2,
            "min_rest_hours": 48,
            "enforce_week_limit": True,
        }

    saved = st.session_state["opts"]

    # Show sliders initialized with saved values
    max_cons_home = st.slider("Max consecutive HOME matches", 2, 4, saved["max_cons_home"])
    max_cons_away = st.slider("Max consecutive AWAY matches", 2, 4, saved["max_cons_away"])
    min_rest_hours = st.slider("Minimum rest gap (hours)", 24, 96, saved["min_rest_hours"], step=12)
    enforce_week_limit = st.checkbox("One match per team per calendar week", value=saved["enforce_week_limit"])

    # Save button (prevents auto-reset)
    if st.button("Save Settings"):
        st.session_state["opts"] = {
            "max_cons_home": max_cons_home,
            "max_cons_away": max_cons_away,
            "min_rest_hours": min_rest_hours,
            "enforce_week_limit": enforce_week_limit,
        }
        st.success("✅ Settings saved successfully! They will remain until you change them again.")
    else:
        st.info("Adjust values and press **Save Settings** to apply changes.")


# ---------------------
# Scheduler (Greedy)
# ---------------------
def build_schedule(rounds, slots_df, opts):
    """
    Greedy assignment of each round's fixtures into available slots.
    - One match per slot
    - Team not double-booked
    - Min rest hours
    - Max consecutive home/away (uses sliders)
    - One match per team per calendar week (if enabled)
    """
    team_last_dt = {t: None for t in team_names}
    team_week_used = defaultdict(set)
    slot_used = np.zeros(len(slots_df), dtype=bool)

    # Track streaks: last venue ('H'/'A') and current count
    team_streak = {t: {"last": None, "count": 0} for t in team_names}

    maxH = opts.get("max_cons_home", 2)
    maxA = opts.get("max_cons_away", 2)

    assignments = []
    slot_idx = 0

    def ok_rest(t, dt):
        last = team_last_dt[t]
        return True if last is None else (dt - last).total_seconds() >= opts["min_rest_hours"] * 3600

    def ok_week(t, w):
        return True if not opts.get("enforce_week_limit", True) else (w not in team_week_used[t])

    def ok_consec(team, venue_char):  # venue_char in {'H','A'}
        last = team_streak[team]["last"]
        cnt = team_streak[team]["count"]
        if last != venue_char:
            return True  # streak resets
        # would become cnt+1
        if venue_char == 'H':
            return (cnt + 1) <= maxH
        else:
            return (cnt + 1) <= maxA

    def apply_streak(team, venue_char):
        last = team_streak[team]["last"]
        if last == venue_char:
            team_streak[team]["count"] += 1
        else:
            team_streak[team] = {"last": venue_char, "count": 1}

    # Iterate rounds
    for rnd_idx, matches in enumerate(rounds, start=1):
        for (home, away) in matches:
            placed = False
            j = slot_idx
            while j < len(slots_df):
                if slot_used[j]:
                    j += 1
                    continue
                row = slots_df.iloc[j]
                w = int(row["week"])
                dt = row["dt_uk"]

                if not ok_week(home, w) or not ok_week(away, w):
                    j += 1; continue
                if not ok_rest(home, dt) or not ok_rest(away, dt):
                    j += 1; continue
                if not ok_consec(home, 'H') or not ok_consec(away, 'A'):
                    j += 1; continue

                # assign
                slot_used[j] = True
                team_last_dt[home] = dt
                team_last_dt[away] = dt
                team_week_used[home].add(w)
                team_week_used[away].add(w)
                apply_streak(home, 'H')
                apply_streak(away, 'A')

                assignments.append({
                    "round": rnd_idx,
                    "home": home,
                    "away": away,
                    "slot_id": int(row["slot_id"]),
                    "date": row["date"],
                    "day_name": row["day_name"],
                    "week": w,
                    "time_uk": row["time_uk"],
                    "time_ist": row["time_ist"],
                    "label": row["label"],
                })
                placed = True
                if j == slot_idx:
                    slot_idx += 1
                break

            if not placed:
                assignments.append({
                    "round": rnd_idx,
                    "home": home,
                    "away": away,
                    "slot_id": None,
                    "date": None,
                    "day_name": None,
                    "week": None,
                    "time_uk": None,
                    "time_ist": None,
                    "label": "UNASSIGNED",
                })

    return pd.DataFrame(assignments)

# ---------------------
# Generate Schedule
# ---------------------
if page == "Generate Schedule":
    st.title("Generate Full Season Schedule")
    opts = st.session_state.get(
        "opts",
        {"max_cons_home": 2, "max_cons_away": 2, "min_rest_hours": 48, "enforce_week_limit": True},
    )
    if st.button("Run Scheduler"):
        with st.spinner("Building fixtures and assigning slots..."):
            sched_df = build_schedule(rounds, slots_df, opts)
            st.session_state["schedule"] = sched_df
        st.success("Scheduling complete.")

    if "schedule" in st.session_state:
        df = st.session_state["schedule"].copy()
        st.write(
            f"Total matches: {len(df)} | Assigned: {df['slot_id'].notna().sum()} | Unassigned: {df['slot_id'].isna().sum()}"
        )
        st.dataframe(df.head(50))
        week = st.number_input(
            "Jump to week",
            min_value=int(slots_df["week"].min()),
            max_value=int(slots_df["week"].max()),
            value=int(slots_df["week"].min()),
        )
        week_matches = df[df["week"] == week].sort_values("time_uk")
        st.subheader(f"Week {week} matches")
        st.dataframe(week_matches)

# ---------------------
# Rescheduling (UEFA)
# ---------------------
def reschedule_team_week(schedule_df, team, bad_week, slots_df, min_rest_hours=48):
    """
    Move the selected team's match from bad_week to the nearest future slot
    where both teams are free (week limit + rest), without touching others.
    """
    df = schedule_df.copy()
    mask = ((df["home"] == team) | (df["away"] == team)) & (df["week"] == bad_week)
    if not mask.any():
        return df, "No match found for the selected team in that week."

    idx = df[mask].index[0]
    row = df.loc[idx]
    opp = row["away"] if row["home"] == team else row["home"]

    # Clear current assignment
    df.loc[idx, ["slot_id", "date", "day_name", "week", "time_uk", "time_ist", "label"]] = [None] * 7

    # Build availability from remaining matches
    team_weeks = defaultdict(set)
    team_last_dt = defaultdict(lambda: None)
    for _, r in df.dropna(subset=["slot_id"]).iterrows():
        wk = int(r["week"])
        dt = datetime.strptime(r["time_uk"].rsplit(" ", 1)[0], "%Y-%m-%d %H:%M")
        team_weeks[r["home"]].add(wk)
        team_weeks[r["away"]].add(wk)
        for t in (r["home"], r["away"]):
            if team_last_dt[t] is None or dt > team_last_dt[t]:
                team_last_dt[t] = dt

    def ok_rest(t, dt):
        last = team_last_dt[t]
        return True if last is None else (dt - last).total_seconds() >= min_rest_hours * 3600

    # search forward
    candidates = slots_df[slots_df["week"] >= bad_week].sort_values(["dt_uk", "parallel_index"])
    assigned = False
    msg = None
    for _, s in candidates.iterrows():
        w = int(s["week"])
        dt = s["dt_uk"]
        if w in team_weeks[team] or w in team_weeks[opp]:
            continue
        if not ok_rest(team, dt) or not ok_rest(opp, dt):
            continue
        # ensure slot not already used elsewhere in df
        if ((df["slot_id"] == s["slot_id"]) & (df["slot_id"].notna())).any():
            continue
        # assign
        df.loc[idx, ["slot_id", "date", "day_name", "week", "time_uk", "time_ist", "label"]] = [
            int(s["slot_id"]), s["date"], s["day_name"], w, s["time_uk"], s["time_ist"], s["label"]
        ]
        assigned = True
        msg = f"Rescheduled {team} vs {opp} to week {w} — {s['label']}"
        break

    if not assigned:
        msg = "Could not find a feasible future slot without disturbing other fixtures."
    return df, msg

if page == "Rescheduling":
    st.title("UEFA Rescheduling (Club-specific)")
    if "schedule" not in st.session_state:
        st.info("Run the scheduler first on the **Generate Schedule** page.")
    else:
        team_sel = st.selectbox("Select team", team_names)
        week_sel = st.number_input(
            "UEFA week (move this club's league game to future week)",
            min_value=int(slots_df["week"].min()),
            max_value=int(slots_df["week"].max()),
            value=int(slots_df["week"].min()),
        )
        if st.button("Reschedule"):
            new_df, msg = reschedule_team_week(
                st.session_state["schedule"], team_sel, int(week_sel), slots_df,
                min_rest_hours=st.session_state.get("opts", {}).get("min_rest_hours", 48)
            )
            st.session_state["schedule"] = new_df
            st.success(msg)
        st.dataframe(st.session_state["schedule"].head(50))

# ---------------------
# Downloads
# ---------------------
if page == "Downloads":
    st.title("Download Schedules & Data")
    st.markdown("Download the base datasets used by the app:")
    st.download_button("teams.csv", data=teams_df.to_csv(index=False), file_name="teams.csv", mime="text/csv")
    st.download_button("distances.csv", data=dist_df.to_csv(index=False), file_name="distances.csv", mime="text/csv")
    st.download_button("slots.csv", data=slots_df.to_csv(index=False), file_name="slots.csv", mime="text/csv")
    if "schedule" in st.session_state:
        st.download_button(
            "final_schedule.csv",
            data=st.session_state["schedule"].to_csv(index=False),
            file_name="final_schedule.csv",
            mime="text/csv",
        )
    else:
        st.info("Run scheduler to enable schedule download.")

# ---------------------
# About
# ---------------------
if page == "About":
    st.title("About This Project")
    st.markdown(
        """
**Premier League Scheduler** built for AI/CSP demonstration:
- Constraint Satisfaction & Heuristics
- Real calendar with UK/IST times and OFF weeks
- Interactive rescheduling for UEFA conflicts
        """
    )
