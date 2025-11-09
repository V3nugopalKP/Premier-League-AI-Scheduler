# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from ortools.sat.python import cp_model

# -------------------------------
# Streamlit setup + CSS
# -------------------------------
st.set_page_config(page_title="Premier League Scheduler (CSP Basic)", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 20px !important; }
h1 { font-size: 42px !important; }
h2 { font-size: 34px !important; }
h3 { font-size: 28px !important; }
.sidebar .stButton>button { font-size: 20px !important; padding: 12px; border-radius: 8px; }
.stButton>button { font-size: 20px !important; padding: 14px 20px; border-radius: 8px; }
.dataframe tbody tr td { font-size: 18px !important; }
.dataframe thead tr th { font-size: 20px !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data():
    teams = pd.read_csv("teams.csv")
    slots = pd.read_csv("slots.csv")
    # distances loaded but NOT used (kept for completeness)
    try:
        dists = pd.read_csv("distances.csv")
    except Exception:
        # create an empty placeholder if not found
        dists = pd.DataFrame(columns=["team_a","team_b","distance_km"])

    def parse_dt(s):
        # removes trailing timezone token e.g. "2026-08-15 15:00 BST"
        try:
            parts = s.rsplit(" ", 1)
            return datetime.strptime(parts[0], "%Y-%m-%d %H:%M")
        except Exception:
            return None

    slots["dt_uk"] = slots["time_uk"].apply(parse_dt)
    slots["dt_ist"] = slots["time_ist"].apply(parse_dt)
    # convenience columns
    slots["is_weekend"] = slots["day_name"].isin(["Saturday", "Sunday"])
    if "parallel_index" in slots.columns:
        slots.sort_values(["dt_uk", "parallel_index"], inplace=True, ignore_index=True)
    else:
        slots.sort_values(["dt_uk"], inplace=True, ignore_index=True)

    return teams, dists, slots

teams_df, dist_df, slots_df = load_data()
team_names = teams_df["team"].tolist()
weeks_sorted = sorted(slots_df["week"].astype(int).unique())

# -------------------------------
# Sidebar navigation (buttons)
# -------------------------------
st.sidebar.title("Premier League Scheduler")

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

def nav(label, key):
    if st.sidebar.button(label, use_container_width=True):
        st.session_state["page"] = key

nav("🏠 Home", "Home")
nav("📋 Teams", "Teams")
nav("🗺 Stadium Map", "Map")
nav("⚙ Rules & Options", "Options")
nav("📅 Generate (Greedy)", "Greedy")
nav("🧠 Generate (CSP Basic)", "CSP")
nav("⬇ Downloads", "Download")
nav("ℹ About", "About")

page = st.session_state["page"]

# -------------------------------
# Double Round-Robin fixtures
# -------------------------------
def generate_double_round_robin(teams):
    """
    Circle method for even number of teams.
    Returns list of 38 rounds; each round is list of (home, away).
    """
    n = len(teams)
    assert n % 2 == 0, "Number of teams must be even"
    T = list(teams)
    rounds = []
    for r in range(n - 1):
        pairings = []
        for i in range(n // 2):
            t1 = T[i]; t2 = T[n - 1 - i]
            pairings.append((t1, t2) if r % 2 == 0 else (t2, t1))
        rounds.append(pairings)
        # rotate while keeping first fixed
        T = [T[0]] + [T[-1]] + T[1:-1]
    second_half = [[(b, a) for (a, b) in rnd] for rnd in rounds]
    return rounds + second_half  # 38 rounds

rounds = generate_double_round_robin(team_names)

# -------------------------------
# Options (rules) state
# -------------------------------
if "opts" not in st.session_state:
    st.session_state["opts"] = {
        "max_home": 2,
        "max_away": 2,
        "min_rest": 48,     # used only by greedy; CSP basic ignores rest for speed
        "weekly": True,
    }

# -------------------------------
# Greedy Scheduler (fast baseline)
# -------------------------------
def greedy_schedule(rounds, slots, opts):
    """
    Greedy forward assignment (no backtracking).
    Enforces:
      - At most one match per slot
      - 1 match per team per week (if enabled)
      - Min rest (hours) between matches
      - Max consecutive home/away
    """
    team_last = {t: None for t in team_names}
    team_week = defaultdict(set)
    slot_used = np.zeros(len(slots), bool)

    maxH = int(opts["max_home"])
    maxA = int(opts["max_away"])
    min_rest = int(opts["min_rest"])
    weekly = bool(opts["weekly"])

    streak = {t: ("-", 0) for t in team_names}

    def ok_rest(t, dt):
        if team_last[t] is None: return True
        return (dt - team_last[t]).total_seconds() >= min_rest * 3600

    def ok_week(t, w):
        return (not weekly) or (w not in team_week[t])

    def ok_streak(t, val):
        last, count = streak[t]
        if last != val: return True
        if val == "H": return (count + 1) <= maxH
        if val == "A": return (count + 1) <= maxA

    def push_streak(t, val):
        last, count = streak[t]
        if last == val:
            streak[t] = (val, count + 1)
        else:
            streak[t] = (val, 1)

    assignments = []
    s_index = 0

    for rnd, matches in enumerate(rounds, start=1):
        for home, away in matches:
            placed = False
            j = s_index
            while j < len(slots):
                if slot_used[j]:
                    j += 1; continue
                row = slots.iloc[j]
                w = int(row["week"]); dt = row["dt_uk"]

                if not ok_week(home, w): j += 1; continue
                if not ok_week(away, w): j += 1; continue
                if not ok_rest(home, dt): j += 1; continue
                if not ok_rest(away, dt): j += 1; continue
                if not ok_streak(home, "H"): j += 1; continue
                if not ok_streak(away, "A"): j += 1; continue

                # assign
                slot_used[j] = True
                team_last[home] = dt; team_last[away] = dt
                team_week[home].add(w); team_week[away].add(w)
                push_streak(home, "H"); push_streak(away, "A")

                assignments.append({
                    "round": rnd, "home": home, "away": away,
                    "slot_id": row["slot_id"], "week": row["week"],
                    "date": row["date"], "time_uk": row["time_uk"],
                    "time_ist": row["time_ist"], "label": row["label"],
                })
                placed = True
                if j == s_index: s_index += 1
                break

            if not placed:
                assignments.append({
                    "round": rnd, "home": home, "away": away,
                    "slot_id": None, "week": None, "date": None,
                    "time_uk": None, "time_ist": None, "label": "UNASSIGNED",
                })

    return pd.DataFrame(assignments)

# -------------------------------
# BASIC CSP (fast, no distance)
# -------------------------------
def run_basic_csp_with_homeaway(
    rounds,
    slots_df,
    opts,
    week_window=1,
    time_limit=10,
    uefa_blocks=None,
    workers=8
):
    """
    UPDATED VERSION (FAST + SAFE):
    ✅ Proper week-window pruning (index-based)
    ✅ Correct home/away streak logic
    ✅ Weekly limit handled safely
    ✅ Boolean indicator fix
    ✅ Applies worker threads
    ✅ No freezing
    """

    model = cp_model.CpModel()

    # Build match list
    matches = []  # (home, away, round)
    for r, rnd in enumerate(rounds):
        for h, a in rnd:
            matches.append((h, a, r))

    M = len(matches)
    S = len(slots_df)

    slot_week = slots_df["week"].astype(int).tolist()
    all_weeks = sorted(set(slot_week))

    # week positions for correct window calculation (IMPORTANT)
    week_pos = {w:i for i, w in enumerate(all_weeks)}

    # Map round → approximate calendar week
    round_to_week = []
    for r in range(38):
        idx = int((len(all_weeks) - 1) * r / 37.0)
        round_to_week.append(all_weeks[idx])

    # ---------------------------------------
    # ✅ Eligibility pruning (FAST + SAFE)
    # ---------------------------------------
    eligible = [[] for _ in range(M)]
    uefa_blocks = uefa_blocks or {}

    for m, (h, a, r) in enumerate(matches):
        pref_week = round_to_week[r]
        pref_pos = week_pos[pref_week]

        block_home = uefa_blocks.get(h, set())
        block_away = uefa_blocks.get(a, set())

        for s in range(S):
            w = slot_week[s]

            # skip blocked uefa weeks
            if w in block_home or w in block_away:
                continue

            # use week positions (fix)
            if abs(week_pos[w] - pref_pos) <= week_window:
                eligible[m].append(s)

        # emergency fallback — pick 5 nearest slots
        if not eligible[m]:
            d = [(abs(week_pos[slot_week[s]] - pref_pos), s) for s in range(S)]
            d.sort()
            eligible[m] = [s for _, s in d[:5]]  # reduced from 10 → faster

    # ---------------------------------------
    # ✅ Decision variables
    # ---------------------------------------
    x = {}
    for m in range(M):
        for s in eligible[m]:
            x[(m, s)] = model.NewBoolVar(f"x_{m}_{s}")

    # 1) each match exactly one slot
    for m in range(M):
        model.Add(sum(x[(m, s)] for s in eligible[m]) == 1)

    # 2) at most one match per slot
    for s in range(S):
        ms_here = [m for m in range(M) if s in eligible[m]]
        if ms_here:
            model.Add(sum(x[(m, s)] for m in ms_here) <= 1)

    # ---------------------------------------
    # ✅ Group matches by team
    # ---------------------------------------
    teams = sorted(set([h for (h, _, _) in matches] +
                       [a for (_, a, _) in matches]))

    home_of = {t: [] for t in teams}
    away_of = {t: [] for t in teams}

    for m, (h, a, r) in enumerate(matches):
        home_of[h].append(m)
        away_of[a].append(m)

    weekly = bool(opts.get("weekly", True))
    max_home = int(opts.get("max_home", 2))
    max_away = int(opts.get("max_away", 2))

    # ---------------------------------------
    # ✅ Weekly & streak indicators (FIXED BOOLEAN LOGIC)
    # ---------------------------------------
    H = {}
    A = {}

    for t in teams:
        for w in all_weeks:

            vH = model.NewBoolVar(f"H_{t}_{w}")
            vA = model.NewBoolVar(f"A_{t}_{w}")

            H[(t, w)] = vH
            A[(t, w)] = vA

            # home indicator
            home_vars = []
            for m in home_of[t]:
                for s in eligible[m]:
                    if slot_week[s] == w:
                        home_vars.append(x[(m, s)])

            if home_vars:
                for v in home_vars:
                    model.Add(vH >= v)       # at least one home match → vH=1
                model.Add(vH <= sum(home_vars))  # if no home match → sum=0 → vH=0
            else:
                model.Add(vH == 0)

            # away indicator
            away_vars = []
            for m in away_of[t]:
                for s in eligible[m]:
                    if slot_week[s] == w:
                        away_vars.append(x[(m, s)])

            if away_vars:
                for v in away_vars:
                    model.Add(vA >= v)
                model.Add(vA <= sum(away_vars))
            else:
                model.Add(vA == 0)

            # weekly limit (only if enabled)
            if weekly:
                model.Add(vH + vA <= 1)

        # home streak sliding window
        for i in range(len(all_weeks) - max_home):
            window = all_weeks[i:i + max_home + 1]
            model.Add(sum(H[(t, w)] for w in window) <= max_home)

        # away streak sliding window
        for i in range(len(all_weeks) - max_away):
            window = all_weeks[i:i + max_away + 1]
            model.Add(sum(A[(t, w)] for w in window) <= max_away)

        # hard UCL blocks
        for w in uefa_blocks.get(t, set()):
            model.Add(H[(t, w)] == 0)
            model.Add(A[(t, w)] == 0)

    # ---------------------------------------
    # ✅ Solve (time-limited)
    # ---------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.max_memory_in_mb = 2000  # optional safety

    status = solver.Solve(model)

    # ---------------------------------------
    # ✅ Build output
    # ---------------------------------------
    assigned = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for m in range(M):
            for s in eligible[m]:
                if solver.Value(x[(m, s)]) == 1:
                    h, a, r = matches[m]
                    row = slots_df.iloc[s]
                    assigned.append({
                        "round": r + 1,
                        "home": h,
                        "away": a,
                        "slot_id": row["slot_id"],
                        "week": row["week"],
                        "date": row["date"],
                        "time_uk": row["time_uk"],
                        "time_ist": row["time_ist"],
                        "label": row["label"]
                    })

    return pd.DataFrame(assigned), solver.StatusName(status)

# -------------------------------
# PAGES
# -------------------------------
if page == "Home":
    st.title("⚽ Premier League Scheduler — Greedy + CSP (Basic)")
    st.markdown("""
Welcome! This app builds a **full Premier League season** with two engines:

- **Greedy (Fast)**: quick baseline; may leave some matches unassigned  
- **CSP (Basic)**: time-limited CP-SAT; **home/away streaks + weekly limit**; no distance; fast & reliable  

Use **Rules & Options** to set:
- Max consecutive home / away  
- Weekly limit (on/off)  

Then try **Generate (Greedy)** or **Generate (CSP Basic)**.
""")

if page == "Teams":
    st.title("Teams & Stadiums")
    st.dataframe(teams_df)

if page == "Map":
    st.title("Stadium Map (All Clubs Connected)")
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.scatter(teams_df["lon"], teams_df["lat"])
    for i in range(len(teams_df)):
        xi, yi = teams_df.loc[i, "lon"], teams_df.loc[i, "lat"]
        ax.text(xi + 0.06, yi + 0.06, teams_df.loc[i, "team"], fontsize=9)
        for j in range(i + 1, len(teams_df)):
            xj, yj = teams_df.loc[j, "lon"], teams_df.loc[j, "lat"]
            ax.plot([xi, xj], [yi, yj], lw=0.2, alpha=0.3)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(alpha=0.3)
    st.pyplot(fig)

if page == "Options":
    st.title("Scheduling Options & Constraints")
    cfg = st.session_state["opts"]

    max_home = st.slider("Max consecutive HOME", 2, 4, cfg["max_home"])
    max_away = st.slider("Max consecutive AWAY", 2, 4, cfg["max_away"])
    min_rest = st.slider("Minimum rest (hours, Greedy only)", 24, 96, cfg["min_rest"], step=12)
    weekly = st.checkbox("One match per team per week (recommended)", value=cfg["weekly"])

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
    st.info(f"""
**Active Rules**
- Max HOME in a row: **{st.session_state['opts']['max_home']}**
- Max AWAY in a row: **{st.session_state['opts']['max_away']}**
- Weekly limit: **{st.session_state['opts']['weekly']}**
- Min rest (Greedy only): **{st.session_state['opts']['min_rest']}h**
""")

if page == "Greedy":
    st.title("Generate Schedule — Greedy (Fast)")
    if st.button("Run Greedy Scheduler", use_container_width=True):
        with st.spinner("Assigning matches..."):
            df = greedy_schedule(rounds, slots_df, st.session_state["opts"])
            st.session_state["greedy"] = df
        st.success("Greedy scheduling complete.")
    if "greedy" in st.session_state:
        df = st.session_state["greedy"]
        st.write(f"Total matches: 380 | Assigned: {df['slot_id'].notna().sum()} | Unassigned: {df['slot_id'].isna().sum()}")
        st.dataframe(df)

if page == "CSP":
    st.title("Generate Schedule — CSP (Basic, Fast, No Distance)")
    with st.expander("Solver Settings", expanded=True):
        colA, colB, colC = st.columns(3)
        with colA:
            wk_window = st.number_input("Round→Week window (± weeks)", 1, 5, 1)
        with colB:
            time_limit = st.number_input("Time limit (seconds)", 5, 120, 10)
        with colC:
            workers = st.number_input("Workers (threads, info only)", 1, 16, 8, help="Fixed to 8 in code for safety on shared machines.")

    with st.expander("UEFA Blocks (optional)", expanded=False):
        t_sel = st.selectbox("Team to block", team_names, index=0)
        w_sel = st.selectbox("Week to block", weeks_sorted, index=0)
        if st.button("➕ Add Block (team cannot play that week)"):
            st.session_state.setdefault("uefa_blocks", set()).add((t_sel, int(w_sel)))
        st.write("Current blocks:", sorted(list(st.session_state.get("uefa_blocks", set()))))

    # convert blocks to dict: {team: set(weeks)}
    block_dict = {}
    for (t, w) in st.session_state.get("uefa_blocks", set()):
        block_dict.setdefault(t, set()).add(int(w))

    if st.button("Run CSP (Basic)"):
        with st.spinner("Solving CSP..."):
            df_csp, status = run_basic_csp_with_homeaway(
                rounds,
                slots_df,
                opts=st.session_state["opts"],
                week_window=int(wk_window),
                time_limit=int(time_limit),
                uefa_blocks=block_dict
            )
            st.session_state["csp"] = df_csp
            st.session_state["csp_status"] = status
        st.success(f"CSP finished with status: {status}")

    if "csp" in st.session_state:
        df = st.session_state["csp"]
        status = st.session_state.get("csp_status", "UNKNOWN")
        st.write(f"Assigned: **{len(df)} / 380** | Status: **{status}**")
        st.dataframe(df)

if page == "Download":
    st.title("Downloads")
    st.download_button("Download teams.csv", teams_df.to_csv(index=False), "teams.csv")
    st.download_button("Download slots.csv", slots_df.to_csv(index=False), "slots.csv")
    if "greedy" in st.session_state:
        st.download_button("Greedy schedule.csv", st.session_state["greedy"].to_csv(index=False), "greedy_schedule.csv")
    if "csp" in st.session_state:
        st.download_button("CSP schedule.csv", st.session_state["csp"].to_csv(index=False), "csp_schedule.csv")

if page == "About":
    st.title("About This Project")
    st.markdown("""
This app demonstrates **sports scheduling** with:
- **Greedy heuristic** (fast baseline)
- **Basic CSP** (Constraint Satisfaction via OR-Tools CP-SAT)
- **Home/Away streak controls** (≤2 default)
- **Weekly limit** (one match per team per calendar week)
- Real calendar slots in UK time (IST shown too)

The **Basic CSP** is time-limited and pruned by a small week window, so it **won’t run forever**.
Distance/travel optimization is intentionally **not included** in this version to keep it fast and simple.
""")
