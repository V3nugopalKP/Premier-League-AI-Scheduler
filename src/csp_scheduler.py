# csp_scheduler.py

import pandas as pd
from collections import defaultdict
from ortools.sat.python import cp_model


def run_basic_csp_with_homeaway(
    rounds,
    slots_df: pd.DataFrame,
    opts: dict,
    week_window: int = 1,
    time_limit: int = 10,
    uefa_blocks: dict | None = None,   # {team: set(week_labels)}
    workers: int = 8,
):
    """
    FINAL CSP SCHEDULER (robust + fast)

    Inputs
    ------
    rounds       : list[list[(home, away)]]  # 38 rounds from round_robin.generate_double_round_robin
    slots_df     : DataFrame with at least columns:
                   ['slot_id','week','date','time_uk','time_ist','label']
                   - date: 'YYYY-MM-DD'
                   - time_uk: like '15:00 BST' or any string containing 'HH:MM'
    opts         : {
                      'min_rest': int hours (e.g., 48 or 72),
                      'weekly': bool,
                      'max_home': int (default 2),
                      'max_away': int (default 2),
                   }
    week_window  : int, allowed ± distance in chronological week space (speed control)
    time_limit   : CP-SAT time limit (seconds)
    uefa_blocks  : {team: set(week_label_int)} weeks a team cannot play
    workers      : solver threads

    Returns
    -------
    (df, status_name)
      df columns: ['round','home','away','slot_id','week','date','time_uk','time_ist','label']
      Sorted strictly by calendar datetime ascending.
    """

    # ------------------------------------------------------------
    # SAFE DATETIME PARSING (build dt from date + HH:MM in time_uk)
    # ------------------------------------------------------------
    slots_df = slots_df.copy()

    def extract_hhmm(x):
        """Return 'HH:MM' found in a string, else None."""
        if not isinstance(x, str):
            return None
        for part in x.split():
            if ":" in part and len(part) <= 5:
                return part
        return None

    slots_df["clean_time"] = slots_df["time_uk"].apply(extract_hhmm)
    # If time is missing, default to midnight to keep parsing valid
    slots_df["clean_time"] = slots_df["clean_time"].fillna("00:00")

    # Build datetime; ignore timezone token
    slots_df["dt"] = pd.to_datetime(
        slots_df["date"].astype(str) + " " + slots_df["clean_time"].astype(str),
        format="%Y-%m-%d %H:%M",
        errors="coerce"
    )

    # Fallback: try generic parse (in case of unusual formats)
    bad = slots_df["dt"].isna()
    if bad.any():
        slots_df.loc[bad, "dt"] = pd.to_datetime(
            slots_df.loc[bad, "date"].astype(str) + " " + slots_df.loc[bad, "clean_time"].astype(str),
            errors="coerce"
        )

    # Final guard
    if slots_df["dt"].isna().any():
        bad_rows = slots_df[slots_df["dt"].isna()][["slot_id", "date", "time_uk"]]
        raise ValueError(
            "Datetime parse failed for some rows in slots_df. Offending rows:\n"
            + bad_rows.to_string(index=False)
        )

    # Handy lists
    slot_dt_list = slots_df["dt"].tolist()
    slot_week_label = slots_df["week"].astype(int).tolist()
    slot_date_only = slots_df["dt"].dt.date.tolist()

    # ------------------------------------------------------------
    # Derive chronological order of week labels from real dates
    # ------------------------------------------------------------
    tmp = slots_df[["week", "dt"]].dropna().copy()
    tmp["week"] = tmp["week"].astype(int)
    tmp.sort_values("dt", inplace=True)

    ordered_weeks = []
    seen = set()
    for wk in tmp["week"]:
        if wk not in seen:
            ordered_weeks.append(wk)
            seen.add(wk)

    all_weeks = ordered_weeks[:]  # list of week labels in true date order
    week_pos = {w: i for i, w in enumerate(all_weeks)}

    # ------------------------------------------------------------
    # Spread 38 rounds across chronological weeks
    # ------------------------------------------------------------
    round_to_week = []
    for r in range(38):
        idx = int((len(all_weeks) - 1) * r / 37) if len(all_weeks) > 1 else 0
        round_to_week.append(all_weeks[idx])

    # ------------------------------------------------------------
    # Build match list
    # ------------------------------------------------------------
    matches: list[tuple[str, str, int]] = []
    for r_idx, rnd in enumerate(rounds):
        for h, a in rnd:
            matches.append((h, a, r_idx))

    M = len(matches)
    S = len(slots_df)

    # ------------------------------------------------------------
    # Domain pruning (eligible slots per match)
    # ------------------------------------------------------------
    uefa_blocks = uefa_blocks or {}
    eligible: list[list[int]] = [[] for _ in range(M)]

    for m, (home, away, r) in enumerate(matches):
        pref_week = round_to_week[r]
        pref_pos = week_pos[pref_week]

        blocked_home = uefa_blocks.get(home, set())
        blocked_away = uefa_blocks.get(away, set())

        for s in range(S):
            wk = slot_week_label[s]
            if (wk in blocked_home) or (wk in blocked_away):
                continue
            if abs(week_pos[wk] - pref_pos) <= int(week_window):
                eligible[m].append(s)

        # Safety fallback: take 5 nearest weeks by position
        if not eligible[m]:
            nearest = sorted(
                ((abs(week_pos[slot_week_label[s]] - pref_pos), s) for s in range(S))
            )
            eligible[m] = [s for _, s in nearest[:5]]

    # ------------------------------------------------------------
    # CP-SAT model
    # ------------------------------------------------------------
    model = cp_model.CpModel()
    x: dict[tuple[int, int], cp_model.IntVar] = {}

    # Decision variables
    for m in range(M):
        for s in eligible[m]:
            x[(m, s)] = model.NewBoolVar(f"x_{m}_{s}")

    # 1) Each match in exactly one slot
    for m in range(M):
        model.Add(sum(x[(m, s)] for s in eligible[m]) == 1)

    # 2) At most one match per slot
    for s in range(S):
        ms_here = [m for m in range(M) if s in eligible[m]]
        if ms_here:
            model.Add(sum(x[(m, s)] for m in ms_here) <= 1)

    # ------------------------------------------------------------
    # Group matches by team
    # ------------------------------------------------------------
    teams = sorted({h for (h, _, _) in matches} | {a for (_, a, _) in matches})
    home_of = {t: [] for t in teams}
    away_of = {t: [] for t in teams}
    for m, (h, a, r) in enumerate(matches):
        home_of[h].append(m)
        away_of[a].append(m)

    weekly_limit = bool(opts.get("weekly", True))
    max_home = int(opts.get("max_home", 2))
    max_away = int(opts.get("max_away", 2))
    min_rest_hours = int(opts.get("min_rest", 48))  # mandatory rest-gap

    # ------------------------------------------------------------
    # Team-level constraints
    # ------------------------------------------------------------
    H, A = {}, {}

    for t in teams:
        # 3) One match per DATE for each team
        date_groups = defaultdict(list)
        for m in home_of[t] + away_of[t]:
            for s in eligible[m]:
                date_groups[slot_date_only[s]].append(x[(m, s)])
        for d, vars_on_day in date_groups.items():
            model.Add(sum(vars_on_day) <= 1)

        # 4) Minimum REST GAP between any two matches of same team
        t_matches = home_of[t] + away_of[t]
        for i in range(len(t_matches)):
            m1 = t_matches[i]
            for j in range(i + 1, len(t_matches)):
                m2 = t_matches[j]
                for s1 in eligible[m1]:
                    dt1 = slot_dt_list[s1]
                    for s2 in eligible[m2]:
                        dt2 = slot_dt_list[s2]
                        if pd.isna(dt1) or pd.isna(dt2):
                            continue
                        diff_hours = abs((dt2 - dt1).total_seconds()) / 3600.0
                        if diff_hours < min_rest_hours:
                            model.Add(x[(m1, s1)] + x[(m2, s2)] <= 1)

        # 5) Week indicators (by chronological week labels)
        for wk in all_weeks:
            vH = model.NewBoolVar(f"H_{t}_{wk}")
            vA = model.NewBoolVar(f"A_{t}_{wk}")
            H[(t, wk)] = vH
            A[(t, wk)] = vA

            # home indicator
            hvars = []
            for m in home_of[t]:
                for s in eligible[m]:
                    if slot_week_label[s] == wk:
                        hvars.append(x[(m, s)])
            if hvars:
                for v in hvars:
                    model.Add(vH >= v)
                model.Add(vH <= sum(hvars))
            else:
                model.Add(vH == 0)

            # away indicator
            avars = []
            for m in away_of[t]:
                for s in eligible[m]:
                    if slot_week_label[s] == wk:
                        avars.append(x[(m, s)])
            if avars:
                for v in avars:
                    model.Add(vA >= v)
                model.Add(vA <= sum(avars))
            else:
                model.Add(vA == 0)

            # Optional one-match-per-week
            if weekly_limit:
                model.Add(vH + vA <= 1)

        # 6) Home/Away streak sliding windows
        for i in range(len(all_weeks) - max_home):
            win = all_weeks[i : i + max_home + 1]
            model.Add(sum(H[(t, w)] for w in win) <= max_home)

        for i in range(len(all_weeks) - max_away):
            win = all_weeks[i : i + max_away + 1]
            model.Add(sum(A[(t, w)] for w in win) <= max_away)

    # ------------------------------------------------------------
    # Solve (time-limited, multithreaded)
    # ------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)

    status = solver.Solve(model)

    # ------------------------------------------------------------
    # Build solution — sorted by true datetime ascending
    # ------------------------------------------------------------
    rows = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for m, (h, a, r_idx) in enumerate(matches):
            for s in eligible[m]:
                if solver.Value(x[(m, s)]) == 1:
                    row = slots_df.iloc[s]
                    rows.append({
                        "round": r_idx + 1,
                        "home": h,
                        "away": a,
                        "slot_id": int(row["slot_id"]),
                        "week": int(row["week"]),         # keep as label (reference only)
                        "date": str(row["date"]),
                        "time_uk": row["time_uk"],
                        "time_ist": row["time_ist"],
                        "label": row["label"],
                        "_dt": row["dt"],                 # internal for sorting
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["_dt", "round"], inplace=True)  # PURELY by datetime (then round)
        out.drop(columns=["_dt"], inplace=True)

    return out, solver.StatusName(status)
