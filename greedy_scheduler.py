import numpy as np
from collections import defaultdict

def greedy_schedule(rounds, slots, team_names, opts):
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
        last, cnt = streak[t]
        if last != val: return True
        if val == "H": return cnt + 1 <= maxH
        return cnt + 1 <= maxA

    def push_streak(t, val):
        last, cnt = streak[t]
        streak[t] = (val, cnt + 1) if last == val else (val, 1)

    assignments = []
    s_idx = 0

    for rnd_idx, matches in enumerate(rounds, start=1):
        for home, away in matches:
            placed = False
            j = s_idx
            while j < len(slots):
                if slot_used[j]:
                    j += 1; continue
                row = slots.iloc[j]
                w = int(row["week"]); dt = row["dt_uk"]

                if not ok_week(home, w) or not ok_week(away, w):
                    j += 1; continue
                if not ok_rest(home, dt) or not ok_rest(away, dt):
                    j += 1; continue
                if not ok_streak(home, "H") or not ok_streak(away, "A"):
                    j += 1; continue

                slot_used[j] = True
                team_last[home] = dt; team_last[away] = dt
                team_week[home].add(w); team_week[away].add(w)
                push_streak(home, "H"); push_streak(away, "A")

                assignments.append({
                    "round": rnd_idx, "home": home, "away": away,
                    "slot_id": row["slot_id"], "week": w,
                    "date": row["date"], "time_uk": row["time_uk"],
                    "time_ist": row["time_ist"], "label": row["label"]
                })
                placed = True
                if j == s_idx:
                    s_idx += 1
                break

            if not placed:
                assignments.append({
                    "round": rnd_idx, "home": home, "away": away,
                    "slot_id": None, "week": None, "date": None,
                    "time_uk": None, "time_ist": None, "label": "UNASSIGNED"
                })

    return assignments
