import pandas as pd
from datetime import datetime

def load_data():
    teams = pd.read_csv("teams.csv")
    slots = pd.read_csv("slots.csv")

    try:
        dists = pd.read_csv("distances.csv")
    except:
        dists = pd.DataFrame(columns=["team_a", "team_b", "distance_km"])

    def parse_dt(s):
        try:
            parts = s.rsplit(" ", 1)
            return datetime.strptime(parts[0], "%Y-%m-%d %H:%M")
        except:
            return None

    slots["dt_uk"] = slots["time_uk"].apply(parse_dt)
    slots["dt_ist"] = slots["time_ist"].apply(parse_dt)
    slots["is_weekend"] = slots["day_name"].isin(["Saturday", "Sunday"])

    if "parallel_index" in slots.columns:
        slots.sort_values(["dt_uk", "parallel_index"], inplace=True)
    else:
        slots.sort_values(["dt_uk"], inplace=True)

    return teams, dists, slots
