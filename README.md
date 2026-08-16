# ⚽ Premier League AI Match Scheduler

**Greedy + Constraint Satisfaction (CSP) Based Fixture Generator**

An AI-driven scheduling engine that generates a full-season Premier League fixture list using greedy heuristics and constraint satisfaction (CSP), simulating a real-world sports scheduling problem through combinatorial optimization.

## Overview

The system generates a realistic, constraint-compliant Premier League schedule — 20 teams, double round robin, 38 rounds, 380 matches — using two interchangeable scheduling engines: a fast greedy heuristic and an AI-optimized CSP solver built on Google OR-Tools. Matches are assigned to real calendar slots with UK/IST time conversion, and the full schedule is explorable through a Streamlit UI.

## Project Goal

To build an intelligent fixture scheduling system that follows constraints similar to real-world football leagues, demonstrating practical use of:

- Constraint satisfaction
- Heuristic search
- Combinatorial optimization
- Interactive UI

## Features

### Premier League–Style Structure
- 20 teams
- Double round robin → 38 rounds, 380 matches
- Circle Method fixture generation

### Two Scheduling Engines

**Greedy Scheduler (fast baseline)**
- Assigns fixtures sequentially
- Respects max consecutive home/away games, minimum rest gap (hours), and an optional one-match-per-week limit
- Always finishes instantly, but may leave some matches unassigned

**CSP Scheduler (AI-optimized)**

Uses Google OR-Tools CP-SAT to satisfy hard constraints:
- One match per slot, one match per team per date
- Minimum rest gap (48/72 hrs)
- Home/away streak control
- Optional weekly limit
- Support for UEFA block weeks
- Time-limited solving for fast completion
- Search domain pruning via week-window logic

Produces a chronologically sorted schedule, not just a week-based one.

### Real Calendar Slot System

`slots.csv` provides the scheduling grid the AI assigns matches into, including:
- Realistic match dates and UK time slots (15:00, 20:00, etc.)
- Parallel matches
- IST conversion
- Real calendar week numbers

## Tech Stack

| Component | Technology |
|---|---|
| Frontend UI | Streamlit |
| AI Solver | OR-Tools CP-SAT |
| Greedy Scheduler | Python heuristics |
| Data Handling | Pandas, NumPy |
| Plotting | Matplotlib |
| Datasets | Premier League teams + calendar slots |

## Project Structure

```
Premier-League-AI-Scheduler/
├── app.py                 # Main Streamlit UI
├── data_loader.py          # Loads teams, slots, distances
├── round_robin.py          # Generates double round robin
├── greedy_scheduler.py     # Greedy match assignment
├── csp_scheduler.py        # CSP-based match scheduler
├── teams.csv                # Team info + coordinates
├── slots.csv                # Calendar time slots
├── distances.csv            # Reserved for future use
└── README.md
```

## Installation

```bash
git clone https://github.com/V3nugopalKP/Premier-League-AI-Scheduler.git
cd Premier-League-AI-Scheduler
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Future Improvements

- Incorporate travel distance (`distances.csv`) into scheduling constraints
- Add derby/rivalry match spacing rules
- Support broadcast-slot prioritization
- Export generated schedules to calendar formats (ICS/CSV)

## Author

**Venugopal K P**
M.Tech Computer Science & Engineering (AI & ML), Amrita Vishwa Vidyapeetham

## License

Licensed under the MIT License.
