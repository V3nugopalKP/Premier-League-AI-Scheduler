# ⚽ Premier League AI Match Scheduler  
### _Greedy + Constraint Satisfaction (CSP) Based Fixture Generator_

This project is an **AI-driven scheduling engine** that generates a **full-season Premier League fixture list** using:

✅ **Greedy heuristic scheduling** (fast baseline)  
✅ **Constraint Satisfaction (CSP)** using Google OR-Tools  
✅ Realistic time slots, date/week mapping, IST/UK conversion  
✅ Dynamic home/away streak limits, rest gaps, UEFA block weeks  
✅ Beautiful Streamlit UI for exploring schedules  

The system simulates a **real-world sports scheduling problem** using AI and combinatorial optimization.

---

## 🚀 Features

### ✅ **1. Premier League–Style Structure**
- 20 teams  
- Double Round Robin → 38 rounds, 380 matches  
- Circle Method fixture generation  

### ✅ **2. Two Scheduling Engines**

#### **🔹 Greedy Scheduler (Fast)**
- Assigns fixtures sequentially  
- Respects:
  - Max consecutive home games  
  - Max consecutive away games  
  - Minimum rest gap (hours)  
  - Optional one-match-per-week limit  
- Will always finish instantly  
- Might leave some matches unassigned  

#### **🔹 CSP Scheduler (AI-Optimized)**
Uses **Google OR-Tools CP-SAT** to satisfy hard constraints:

- ✅ One match per slot  
- ✅ One match per team per date  
- ✅ Minimum rest gap (48/72 hrs)  
- ✅ Home/Away streak control  
- ✅ Weekly limit (optional)  
- ✅ Supports UEFA block weeks  
- ✅ Time-limited solving (ensures fast completion)  
- ✅ Search domain pruning via week-window logic  

CSP produces a **chronologically sorted schedule**, not just week-based.

---

## 📅 Real Calendar Slot System

The dataset (`slots.csv`) includes:

- Realistic match dates  
- UK time slots (15:00, 20:00, etc.)  
- Parallel matches  
- IST conversion  
- Real calendar week numbers  

The AI assigns matches precisely into these slots.

---

## 🎯 Project Goal

To build an **intelligent fixture scheduling system** that follows constraints similar to real-world football leagues using:

- Constraint Satisfaction  
- Heuristic Search  
- Combinatorial Optimization  
- Interactive UI  

This project demonstrates practical use of **AI planning & constraint reasoning**.

---

## 🧠 Tech Stack

| Component | Technology Used |
|----------|-----------------|
| Frontend UI | Streamlit |
| AI Solver | OR-Tools CP-SAT |
| Greedy Scheduler | Python Heuristics |
| Data Handling | Pandas, NumPy |
| Plotting | Matplotlib |
| Datasets | Premier League Teams + Calendar Slots |

---

## 🗂 Project Structure

📦 **Premier-League-AI-Scheduler**
│
├── **app.py** — Main Streamlit UI  
├── **data_loader.py** — Loads teams, slots, distances  
├── **round_robin.py** — Generates double round robin  
├── **greedy_scheduler.py** — Greedy match assignment  
├── **csp_scheduler.py** — CSP-based match scheduler  
│
├── **teams.csv** — Team info + coordinates  
├── **slots.csv** — Calendar time slots  
├── **distances.csv** — For Future Use 
│
└── **README.md** — Documentation

## ▶ Running the App

Start the Streamlit UI:

```bash
streamlit run app.py
