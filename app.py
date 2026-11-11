import re
import requests
import pdfplumber
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="Racing Strategy Matrix", page_icon="🏇", layout="wide")

# ==========================================
# 1. SOUTH AFRICA PARSER (RACE COAST PDF)
# ==========================================
def parse_sa_racecoast_pdf(uploaded_file) -> pd.DataFrame:
    runners = []
    current_race = None
    race_distance = 1200
    speed_map = {}
    rest_map = {}

    # Matches: "HOLLYWOODBETS KENILWORTH RACE 1 - Abt/1000m"
    race_header_re = re.compile(r"RACE\s+(\d+)\s*-\s*(?:Abt/)?(\d+)m", re.IGNORECASE)
    
    # Matches: "1 MAKING MILLIONS A 1 59" OR "2 BURSTURBUBBLE A 2 52.5 (64)"
    runner_header_re = re.compile(
        r"^(?P<No>\d{1,2})\s+"
        r"(?P<Horse>[A-Z0-9\s\'\-\’]+?)\s+"
        r"(?P<Gear>[A-Za-z\s]{1,8}?)\s+"
        r"(?P<Draw>\d{1,2}|RESERVE\s*\d)\s+"
        r"(?P<Weight>\d{2}(?:\.\d)?)"
        r"(?:\s+\((?P<MR>\d+)\))?"
        r"(?:\s+\[(?P<LastMR>\d+)\])?$"
    )

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [line.strip() for line in text.splitlines() if line.strip()]

            # Skip ad/index pages that don't have race headers
            page_race_match = None
            for line in lines[:5]:
                m = race_header_re.search(line)
                if m:
                    page_race_match = m
                    break
            if not page_race_match:
                continue

            current_race = int(page_race_match.group(1))
            race_distance = int(page_race_match.group(2))

            # Extract Race-Level Speed Ratings & Days Rested blocks
            full_page_text = " ".join(lines)
            speed_block = re.search(r"SPEED RATINGS:\s*(.*?)(?:BEST AVERAGE:|DAYS RESTED:)", full_page_text)
            if speed_block:
                for item in speed_block.group(1).split(";"):
                    sm = re.search(r"([A-Za-z\s\'\-\’]+?)\s+(-?\d+\.\d+)", item.strip())
                    if sm:
                        speed_map[sm.group(1).strip().upper()] = float(sm.group(2))

            rest_block = re.search(r"DAYS RESTED:\s*(.*?)(?:\b1\s+[A-Z]{2,})", full_page_text)
            if rest_block:
                for item in rest_block.group(1).split(";"):
                    rm = re.search(r"([A-Za-z\s\'\-\’]+?)\s*\((\d+)\s*days\)", item.strip(), re.IGNORECASE)
                    if rm:
                        rest_map[rm.group(1).strip().upper()] = int(rm.group(2))

            # Parse individual runners on the page
            current_runner = None
            for line in lines:
                # Check for runner header
                rm = runner_header_re.match(line)
                if rm:
                    if current_runner:
                        runners.append(current_runner)

                    horse_name = rm.group("Horse").strip().upper()
                    draw_raw = rm.group("Draw")
                    draw_num = int(draw_raw) if draw_raw.isdigit() else 99
                    mr_val = int(rm.group("MR")) if rm.group("MR") else 0

                    current_runner = {
                        "Race": current_race,
                        "Distance": race_distance,
                        "No": int(rm.group("No")),
                        "Horse": horse_name,
                        "Draw": draw_num,
                        "Weight": float(rm.group("Weight")),
                        "MR": mr_val,
                        "Runs": 0,
                        "Wins": 0,
                        "Places": 0,
                        "Stakes": 0,
                        "Speed_Rating": speed_map.get(horse_name, np.nan),
                        "Days_Rested": rest_map.get(horse_name, 21),  # Default ~3 weeks if not listed as rested
                        "Last_3_Pos": [],
                        "Last_Times": []
                    }
                    continue

                if current_runner:
                    # Extract Runs / Wins / Places / Stakes
                    stats_m = re.search(
                        r"Runs:\s*(\d+)\s+Wins:\s*(\d+)\s+Places:\s*(\d+)\s+Stakes Won:\s*R([\d,]+)",
                        line
                    )
                    if stats_m:
                        current_runner["Runs"] = int(stats_m.group(1))
                        current_runner["Wins"] = int(stats_m.group(2))
                        current_runner["Places"] = int(stats_m.group(3))
                        current_runner["Stakes"] = int(stats_m.group(4).replace(",", ""))
                        continue

                    # Extract past run lines (e.g., "D 92 26.10.14 G Omdm 1000 ... 4 4.00 Secret Tactic 60.5 58.68 5/2 411kg")
                    run_m = re.match(
                        r"^[A-Z]\s+\d+\s+\d{2}\.\d{2}\.\d{2}\s+[A-Z]\s+\S+\s+(\d{3,4})\s+.*?\b(\d{1,2})\s+(\d+\.\d{2})\s+.*?\s+(\d{2,3}\.\d{2})\s+\d+/",
                        line
                    )
                    if run_m:
                        dist_ran = int(run_m.group(1))
                        finish_pos = int(run_m.group(2))
                        time_sec = float(run_m.group(4))
                        current_runner["Last_3_Pos"].append(finish_pos)
                        # Normalize speed in meters/sec
                        if time_sec > 0:
                            current_runner["Last_Times"].append(dist_ran / time_sec)

            if current_runner:
                runners.append(current_runner)

    df = pd.DataFrame(runners)
    if df.empty:
        return df

    # Compute clean numeric features for the Strategy Matrix
    df["Avg_Speed_MPS"] = df["Last_Times"].apply(lambda x: np.mean(x[-3:]) if len(x) > 0 else np.nan)
    df["Avg_Speed_MPS"] = df["Avg_Speed_MPS"].fillna(df.groupby("Race")["Avg_Speed_MPS"].transform("mean")).fillna(16.0)
    
    # Fill missing Speed Ratings with normalized Avg_Speed_MPS
    df["Speed_Metric"] = df["Speed_Rating"].fillna(
        df.groupby("Race")["Avg_Speed_MPS"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-5))
    ).fillna(0.0)

    df["Earnings_Per_Run"] = np.where(df["Runs"] > 0, df["Stakes"] / df["Runs"], 0)
    df["Win_Place_Rate"] = np.where(df["Runs"] > 0, (df["Wins"] * 2 + df["Places"]) / df["Runs"], 0)
    df["Recent_Finish_Avg"] = df["Last_3_Pos"].apply(lambda x: np.mean(x[-3:]) if len(x) > 0 else 8.0)

    return df.drop(columns=["Last_3_Pos", "Last_Times"])


# ==========================================
# 2. HONG KONG PARSER (HKJC URL OR PDF)
# ==========================================
def scrape_hkjc_racecard_url(base_url: str) -> pd.DataFrame:
    """Scrapes all races from a ://hkjc.com racecard URL."""
    all_runners = []
    headers = {"User-Agent": "Mozilla/5.0"}

    # Clean URL to iterate through Race 1 to 11
    clean_base = re.sub(r"&RaceNo=\d+", "", base_url)
    sep = "&" if "?" in clean_base else "?"

    for race_no in range(1, 12):
        url = f"{clean_base}{sep}RaceNo={race_no}"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "lxml")
        tables = soup.find_all("table")
        target_table = None

        for tbl in tables:
            if "Horse No." in tbl.get_text() and "Last 6 Runs" in tbl.get_text():
                target_table = tbl
                break

        if not target_table:
            break

        rows = target_table.find_all("tr")
        for tr in rows[1:]:
            cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if len(cols) >= 10 and cols.isdigit():
                last_6 = [int(x) for x in cols.split("/") if x.isdigit()]
                avg_pos = np.mean(last_6) if last_6 else 8.0
                rtg = int(cols) if cols.isdigit() else 50
                rtg_diff = int(cols.replace("+", "")) if re.match(r"^[+-]?\d+$", cols) else 0
                wt = int(cols) if cols.isdigit() else 125
                draw = int(cols) if cols.isdigit() else 7

                all_runners.append({
                    "Race": race_no,
                    "No": int(cols),
                    "Horse": cols,
                    "Last_6_Runs": cols,
                    "Recent_Finish_Avg": avg_pos,
                    "Weight": wt,
                    "Jockey": cols,
                    "Draw": draw,
                    "Trainer": cols,
                    "MR": rtg,
                    "Rtg_Change": rtg_diff,
                    "Horse_Wt": int(cols) if cols.isdigit() else 1100,
                    "Gear": cols if len(cols) > 11 else ""
                })

    return pd.DataFrame(all_runners)


def parse_hkjc_pdf(uploaded_file) -> pd.DataFrame:
    """Parses HKJC 'Download Race Form (All Races)' PDF."""
    all_runners = []
    current_race = 1

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            rm = re.search(r"Race\s+(\d+)", text)
            if rm:
                current_race = int(rm.group(1))

            for table in page.extract_tables():
                for row in table:
                    c = [cell.replace("\n", " ").strip() if cell else "" for cell in row]
                    if len(c) >= 9 and c.isdigit() and 1 <= int(c) <= 14:
                        last_6 = [int(x) for x in c.split("/") if x.isdigit()]
                        all_runners.append({
                            "Race": current_race,
                            "No": int(c),
                            "Last_6_Runs": c,
                            "Recent_Finish_Avg": np.mean(last_6) if last_6 else 8.0,
                            "Horse": c,
                            "Weight": int(c) if c.isdigit() else 125,
                            "Jockey": c,
