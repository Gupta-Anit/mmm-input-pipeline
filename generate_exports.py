"""Generate messy, realistic marketing platform exports for the MMM pipeline.

Each source looks the way it does in real life:
  * search_ads.csv  - daily rows, cost in micros, account in US/Pacific
  * social_ads.csv  - daily rows, spend in EUR (with an FX file), dates in UTC
  * ctv_flights.csv - one row per flight (start/end/total spend), no daily grain
  * signups.csv     - daily sign-ups, US/Eastern
  * invoices.csv    - monthly billed spend per channel, used for reconciliation

Signups are simulated from adstocked, saturated spend so the pipeline's
sanity model has a known ground truth to recover.

Usage:
    python generate_exports.py --out raw/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "2025-01-06", "2026-06-28"  # Monday to Sunday, 77 full weeks


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def generate(out: Path, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(START, END, freq="D")
    n = len(days)
    season = 1 + 0.25 * np.sin(2 * np.pi * np.arange(n) / 365)

    search = rng.gamma(9, 900, n) * season
    social_usd = rng.gamma(6, 800, n) * season
    eur_usd = 1.08 + 0.03 * np.sin(np.arange(n) / 40)

    # CTV runs in flights: bursts of 2-3 weeks, sometimes around live sports events.
    ctv_daily = np.zeros(n)
    flights = []
    i = 10
    while i < n - 20:
        length = int(rng.integers(14, 22))
        total = float(rng.uniform(150_000, 320_000))
        ctv_daily[i:i + length] = total / length
        flights.append((days[i].date(), days[i + length - 1].date(), "ctv_network_a", round(total, 2)))
        i += length + int(rng.integers(20, 45))

    base = 900 * season
    signups = (
        base
        + 22 * np.sqrt(adstock(search, 0.3))
        + 18 * np.sqrt(adstock(social_usd, 0.5))
        + 9 * np.sqrt(adstock(ctv_daily, 0.8))
        + rng.normal(0, 40, n)
    ).round()

    pd.DataFrame({
        "date": days.date, "campaign": "brand_and_generic",
        "cost_micros": (search * 1_000_000).astype("int64"),
        "impressions": (search * rng.uniform(55, 70, n)).astype(int),
        "account_timezone": "America/Los_Angeles",
    }).to_csv(out / "search_ads.csv", index=False)

    social_eur = social_usd / eur_usd
    pd.DataFrame({
        "date_start": [d.strftime("%Y-%m-%dT00:00:00Z") for d in days],
        "adset": "prospecting",
        "amount_spent": social_eur.round(2),
        "currency": "EUR",
        "impressions": (social_usd * rng.uniform(80, 110, n)).astype(int),
    }).to_csv(out / "social_ads.csv", index=False)

    pd.DataFrame({"date": days.date, "eur_usd": eur_usd.round(4)}).to_csv(out / "fx_rates.csv", index=False)
    pd.DataFrame(flights, columns=["flight_start", "flight_end", "network", "total_spend_usd"]).to_csv(
        out / "ctv_flights.csv", index=False)
    pd.DataFrame({"date": days.date, "signups": signups.astype(int)}).to_csv(out / "signups.csv", index=False)

    month = pd.Series(days).dt.to_period("M").astype(str).values
    inv = pd.DataFrame({"month": month, "search": search, "social": social_usd, "ctv": ctv_daily})
    inv = inv.groupby("month").sum().round(2).reset_index().melt(
        id_vars="month", var_name="channel", value_name="billed_usd")
    inv.to_csv(out / "invoices.csv", index=False)
    print(f"wrote {n} days, {len(flights)} CTV flights to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    generate(Path(a.out), a.seed)
