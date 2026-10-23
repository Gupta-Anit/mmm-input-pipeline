"""Build model-ready Media Mix Modeling (MMM) inputs from raw platform exports.

Stages:
  1. Contracts   - every source must match a declared schema before it is loaded.
  2. Normalize   - one canonical daily grain: date, channel, spend_usd, impressions.
                   (micros -> dollars, EUR -> USD, flight totals -> daily spend)
  3. Load        - idempotent upsert into DuckDB, replacing a source's date range,
                   with an ingestion log keyed by file checksum.
  4. Reconcile   - monthly platform spend vs billed invoices, flagged above tolerance.
  5. Weekly mart - ISO weeks, one column per channel, zero-filled.
  6. Features    - geometric adstock + Hill saturation per channel.
  7. Sanity fit  - a ridge regression to check the inputs carry signal.

Usage:
    python mmm_inputs.py --raw raw/ --db mmm.duckdb
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

CHANNELS = ["search", "social", "ctv"]
ADSTOCK_DECAY = {"search": 0.3, "social": 0.5, "ctv": 0.8}
RECONCILE_TOLERANCE = 0.01  # 1% of billed spend

CONTRACTS: dict[str, dict[str, str]] = {
    "search_ads.csv": {"date": "date", "campaign": "str", "cost_micros": "int", "impressions": "int"},
    "social_ads.csv": {"date_start": "str", "adset": "str", "amount_spent": "float", "currency": "str",
                       "impressions": "int"},
    "ctv_flights.csv": {"flight_start": "date", "flight_end": "date", "network": "str",
                        "total_spend_usd": "float"},
    "fx_rates.csv": {"date": "date", "eur_usd": "float"},
    "signups.csv": {"date": "date", "signups": "int"},
    "invoices.csv": {"month": "str", "channel": "str", "billed_usd": "float"},
}


class ContractError(ValueError):
    pass


@dataclass
class ReconciliationIssue:
    month: str
    channel: str
    platform_usd: float
    billed_usd: float

    @property
    def pct_diff(self) -> float:
        return (self.platform_usd - self.billed_usd) / self.billed_usd


# ---------- 1. contracts ----------

def read_with_contract(path: Path) -> pd.DataFrame:
    contract = CONTRACTS[path.name]
    df = pd.read_csv(path)
    missing = set(contract) - set(df.columns)
    if missing:
        raise ContractError(f"{path.name}: missing columns {sorted(missing)}")
    for col, kind in contract.items():
        try:
            if kind == "date":
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d").dt.date
            elif kind == "int":
                if df[col].isna().any() or not np.allclose(df[col], df[col].round()):
                    raise ValueError("non-integer values")
                df[col] = df[col].astype("int64")
            elif kind == "float":
                df[col] = df[col].astype(float)
            else:
                df[col] = df[col].astype(str)
        except (ValueError, TypeError) as exc:
            raise ContractError(f"{path.name}.{col}: expected {kind} ({exc})") from exc
    return df


# ---------- 2. normalize ----------

def normalize(raw: Path) -> pd.DataFrame:
    search = read_with_contract(raw / "search_ads.csv")
    search = pd.DataFrame({
        "date": search["date"], "channel": "search",
        "spend_usd": search["cost_micros"] / 1_000_000, "impressions": search["impressions"],
    })

    social = read_with_contract(raw / "social_ads.csv")
    fx = read_with_contract(raw / "fx_rates.csv")
    # Social reports midnight UTC; the account books in UTC, so the UTC date is the spend date.
    social["date"] = pd.to_datetime(social["date_start"], utc=True).dt.date
    social = social.merge(fx, on="date", how="left")
    if social["eur_usd"].isna().any():
        raise ContractError("social_ads.csv: missing FX rate for some dates")
    rate = np.where(social["currency"] == "EUR", social["eur_usd"], 1.0)
    social = pd.DataFrame({
        "date": social["date"], "channel": "social",
        "spend_usd": social["amount_spent"] * rate, "impressions": social["impressions"],
    })

    flights = read_with_contract(raw / "ctv_flights.csv")
    rows = []
    for f in flights.itertuples():
        days = pd.date_range(f.flight_start, f.flight_end, freq="D")
        for d in days:
            rows.append((d.date(), "ctv", f.total_spend_usd / len(days), 0))
    ctv = pd.DataFrame(rows, columns=["date", "channel", "spend_usd", "impressions"])

    media = pd.concat([search, social, ctv], ignore_index=True)
    if (media["spend_usd"] < 0).any():
        raise ContractError("negative spend found after normalization")
    return media.groupby(["date", "channel"], as_index=False).sum()


# ---------- 3. idempotent load ----------

def file_checksum(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def load(con: duckdb.DuckDBPyConnection, media: pd.DataFrame, signups: pd.DataFrame, checksum: str) -> bool:
    """Upsert by replacing each channel's date range. Returns False if this batch was already loaded."""
    con.execute("""CREATE TABLE IF NOT EXISTS fct_media_daily
                   (date DATE, channel VARCHAR, spend_usd DOUBLE, impressions BIGINT)""")
    con.execute("CREATE TABLE IF NOT EXISTS fct_signups_daily (date DATE, signups BIGINT)")
    con.execute("""CREATE TABLE IF NOT EXISTS ingestion_log
                   (checksum VARCHAR PRIMARY KEY, loaded_at TIMESTAMP, media_rows BIGINT)""")
    if con.execute("SELECT 1 FROM ingestion_log WHERE checksum = ?", [checksum]).fetchall():
        return False

    con.execute("BEGIN")
    try:
        con.register("batch_media", media)
        con.execute("""
            DELETE FROM fct_media_daily t USING (
                SELECT channel, MIN(date) AS lo, MAX(date) AS hi FROM batch_media GROUP BY channel
            ) b
            WHERE t.channel = b.channel AND t.date BETWEEN b.lo AND b.hi
        """)
        con.execute("INSERT INTO fct_media_daily SELECT date, channel, spend_usd, impressions FROM batch_media")
        con.register("batch_signups", signups)
        con.execute("""DELETE FROM fct_signups_daily
                       WHERE date BETWEEN (SELECT MIN(date) FROM batch_signups)
                                      AND (SELECT MAX(date) FROM batch_signups)""")
        con.execute("INSERT INTO fct_signups_daily SELECT date, signups FROM batch_signups")
        con.execute("INSERT INTO ingestion_log VALUES (?, now(), ?)", [checksum, len(media)])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return True


# ---------- 4. reconcile ----------

def reconcile(con: duckdb.DuckDBPyConnection, invoices: pd.DataFrame,
              tolerance: float = RECONCILE_TOLERANCE) -> list[ReconciliationIssue]:
    con.register("invoices", invoices)
    rows = con.execute("""
        SELECT i.month, i.channel, COALESCE(SUM(m.spend_usd), 0) AS platform_usd, i.billed_usd
        FROM invoices i
        LEFT JOIN fct_media_daily m
          ON m.channel = i.channel AND strftime(m.date, '%Y-%m') = i.month
        GROUP BY i.month, i.channel, i.billed_usd
        ORDER BY i.month, i.channel
    """).fetchall()
    issues = [ReconciliationIssue(*r) for r in rows]
    return [x for x in issues if x.billed_usd > 0 and abs(x.pct_diff) > tolerance]


# ---------- 5. weekly mart ----------

def build_weekly(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    pivots = ", ".join(
        f"COALESCE(SUM(spend_usd) FILTER (WHERE channel = '{c}'), 0) AS spend_{c}" for c in CHANNELS)
    con.execute(f"""
        CREATE OR REPLACE TABLE mart_mmm_weekly AS
        WITH weeks AS (
            SELECT DISTINCT date_trunc('week', date) AS week FROM fct_signups_daily
        ),
        media AS (
            SELECT date_trunc('week', date) AS week, {pivots}
            FROM fct_media_daily GROUP BY 1
        ),
        sig AS (
            SELECT date_trunc('week', date) AS week, SUM(signups) AS signups,
                   COUNT(*) AS days_observed
            FROM fct_signups_daily GROUP BY 1
        )
        SELECT w.week, {", ".join(f"COALESCE(media.spend_{c}, 0) AS spend_{c}" for c in CHANNELS)},
               sig.signups
        FROM weeks w
        LEFT JOIN media USING (week)
        JOIN sig USING (week)
        WHERE sig.days_observed = 7          -- drop partial weeks at the edges
        ORDER BY w.week
    """)
    return con.execute("SELECT * FROM mart_mmm_weekly").df()


# ---------- 6. features ----------

def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.empty(len(x), dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def hill(x: np.ndarray, half_saturation: float, slope: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x ** slope / (x ** slope + half_saturation ** slope)


def build_features(weekly: pd.DataFrame) -> pd.DataFrame:
    feats = weekly[["week", "signups"]].copy()
    for c in CHANNELS:
        weekly_decay = ADSTOCK_DECAY[c] ** 7  # daily decay compounded over a week
        ad = geometric_adstock(weekly[f"spend_{c}"].to_numpy(), weekly_decay)
        half = np.median(ad[ad > 0]) if (ad > 0).any() else 1.0
        feats[f"{c}_adstock"] = ad
        feats[f"{c}_saturated"] = hill(ad, half)
    return feats


# ---------- 7. sanity fit ----------

def ridge_fit(features: pd.DataFrame, alpha: float = 1.0) -> pd.Series:
    cols = [f"{c}_saturated" for c in CHANNELS]
    x = features[cols].to_numpy()
    y = features["signups"].to_numpy(dtype=float)
    x_mean, y_mean = x.mean(0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    coef = np.linalg.solve(xc.T @ xc + alpha * np.eye(len(cols)), xc.T @ yc)
    pred = y_mean + xc @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y_mean) ** 2).sum()
    return pd.Series({**dict(zip(cols, coef.round(1))), "r2": round(float(r2), 3)})


def run(raw: Path, db: str = ":memory:") -> dict:
    con = duckdb.connect(db)
    media = normalize(raw)
    signups = read_with_contract(raw / "signups.csv")
    loaded = load(con, media, signups, file_checksum(list(raw.glob("*.csv"))))
    issues = reconcile(con, read_with_contract(raw / "invoices.csv"))
    weekly = build_weekly(con)
    features = build_features(weekly)
    con.register("features_df", features)
    con.execute("CREATE OR REPLACE TABLE mart_mmm_features AS SELECT * FROM features_df")
    return {"loaded": loaded, "issues": issues, "weekly": weekly, "fit": ridge_fit(features)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--db", default="mmm.duckdb")
    a = ap.parse_args()
    res = run(Path(a.raw), a.db)
    print("new batch loaded" if res["loaded"] else "batch already loaded, skipped (idempotent)")
    print(f"weeks in mart: {len(res['weekly'])}")
    print(f"reconciliation issues (>{RECONCILE_TOLERANCE:.0%}): {len(res['issues'])}")
    for i in res["issues"][:10]:
        print(f"  {i.month} {i.channel}: platform ${i.platform_usd:,.0f} vs billed ${i.billed_usd:,.0f} "
              f"({i.pct_diff:+.1%})")
    print("sanity fit (signups ~ saturated media):")
    print(res["fit"].to_string())


if __name__ == "__main__":
    main()
