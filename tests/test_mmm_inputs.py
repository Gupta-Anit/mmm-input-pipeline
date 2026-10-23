import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import mmm_inputs as m
from generate_exports import generate


@pytest.fixture(scope="module")
def raw(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("raw")
    generate(p, seed=5)
    return p


def test_contract_rejects_missing_column(tmp_path, raw):
    shutil.copytree(raw, tmp_path / "r", dirs_exist_ok=True)
    df = pd.read_csv(tmp_path / "r" / "search_ads.csv").drop(columns=["cost_micros"])
    df.to_csv(tmp_path / "r" / "search_ads.csv", index=False)
    with pytest.raises(m.ContractError, match="cost_micros"):
        m.normalize(tmp_path / "r")


def test_contract_rejects_bad_types(tmp_path, raw):
    shutil.copytree(raw, tmp_path / "r", dirs_exist_ok=True)
    df = pd.read_csv(tmp_path / "r" / "signups.csv")
    df["signups"] = df["signups"].astype(float)
    df.loc[3, "signups"] = 12.5
    df.to_csv(tmp_path / "r" / "signups.csv", index=False)
    with pytest.raises(m.ContractError, match="signups"):
        m.read_with_contract(tmp_path / "r" / "signups.csv")


def test_units_and_currency_are_normalized(raw):
    media = m.normalize(raw)
    search = pd.read_csv(raw / "search_ads.csv")
    assert media.query("channel == 'search'")["spend_usd"].sum() == pytest.approx(search["cost_micros"].sum() / 1e6)
    social = pd.read_csv(raw / "social_ads.csv")
    fx = pd.read_csv(raw / "fx_rates.csv")
    expected = (social["amount_spent"].to_numpy() * fx["eur_usd"].to_numpy()).sum()
    assert media.query("channel == 'social'")["spend_usd"].sum() == pytest.approx(expected)


def test_ctv_flights_spread_to_daily(raw):
    media = m.normalize(raw)
    flights = pd.read_csv(raw / "ctv_flights.csv")
    assert media.query("channel == 'ctv'")["spend_usd"].sum() == pytest.approx(flights["total_spend_usd"].sum())


def test_load_is_idempotent(raw):
    con = duckdb.connect()
    media = m.normalize(raw)
    signups = m.read_with_contract(raw / "signups.csv")
    assert m.load(con, media, signups, "abc") is True
    first = con.execute("SELECT COUNT(*), SUM(spend_usd) FROM fct_media_daily").fetchone()
    assert m.load(con, media, signups, "abc") is False           # same checksum: skipped
    assert m.load(con, media, signups, "restated") is True       # restatement: replaces, never duplicates
    again = con.execute("SELECT COUNT(*), SUM(spend_usd) FROM fct_media_daily").fetchone()
    assert again[0] == first[0] and again[1] == pytest.approx(first[1])


def test_reconciliation_flags_billing_gap(raw):
    con = duckdb.connect()
    m.load(con, m.normalize(raw), m.read_with_contract(raw / "signups.csv"), "x")
    invoices = m.read_with_contract(raw / "invoices.csv")
    assert m.reconcile(con, invoices) == []
    idx = invoices.index[(invoices["channel"] == "social")][2]
    invoices.loc[idx, "billed_usd"] *= 1.05
    issues = m.reconcile(con, invoices)
    assert len(issues) == 1 and issues[0].channel == "social"
    assert issues[0].pct_diff == pytest.approx(-0.05 / 1.05, rel=1e-3)


def test_weekly_mart_has_full_weeks_only(raw):
    res = m.run(raw)
    weekly = res["weekly"]
    assert (pd.to_datetime(weekly["week"]).dt.dayofweek == 0).all()
    total_media = m.normalize(raw)["spend_usd"].sum()
    mart_total = weekly[[f"spend_{c}" for c in m.CHANNELS]].to_numpy().sum()
    assert mart_total == pytest.approx(total_media)


def test_adstock_and_hill():
    np.testing.assert_allclose(m.geometric_adstock(np.array([100, 0, 0]), 0.5), [100, 50, 25])
    assert m.hill(np.array([10.0]), half_saturation=10.0)[0] == pytest.approx(0.5)
    assert np.all(np.diff(m.hill(np.linspace(0, 100, 50), 20)) > 0)


def test_sanity_fit_recovers_positive_effects(raw):
    fit = m.run(raw)["fit"]
    assert fit["r2"] > 0.3
    assert all(fit[f"{c}_saturated"] > 0 for c in m.CHANNELS)
