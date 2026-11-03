import datetime as dt, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import pandas as pd, pytest
from audit import maturity as M

CAL = M.sessions_2026()
D = lambda s: {"ticker": "X" + s, "discovered_at": s}


def row(scan, last="2026-10-30", H=63):
    return M.table([D(scan)], pd.Timestamp(last), CAL).iloc[0]


def test_entry_is_the_first_session_strictly_after_the_scan_date():
    assert row("2026-08-06").entry_session == "2026-08-07"            # Thursday scan -> Friday open
    assert row("2026-10-31").entry_session == "2026-11-02"            # Saturday scan -> Monday open
    assert row("2026-10-30").entry_session == "2026-11-02"            # Friday scan -> Monday open, never the same day
    assert row("2026-09-04").entry_session == "2026-09-08"            # Labor Day skipped


def test_no_entry_session_yet_when_the_next_open_is_after_the_data():
    r = row("2026-10-30"); assert not r.has_entry and r.sessions_since_entry == -1 and not r.matured_21


def test_maturity_boundary_is_exact_for_each_horizon():
    e = CAL.get_loc(pd.Timestamp("2026-08-07"))
    for H in (21, 63):
        last_ok = CAL[e + H]; last_short = CAL[e + H - 1]
        assert M.table([D("2026-08-06")], last_ok, CAL).iloc[0][f"matured_{H}"]
        assert not M.table([D("2026-08-06")], last_short, CAL).iloc[0][f"matured_{H}"]


def test_last_completed_session_is_strictly_before_today():
    assert M.last_completed_session(dt.date(2026, 11, 1), CAL) == pd.Timestamp("2026-10-30")     # Sunday
    assert M.last_completed_session(dt.date(2026, 11, 2), CAL) == pd.Timestamp("2026-10-30")     # Monday: Friday is the last complete one
    assert M.last_completed_session(dt.date(2026, 11, 3), CAL) == pd.Timestamp("2026-11-02")


def test_summary_counts_and_bands_add_up():
    ds = [D("2026-08-06"), D("2026-09-30"), D("2026-10-20"), D("2026-10-31")]
    df = M.table(ds, pd.Timestamp("2026-10-30"), CAL); s = M.summary(df)
    assert s["discoveries"] == 4 and s["no_entry_yet"] == 1 and sum(s["bands"].values()) == 4
    assert s["matured_21"] == 2 and s["matured_63"] == 0 and s["immature_63"] == 4


def test_earliest_maturity_date_matches_the_hand_count():
    df = M.table([D("2026-08-06")], pd.Timestamp("2026-10-30"), CAL)
    assert M.earliest_maturity(df, CAL, 63) == dt.date(2026, 11, 5)


def test_the_real_discovery_log_has_no_63_session_matured_observation_at_30_oct():
    disc = json.loads((REPO / "data" / "discoveries.json").read_text(encoding="utf-8"))
    s = M.summary(M.table(disc, pd.Timestamp("2026-10-30"), CAL))
    assert s["matured_63"] == 0 and s["matured_126"] == 0 and s["discoveries"] == len(disc) and s["matured_21"] > 500
