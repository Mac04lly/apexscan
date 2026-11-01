import sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd, pytest
from audit import execution as E

CAL = pd.bdate_range("2025-01-02", periods=300)

def stock(vol=2_000_000.0, price=20.0, cal=CAL):
    c = price + np.arange(len(cal)) * 0.05
    return pd.DataFrame({"Open": c, "High": c * 1.02, "Low": c * 0.98, "Close": c, "Volume": vol}, index=cal)

T = CAL[100]                                        # signal session

def test_entry_is_next_session_open_and_exit_is_63_sessions_later_with_hand_calculated_return():
    h = stock(); r = E.plan_trade(h, CAL, T)
    assert r["status"] == "ok" and r["entry_date"] == CAL[101] and r["exit_date"] == CAL[164] and r["holding_sessions"] == 63
    eo, xo, c = h.at[CAL[101], "Open"], h.at[CAL[164], "Open"], E.cost_per_side(20.0 * 2e6)
    assert c == 0.0010 and r["net_return"] == pytest.approx(xo * (1 - c) / (eo * (1 + c)) - 1)

@pytest.mark.parametrize("dv,c", [(20e6, .001), (19_999_999, .003), (2e6, .003), (1_999_999, .0075), (0, .0075)])
def test_cost_tier_boundaries(dv, c): assert E.cost_per_side(dv) == c

def test_cost_multiplier_applies_to_both_sensitivities():
    assert E.cost_per_side(30e6, 0.5) == pytest.approx(.0005) and E.cost_per_side(30e6, 2) == pytest.approx(.002)

def test_missing_stock_row_does_not_shift_the_exit_session():
    h = stock().drop(CAL[120]); r = E.plan_trade(h, CAL, T)
    assert r["status"] == "ok" and r["exit_date"] == CAL[164]

@pytest.mark.parametrize("mut,reason", [
    (lambda h: h.drop(CAL[101]),                                         "entry_session_missing"),
    (lambda h: h.assign(Open=h.Open.mask(h.index == CAL[101])),          "entry_open_missing"),
    (lambda h: h.assign(Open=h.Open.mask(h.index == CAL[101], 0.0)),     "entry_nonpositive_open"),
    (lambda h: h.assign(Open=h.Open.mask(h.index == CAL[164])),          "exit_open_missing"),
    (lambda h: h.drop(CAL[164]),                                         "exit_session_missing"),
    (lambda h: h[h.index < CAL[150]],                                    "no_exit_session_stock_stopped_trading"),
    (lambda h: h.assign(Close=h.Close.mask(h.index == CAL[95])),         "cost_tier_unavailable"),
    (lambda h: h.assign(Volume=h.Volume.mask(h.index == CAL[95])),       "cost_tier_unavailable"),
    (lambda h: h.drop(CAL[95]),                                          "cost_tier_unavailable"),
])
def test_every_kind_of_missing_data_excludes_the_trade_with_its_own_reason(mut, reason):
    h = mut(stock()); r = E.plan_trade(h, CAL, T)
    assert r["status"] == "excluded" and r["reason"] == reason and "net_return" not in r

def test_missing_high_low_keeps_the_trade_but_marks_excursions_unavailable():
    h = stock(); h.loc[CAL[130], "High"] = np.nan
    r = E.plan_trade(h, CAL, T); assert r["status"] == "ok" and np.isnan(r["mfe"]) and r["excursion"].startswith("excursion_unavailable")

def test_excursions_are_computed_when_data_is_complete():
    h = stock(); r = E.plan_trade(h, CAL, T); assert r["excursion"] == "ok" and r["mfe"] > 0 and r["mae"] < 0

def test_incomplete_horizon_and_end_of_calendar():
    assert E.plan_trade(stock(), CAL, CAL[250])["reason"] == "horizon_incomplete"
    assert E.plan_trade(stock(), CAL, CAL[-1])["reason"] == "entry_session_not_yet_available"

def test_inputs_are_never_modified_or_filled():
    h = stock(); h.loc[CAL[101], "Open"] = np.nan; b = h.copy(); E.plan_trade(h, CAL, T); pd.testing.assert_frame_equal(h, b)

# ---------- paired E2 vs E1 (Amendment 5) ----------
T2, T1 = CAL[100], CAL[110]

def test_paired_uses_each_next_open_and_ONE_common_exit_63_sessions_after_the_E1_entry():
    h = stock(); r = E.paired_e2_vs_e1(h, CAL, T2, T1)
    assert r["status"] == "ok" and r["common_exit_date"] == CAL[111 + 63]
    e2o, e1o, xo = h.at[CAL[101], "Open"], h.at[CAL[111], "Open"], h.at[CAL[174], "Open"]
    c = 0.0010
    assert r["e1_net"] == pytest.approx(xo * (1 - c) / (e1o * (1 + c)) - 1)
    assert r["e2_net"] == pytest.approx(xo * (1 - c) / (e2o * (1 + c)) - 1)
    assert r["diff_e2_minus_e1"] == pytest.approx(r["e2_net"] - r["e1_net"])

def test_paired_discloses_the_different_holding_periods():
    r = E.paired_e2_vs_e1(stock(), CAL, T2, T1)
    assert r["e1_holding_sessions"] == 63 and r["e2_holding_sessions"] == 73 and r["gap_sessions"] == 10

def test_paired_each_leg_pays_its_own_cost_tier():
    h = stock(); h.loc[CAL[81]:CAL[100], "Volume"] = 50_000.0            # E2's liquidity window is thin (1M dollars)
    r = E.paired_e2_vs_e1(h, CAL, T2, T1)
    assert r["e2_cost_side"] == 0.0075 and r["e1_cost_side"] in (0.0010, 0.0030)

def test_paired_excludes_when_gap_over_60_or_order_wrong_or_any_leg_missing():
    assert E.paired_e2_vs_e1(stock(), CAL, CAL[50], CAL[111])["reason"] == "e1_more_than_60_sessions_after_e2"
    assert E.paired_e2_vs_e1(stock(), CAL, CAL[110], CAL[100])["reason"] == "e1_not_after_e2"
    h = stock().drop(CAL[111]); assert E.paired_e2_vs_e1(h, CAL, T2, T1)["reason"] == "e1_leg_entry_session_missing"
    h = stock().drop(CAL[61]); assert E.paired_e2_vs_e1(h, CAL, CAL[60], CAL[100])["reason"] == "e2_leg_entry_session_missing"

def test_exclusions_are_reported_by_ticker_and_approach():
    recs = [{"ticker": "A", "approach": "E1", **E.plan_trade(stock().drop(CAL[101]), CAL, T)},
            {"ticker": "A", "approach": "E1", **E.plan_trade(stock().drop(CAL[101]), CAL, T)},
            {"ticker": "B", "approach": "E3", **E.plan_trade(stock(), CAL, T)}]
    t = E.exclusion_table(recs); assert len(t) == 1 and t.iloc[0].n == 2 and t.iloc[0].reason == "entry_session_missing"
