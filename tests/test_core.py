import pandas as pd

from cubems.anomaly import detect_episodes, kpis
from cubems.folds import make_folds
from cubems.loader import parse_channel


def test_parse_channel():
    c = parse_channel("z2_AC14(kW)")
    assert (c.zone, c.kind, c.name) == (2, "AC", "AC14")
    assert parse_channel("z1_S1(RH%)").kind == "rh"
    assert parse_channel("Date") is None


def test_folds_stay_within_season():
    for f in make_folds():
        assert f.test not in f.train
        assert len(f.train) >= 2


def test_episode_needs_consecutive_hours():
    ts = pd.date_range("2019-09-02 08:00", periods=6, freq="h")
    s = pd.DataFrame({"floor": 2, "zone": 1, "timestamp": ts,
                      "kw": [5, 20, 20, 5, 20, 5], "lo": 0.0, "hi": 10.0, "coverage": 1.0})
    scored, ep = detect_episodes(s, margin_kw=0.25, min_hours=2)
    assert len(ep) == 1 and ep.iloc[0]["duration_h"] == 2
    k = kpis(scored, ep)
    assert k.iloc[0]["duration_burden"] == 100 * 2 / 6
