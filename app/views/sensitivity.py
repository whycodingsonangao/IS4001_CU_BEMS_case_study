import pandas as pd
import streamlit as st
from common import card, load_qa, load_qb, load_qb_decision, load_zone_day, need, page_header

page_header("Sensitivity: strict 60/60-minute rule",
            "Every result re-run with an hour valid only if all 60 minutes are recorded, against the main 57/60 rule.",
            eyebrow="Data · robustness")

zd = load_zone_day()
y = zd[zd["year"] == 2019]
cov = y.groupby("group").agg(days=("usable", "size"), main=("usable", "sum"), strict=("usable_strict", "sum"))
with card("Complete zone-days", f"Accepting hours missing 1–3 minutes raises complete C1 zone-days from "
          f"<b>{cov.loc['C1', 'strict']:,}</b> to <b>{cov.loc['C1', 'main']:,}</b> of {cov.loc['C1', 'days']:,}."):
    cov["main_pct"] = 100 * cov["main"] / cov["days"]
    cov["strict_pct"] = 100 * cov["strict"] / cov["days"]
    st.dataframe(cov.reset_index(), hide_index=True, width="stretch", column_config={
        "group": "Group", "days": "Zone-days", "main": "Complete (57/60)", "strict": "Complete (60/60)",
        "main_pct": st.column_config.ProgressColumn("57/60", format="%.0f%%", min_value=0, max_value=100),
        "strict_pct": st.column_config.ProgressColumn("60/60", format="%.0f%%", min_value=0, max_value=100)})

ka, ks_ = need(load_qa("kselect"), "run_question_a.py"), load_qa("kselect", strict=True)
rows = []
for g in ("C1", "C2"):
    for label, t in (("57/60", ka), ("60/60", ks_)):
        if t is None:
            continue
        ch = t[(t["group"] == g) & t["chosen"]].iloc[0]
        rows.append({"group": g, "rule": label, "k": int(ch["k"]), "silhouette": ch["silhouette"],
                     "half_year_ari": ch["half_year_ari"], "ward_ari": ch["ward_ari"], "reason": ch["reason"]})
qa = pd.DataFrame(rows)
same = qa.groupby("group")["k"].nunique().eq(1)
msg = " · ".join(f"<b>{g}</b>: {'same k' if ok else 'k changes ' + ' → '.join(qa[qa.group == g]['k'].astype(str))}"
                 for g, ok in same.items())
with card("Question A: does the clustering change?", msg, "Same k and similar quality under both rules means the "
          "patterns are not an artefact of accepting 57-minute hours; a changed k needs a sentence in the report."):
    st.dataframe(qa, hide_index=True, width="stretch", column_config={
        "silhouette": st.column_config.NumberColumn(format="%.2f"),
        "half_year_ari": st.column_config.NumberColumn("Half-year ARI", format="%.2f"),
        "ward_ari": st.column_config.NumberColumn("Ward ARI", format="%.2f")})

rows = []
for label, strict in (("57/60", False), ("60/60", True)):
    d, m = load_qb_decision(strict), load_qb("metrics", strict)
    if d is None:
        continue
    dev = m[m["subset"] == "development"].set_index("model")["mae"]
    fin = m[(m["subset"] == "final (all floors)")].set_index("model")
    rows.append({"rule": label, "chosen": d["chosen"], "dev_boosting": dev["boosting"], "dev_ridge": dev["ridge"],
                 "dev_benchmark": dev["benchmark"], "final_mae": fin.loc[d["chosen"], "mae"],
                 "final_skill": fin.loc[d["chosen"], "skill_vs_benchmark"],
                 "exception_days": d["exception_days"]})
qb = pd.DataFrame(rows)
if len(qb):
    beats = (qb["dev_boosting"] < qb["dev_benchmark"]).tolist()
    with card("Question B: does the model decision change?",
              f"{'The same model is' if qb.chosen.nunique() == 1 else 'Different models are'} chosen under "
              "the two completeness rules. Development boosting is "
              + ("ahead of" if all(beats) else "not consistently ahead of")
              + " the seven-day daily benchmark.", None):
        st.dataframe(qb, hide_index=True, width="stretch", column_config={
            "dev_boosting": st.column_config.NumberColumn("Dev MAE boosting", format="%.3f"),
            "dev_ridge": st.column_config.NumberColumn("Dev MAE ridge", format="%.3f"),
            "dev_benchmark": st.column_config.NumberColumn("Dev MAE benchmark", format="%.3f"),
            "final_mae": st.column_config.NumberColumn("Final MAE", format="%.3f"),
            "final_skill": st.column_config.NumberColumn("Final skill", format="percent"),
            "exception_days": st.column_config.NumberColumn("Exception zone-days")})
