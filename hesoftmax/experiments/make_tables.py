import argparse
import json
import os
import numpy as np

NAME = {"HETAL": "HETAL", "NEXUS": "NEXUS", "2Quad": "2Quad", "THOR": "THOR",
        "Cho24": "Cho et al.", "ACS": "\\textbf{ACS}"}
ORDER = ["HETAL", "NEXUS", "THOR", "Cho24", "ACS"]


def get(rows, m, s):
    for r in rows:
        if r.get("method") == m and abs(r.get("spread", -1) - s) < 1e-6 and "error" not in r:
            return r
    return None


def fmt(v, spec="{:.0f}"):
    return "--" if v is None else spec.format(v)


def micro_table(rows, spreads, path):
    lines = [
        r"\begin{table}[!tb]",
        r"\caption{Measured cost of one batch of $256$ softmax evaluations of dimension $128$ at "
        r"$128$-bit security, at the calibrated half-spread $S=105$ and the provable half-spread "
        r"$S=1104$.}",
        r"\label{tab:micro}", r"\centering", r"\footnotesize",
        r"\renewcommand{\arraystretch}{0.92}", r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{lrrrrrr|rrrrrr}", r"\toprule",
        r" & \multicolumn{6}{c|}{$S=%g$} & \multicolumn{6}{c}{$S=%g$}\\" % (int(spreads[0] + 0.5),
                                                                 int(spreads[1] + 0.5)),
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
        r"method & lv & mult & rot & bt & time & bits & lv & mult & rot & bt & time & bits\\",
        r"\midrule"]
    for m in ORDER:
        cells = []
        for s in spreads:
            r = get(rows, m, s)
            if r is None:
                cells += ["--"] * 6
            else:
                bits = "div" if r["bits"] < 0 else fmt(r["bits"], "{:.1f}")
                cells += [fmt(r["levels_used"]), fmt(r["mult_cc"]), fmt(r["rot"]),
                          fmt(r["boot"]), fmt(r["time_s"], "{:.0f}"), bits]
        if m == "ACS":
            cells = ["\\textbf{%s}" % c for c in cells]
        lines.append("%s & %s\\\\" % (NAME[m], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


def protocol_table(rows, spreads, path, blocks=12, cts=6):
    anchor = {"HETAL": "comparison", "NEXUS": "constant", "THOR": "interval",
              "Cho24": "assumed", "ACS": "row mean"}
    calib = {"HETAL": "no", "NEXUS": "yes", "THOR": "yes", "Cho24": "yes", "ACS": "no"}
    sound = {"HETAL": "yes", "NEXUS": "no", "THOR": "no", "Cho24": "no", "ACS": "yes"}
    f = blocks * cts
    lines = [
        r"\begin{table}[!tb]",
        r"\caption{Protocol level comparison for the attention of one GPT-2 forward pass of $128$ "
        r"tokens, at the two half-spreads of Table~\ref{tab:micro}.}",
        r"\label{tab:protocol}", r"\centering", r"\footnotesize",
        r"\renewcommand{\arraystretch}{0.92}", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrr|rrr|ccc}", r"\toprule",
        r" & \multicolumn{3}{c|}{$S=%g$} & \multicolumn{3}{c|}{$S=%g$} & & & \\" % (
            int(spreads[0] + 0.5), int(spreads[1] + 0.5)),
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"protocol & ks & bt & hours & ks & bt & hours & anchor & calib & sound\\",
        r"\midrule"]
    for m in ORDER:
        cells = []
        for s in spreads:
            r = get(rows, m, s)
            if r is None:
                cells += ["--"] * 3
            else:
                cells += [fmt(r["keyswitch"] * f), fmt(r["boot"] * f),
                          fmt(r["time_s"] * f / 3600.0, "{:.2f}")]
        lines.append("%s & %s & %s & %s & %s\\\\" % (
            NAME[m], " & ".join(cells), anchor[m], calib[m], sound[m]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


def ppl_table(acc, path):
    order = ["ACS-prov", "ACS-emp", "Cho24", "THOR", "NEXUS", "HETAL", "2Quad"]
    disp = {"ACS-prov": "\\textbf{ACS, provable $S$}", "ACS-emp": "\\textbf{ACS, measured $S$}",
            "Cho24": "Cho et al.\\ \\cite{Cho2024}", "THOR": "THOR \\cite{THOR2025}",
            "NEXUS": "NEXUS \\cite{NEXUS2025}", "HETAL": "HETAL \\cite{HETAL2023}",
            "2Quad": "2Quad \\cite{MPCFormer2023}"}
    calib = {"ACS-prov": "none", "ACS-emp": "range only", "Cho24": "per block max",
             "THOR": "per block max", "NEXUS": "per block max", "HETAL": "none",
             "2Quad": "per block max"}
    lines = [
        r"\begin{table}[!tb]",
        r"\caption{Perplexity of GPT-2 small on the test split of WikiText-2, in windows of $128$ "
        r"tokens, with the softmax of every head replaced by the method under test. Eight windows "
        r"calibrate the anchors that the methods in the lower part need and the following "
        r"twenty-four are the test set. On the test set the calibrated anchor is exceeded on "
        r"$%s$ of the attention rows, by up to $%.2f$, which is what makes the NEXUS evaluation "
        r"leave the interval on which its division converges.}" % (
            ("%.1f\\cdot10^{-5}" % (acc["violation_rate"] * 1e5)), acc["worst_overshoot"]),
        r"\label{tab:ppl}", r"\centering", r"\small",
        r"\renewcommand{\arraystretch}{0.93}",
        r"\begin{tabular}{lrrrl}", r"\toprule",
        r"method & levels & perplexity & change & calibration\\",
        r"\midrule",
        r"exact softmax & -- & %.3f & -- & --\\" % acc["exact"]["ppl"],
        r"\midrule"]
    for k in order:
        if k not in acc:
            continue
        e = acc[k]
        v = "%.3f" % e["ppl"] if e["ppl"] == e["ppl"] else "diverges"
        d = e["d_ppl"]
        if d != d:
            dv = "--"
        else:
            ex = int(np.floor(np.log10(abs(d))))
            dv = "$%+.1f\\cdot10^{%d}$" % (d / 10.0 ** ex, ex)
        lines.append("%s & %d & %s & %s & %s\\\\" % (disp[k], e["depth"], v, dv, calib[k]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="../final_paper")
    ap.add_argument("--spreads", type=float, nargs="+", default=[105.0, 1103.5])
    a = ap.parse_args()
    rows = merge_results(a.results)
    if rows:
        micro_table(rows, a.spreads, os.path.join(a.out, "tab_micro.tex"))
        protocol_table(rows, a.spreads, os.path.join(a.out, "tab_protocol.tex"))
        print("wrote micro/protocol tables")
    acc = os.path.join(a.results, "gpt2_accuracy.json")
    if os.path.exists(acc):
        ppl_table(json.load(open(acc)), os.path.join(a.out, "tab_ppl.tex"))
        print("wrote ppl table")



def merge_results(results_dir):
    rows = []
    for name in sorted(os.listdir(results_dir)):
        if name.startswith("mb_") and name.endswith(".json"):
            rows += json.load(open(os.path.join(results_dir, name)))
    if os.path.exists(os.path.join(results_dir, "microbench.json")):
        rows += json.load(open(os.path.join(results_dir, "microbench.json")))
    seen = {}
    for r in rows:
        seen[(r.get("method"), r.get("spread"))] = r
    out = list(seen.values())
    with open(os.path.join(results_dir, "microbench.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    main()
