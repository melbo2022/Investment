from __future__ import annotations
from flask import Flask, render_template, request, redirect, url_for, flash
from math import isfinite, log
import datetime as dt

app = Flask(__name__)
app.secret_key = "replace-this-key"

# ---------- Helpers ----------

def parse_float(value, default=0.0):
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        v = float(value)
        if not isfinite(v):
            raise ValueError
        return v
    except Exception:
        return default

def parse_bool_from_radio(val: str, true_value: str = "begin") -> bool:
    return (val or "") == true_value

def fv_lump_sum(pv: float, r_m: float, n: int) -> float:
    if n <= 0:
        return pv
    return pv * ((1 + r_m) ** n)

def fv_annuity(pmt: float, r_m: float, n: int, due: bool) -> float:
    if n <= 0 or pmt == 0:
        return 0.0
    if r_m == 0:
        fv = pmt * n
    else:
        fv = pmt * (((1 + r_m) ** n - 1) / r_m)
    if due:
        fv *= (1 + r_m)
    return fv

def fv_total(pv: float, pmt: float, r_m: float, n: int, due: bool) -> float:
    return fv_lump_sum(pv, r_m, n) + fv_annuity(pmt, r_m, n, due)

def bisection_solve(func, lo, hi, tol=1e-10, max_iter=200):
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    # Ensure sign change
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = func(mid)
        if abs(f_mid) < tol or (hi - lo) / 2 < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2

# ---------- Routes ----------

@app.route("/")
def home():
    return render_template("index.html")

# --- Future Value page ---
@app.route("/fv", methods=["GET", "POST"])
def page_fv():
    result = None
    if request.method == "POST":
        pv = parse_float(request.form.get("pv", "0"))
        pmt = parse_float(request.form.get("pmt", "0"))
        years = parse_float(request.form.get("years", "0"))
        annual = parse_float(request.form.get("annual", "0"))
        due = parse_bool_from_radio(request.form.get("due"), "begin")

        if years < 0 or annual < -100 or annual > 100:
            flash("入力値を確認してください（年数・利率）。", "danger")
            return redirect(url_for("page_fv"))

        n = int(round(years * 12))
        r_m = annual / 100.0 / 12.0
        fv = fv_total(pv, pmt, r_m, n, due)
        result = {"fv": round(fv, 2), "months": n}
    return render_template("fv.html", result=result)

# --- Present Value required page ---
@app.route("/pv", methods=["GET", "POST"])
def page_pv():
    result = None
    if request.method == "POST":
        target_fv = parse_float(request.form.get("target_fv", "0"))
        pmt = parse_float(request.form.get("pmt", "0"))
        years = parse_float(request.form.get("years", "0"))
        annual = parse_float(request.form.get("annual", "0"))
        due = parse_bool_from_radio(request.form.get("due"), "begin")

        if years < 0 or annual < -100 or annual > 100:
            flash("入力値を確認してください（年数・利率）。", "danger")
            return redirect(url_for("page_pv"))

        n = int(round(years * 12))
        r_m = annual / 100.0 / 12.0
        fv_pmt = fv_annuity(pmt, r_m, n, due)
        denom = (1 + r_m) ** n if n > 0 else 1.0
        pv_required = (target_fv - fv_pmt) / denom
        result = {"pv": round(pv_required, 2), "months": n}
    return render_template("pv.html", result=result)

# --- Rate solver page (annual %) ---
@app.route("/rate", methods=["GET", "POST"])
def page_rate():
    result = None
    if request.method == "POST":
        target_fv = parse_float(request.form.get("target_fv", "0"))
        pv = parse_float(request.form.get("pv", "0"))
        pmt = parse_float(request.form.get("pmt", "0"))
        years = parse_float(request.form.get("years", "0"))
        due = parse_bool_from_radio(request.form.get("due"), "begin")

        if years <= 0:
            flash("年数は正の値を指定してください。", "danger")
            return redirect(url_for("page_rate"))

        n = int(round(years * 12))

        def f(r_m):
            return fv_total(pv, pmt, r_m, n, due) - target_fv

        lo, hi = -0.999, 1.0
        r_m = bisection_solve(f, lo, hi, tol=1e-12, max_iter=300)
        if r_m is None:
            flash("解を特定できませんでした。入力値の整合性を見直してください。", "warning")
        else:
            annual_pct = (((1 + r_m) ** 12) - 1) * 100.0
            result = {"annual_rate_pct": round(annual_pct, 6), "monthly_rate_pct": round(r_m*100.0, 6), "months": n}
    return render_template("rate.html", result=result)

# --- Years solver page ---
@app.route("/years", methods=["GET", "POST"])
def page_years():
    result = None
    if request.method == "POST":
        target_fv = parse_float(request.form.get("target_fv", "0"))
        pv = parse_float(request.form.get("pv", "0"))
        pmt = parse_float(request.form.get("pmt", "0"))
        annual = parse_float(request.form.get("annual", "0"))
        due = parse_bool_from_radio(request.form.get("due"), "begin")

        if annual < -100 or annual > 100:
            flash("利率の範囲が不正です。", "danger")
            return redirect(url_for("page_years"))

        r_m = annual / 100.0 / 12.0

        def g(n_float):
            n = max(0, int(round(n_float)))
            return fv_total(pv, pmt, r_m, n, due) - target_fv

        lo, hi = 0.0, 1200.0
        if g(lo) == 0:
            n_sol = 0
        else:
            val_lo = g(lo)
            val_hi = g(hi)
            if val_lo * val_hi > 0:
                flash("目標額に到達できません。入力を見直してください。", "warning")
                return render_template("years.html", result=None)

            for _ in range(200):
                mid = (lo + hi) / 2.0
                val_mid = g(mid)
                if abs(val_mid) < 1e-6 or (hi - lo) < 1e-6:
                    break
                if val_lo * val_mid <= 0:
                    hi = mid
                    val_hi = val_mid
                else:
                    lo = mid
                    val_lo = val_mid
            n_sol = int(round((lo + hi) / 2.0))

        years_needed = n_sol / 12.0
        result = {"months": n_sol, "years": round(years_needed, 3)}
    return render_template("years.html", result=result)

# --- Monthly payment (PMT) solver page ---
@app.route("/pmt", methods=["GET", "POST"])
def page_pmt():
    result = None
    if request.method == "POST":
        target_fv = parse_float(request.form.get("target_fv", "0"))
        pv = parse_float(request.form.get("pv", "0"))
        years = parse_float(request.form.get("years", "0"))
        annual = parse_float(request.form.get("annual", "0"))
        due = parse_bool_from_radio(request.form.get("due"), "begin")

        # 入力バリデーション
        if years <= 0:
            flash("年数は正の値を指定してください。", "danger")
            return redirect(url_for("page_pmt"))
        if annual < -100 or annual > 100:
            flash("利率の範囲が不正です。", "danger")
            return redirect(url_for("page_pmt"))

        n = int(round(years * 12))
        r_m = annual / 100.0 / 12.0

        # FV = pv*(1+r)^n + pmt * [((1+r)^n - 1)/r] * (dueなら*(1+r))
        # → pmt を解く
        fv_pv = fv_lump_sum(pv, r_m, n)

        if n <= 0:
            flash("年数が短すぎます。月数が0になっています。", "danger")
            return redirect(url_for("page_pmt"))

        if r_m == 0:
            # 無利子の場合: FV = pv + pmt * n
            denom = float(n)
        else:
            denom = ((1 + r_m) ** n - 1.0) / r_m
            if due:
                denom *= (1 + r_m)

        # ゼロ除算・極端値対策
        if abs(denom) < 1e-14:
            flash("計算が不安定です。入力値の整合性を見直してください。", "warning")
            return redirect(url_for("page_pmt"))

        pmt_required = (target_fv - fv_pv) / denom
        result = {
            "pmt": round(pmt_required, 2),
            "months": n
        }

    return render_template("pmt.html", result=result)


# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
