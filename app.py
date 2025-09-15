from __future__ import annotations
from flask import Flask, render_template, request, redirect, url_for, flash
from math import isfinite, log, log1p, exp

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

# ---------- 共通：グラフ用ヘルパ ----------

def downsample_yearly(series):
    """年次（12ヶ月ごと）に間引き。index=0（初期値）を含む。端数月があれば末尾も追加。"""
    out = [series[i] for i in range(0, len(series), 12)]
    if (len(series) - 1) % 12 != 0:
        out.append(series[-1])
    return out

def build_balance_series_savings(pv0: float, pmt: float, r_m: float, n: int, due_begin: bool):
    """
    貯蓄の月次残高系列を返す。index=0 が初期残高（月0）。
    期末拠出（end） : S_{k+1} = S_k*(1+r) + PMT
    期首拠出（begin）: S_{k+1} = (S_k + PMT)*(1+r)
    """
    series = [max(0.0, pv0)]
    S = pv0
    for _ in range(n):
        if due_begin:
            S = (S + pmt) * (1.0 + r_m)
        else:
            S = S * (1.0 + r_m) + pmt
        series.append(S)
    return series

def build_balance_series_loan(L0: float, PMT: float, r_m: float, n: int):
    """
    ローンの月次残高系列（元利均等・期末払い）。
    期末払い：S_{k+1} = S_k*(1+r) - PMT
    """
    series = [max(0.0, L0)]
    S = L0
    for _ in range(n):
        S = S * (1.0 + r_m) - PMT
        series.append(S)
    return series

# ---------- Routes ----------

@app.route("/")
def home():
    return render_template("index.html")

# --- Future Value page ---
# --- Savings Solver page (solve exactly one of: FV, PV, PMT, years, rate) ---
@app.route("/savings", methods=["GET", "POST"])
def page_savings():
    """
    積立（毎月拠出・月複利）で、FV / PV / PMT / 年数 / 金利 のうち1つを解く。
    due: 'begin'（期首＝毎月初） or 'end'（期末＝毎月末）

    既存ヘルパ:
      fv_lump_sum(pv, r_m, n)
      fv_annuity(pmt, r_m, n, due)
      fv_total(pv, pmt, r_m, n, due) = 上の合算
      bisection_solve(func, lo, hi, tol, max_iter)
    """

    result = None
    # グラフ用効果的パラメータを入れる箱（POST時に埋める）
    PV_eff = PMT_eff = r_eff = None
    n_eff = 0
    due_eff = False

    if request.method == "POST":
        solve   = (request.form.get("solve") or "fv").strip()

        target_fv = parse_float(request.form.get("target_fv", "0"))
        pv        = parse_float(request.form.get("pv", "0"))
        pmt       = parse_float(request.form.get("pmt", "0"))
        years     = parse_float(request.form.get("years", "0"))
        annual    = parse_float(request.form.get("annual", "0"))
        due       = (request.form.get("due") or "begin") == "begin"

        # バリデーション（軽め）
        if annual < -100 or annual > 100:
            flash("年率の範囲が不正です。", "danger")
            return render_template("savings.html", result=None)

        # 便利変換
        n   = int(round(years * 12)) if years > 0 else 0
        r_m = annual / 100.0 / 12.0

        # ユーティリティ：積立FVの年金係数（将来価値）
        def af_fv(r, n, due_flag):
            if n <= 0:
                return 0.0
            if r == 0.0:
                af = float(n)
            else:
                af = ((1 + r) ** n - 1.0) / r
            if due_flag and r != 0.0:
                af *= (1 + r)
            return af

        # ---------- 解く対象ごと ----------
        if solve == "fv":
            # 将来価値 FV を解く（最も単純）
            if years < 0:
                flash("年数は0以上で指定してください。", "danger")
                return render_template("savings.html", result=None)
            n = int(round(max(0.0, years) * 12))
            fv = fv_total(pv, pmt, r_m, n, due)
            result = {"solve": solve, "fv": round(fv, 2), "months": n}

            # グラフ用
            PV_eff, PMT_eff, r_eff, n_eff, due_eff = pv, pmt, r_m, n, due

        elif solve == "pv":
            # 目標FVに必要な現在元本PV
            if years < 0:
                flash("年数は0以上で指定してください。", "danger")
                return render_template("savings.html", result=None)
            n   = int(round(max(0.0, years) * 12))
            fvP = fv_annuity(pmt, r_m, n, due)
            denom = (1 + r_m) ** n if n > 0 else 1.0
            pv_req = (target_fv - fvP) / denom
            result = {"solve": solve, "pv": round(pv_req, 2), "months": n}

            # グラフ用（解いた pv_req を初期元本にして系列描画）
            PV_eff, PMT_eff, r_eff, n_eff, due_eff = pv_req, pmt, r_m, n, due

        elif solve == "pmt":
            # 目標FVに必要な毎月積立額PMT
            if n <= 0:
                flash("年数は正の値で指定してください。", "danger")
                return render_template("savings.html", result=None)

            if r_m == 0.0:
                denom = float(n)
            else:
                denom = af_fv(r_m, n, due)

            if abs(denom) < 1e-15:
                flash("計算が不安定です。入力値を見直してください。", "warning")
                return render_template("savings.html", result=None)

            fv_pv = fv_lump_sum(pv, r_m, n)
            pmt_req = (target_fv - fv_pv) / denom
            result = {"solve": solve, "pmt": round(pmt_req, 2), "months": n}

            # グラフ用（解いた pmt_req を毎月拠出にして系列描画）
            PV_eff, PMT_eff, r_eff, n_eff, due_eff = pv, pmt_req, r_m, n, due

        elif solve == "years":
            # 目標FVに到達するまでの年数
            if annual < -100 or annual > 100:
                flash("年率の範囲が不正です。", "danger")
                return render_template("savings.html", result=None)

            def g(n_float):
                n_ = max(0, int(round(n_float)))
                return fv_total(pv, pmt, r_m, n_, due) - target_fv

            lo, hi = 0.0, 1200.0  # 最大100年
            val_lo, val_hi = g(lo), g(hi)
            if val_lo == 0:
                n_sol = 0
            elif val_lo * val_hi > 0:
                flash("目標額に到達できません。入力を見直してください。", "warning")
                return render_template("savings.html", result=None)
            else:
                # 2分法
                for _ in range(200):
                    mid = (lo + hi) / 2.0
                    vm  = g(mid)
                    if abs(vm) < 1e-6 or (hi - lo) < 1e-6:
                        break
                    if val_lo * vm <= 0:
                        hi, val_hi = mid, vm
                    else:
                        lo, val_lo = mid, vm
                n_sol = int(round((lo + hi) / 2.0))

            years_needed = n_sol / 12.0
            result = {"solve": solve, "months": n_sol, "years": round(years_needed, 3)}

            # グラフ用（解いた n_sol を使用）
            PV_eff, PMT_eff, r_eff, n_eff, due_eff = pv, pmt, r_m, n_sol, due

        elif solve == "rate":
            # 必要な年率（%）を解く（r_m を数値解 → 年率へ変換）
            if n <= 0:
                flash("年数は正の値で指定してください。", "danger")
                return render_template("savings.html", result=None)

            def f(rm):
                return fv_total(pv, pmt, rm, n, due) - target_fv

            lo, hi = -0.95/12.0, 1.0/12.0  # 月利の探索範囲（年率で約 -95%〜+100%）
            r_m_sol = bisection_solve(f, lo, hi, tol=1e-12, max_iter=300)
            if r_m_sol is None:
                flash("解を特定できませんでした。入力値の整合性を見直してください。", "warning")
            else:
                annual_pct = (((1 + r_m_sol) ** 12) - 1) * 100.0
                result = {
                    "solve": solve,
                    "annual_rate_pct": round(annual_pct, 6),
                    "monthly_rate_pct": round(r_m_sol * 100.0, 6),
                    "months": n,
                }
                # グラフ用（解いた r_m_sol を使用）
                PV_eff, PMT_eff, r_eff, n_eff, due_eff = pv, pmt, r_m_sol, n, due

        else:
            flash("解く対象（どれを求めるか）を選択してください。", "danger")

        # ---- 共通：グラフ用データを作成して result に添付 ----
        if result is not None:
            try:
                if n_eff > 0:
                    series_m = build_balance_series_savings(PV_eff or 0.0, PMT_eff or 0.0, r_eff or 0.0, n_eff, due_eff or False)
                    series_y = downsample_yearly(series_m)
                    labels_y = [f"{i}年" for i in range(len(series_y))]
                    result["chart_labels"] = labels_y
                    result["chart_data"]   = [round(x, 2) for x in series_y]
                else:
                    result["chart_labels"] = []
                    result["chart_data"]   = []
            except Exception:
                result["chart_labels"] = []
                result["chart_data"]   = []

    return render_template("savings.html", result=result)

#---------------------------------------------------------------------------------------------------------------------

# --- Loan Solver page (solve one of: amount, years, payment, rate) ---
@app.route("/loan", methods=["GET", "POST"])
def page_loan():
    """
    元利均等返済（毎月末払い・固定金利）モデル。
    4つのうち1つを解く：
      - 借入金額 L
      - 返済年数 years
      - 月額返済額 PMT
      - 金利（年率, %）
    最終残存元本（バルーン）Bを考慮する。
    記号：r_m = 月利、n = 月数、B = 最終残存元本（> = 0）
    主な関係式（期末払い）：
      残高 B = L*(1+r)^n - PMT * ((1+r)^n - 1)/r
      ⇔ PMT = (L - B*(1+r)^(-n)) * r / (1 - (1+r)^(-n))   （r ≠ 0）
         r=0 のとき PMT = (L - B)/n
    """
    result = None

    # グラフ用の効果的パラメータ（POSTで埋める）
    L_eff = PMT_eff = r_eff = None
    n_eff = 0

    if request.method == "POST":
        solve   = (request.form.get("solve") or "payment").strip()

        L       = parse_float(request.form.get("loan_amount", "0"))       # 借入金額
        years   = parse_float(request.form.get("years", "0"))             # 返済年数
        PMT     = parse_float(request.form.get("monthly_payment", "0"))   # 月額返済額
        annual  = parse_float(request.form.get("annual", "0"))            # 年率(%)
        B       = parse_float(request.form.get("residual", "0"))          # ★ 最終残存元本（バルーン）

        # 入力バリデーション（軽め）
        if annual < -100 or annual > 100:
            flash("金利（年率）の範囲が不正です。", "danger")
            return redirect(url_for("page_loan"))
        if years < 0:
            flash("返済年数は0以上を指定してください。", "danger")
            return redirect(url_for("page_loan"))
        if B < 0:
            flash("最終残存元本（バルーン）は0以上で入力してください。", "danger")
            return redirect(url_for("page_loan"))

        # 月数・月利
        n   = int(round(years * 12)) if years > 0 else 0
        r_m = annual / 100.0 / 12.0

        # --- 数値安定な PMT 計算（B=残存元本対応） ---
        def pmt_from(L_, r_, n_, B_):
            if n_ <= 0 or L_ <= 0:
                return None
            if r_ == 0.0:
                # 無利子：PMT = (L - B)/n。L < B だと返済不能（負のPMT）になる
                return (L_ - B_) / n_
            if 1.0 + r_ <= 0.0:
                return None  # log1pの定義域外

            from math import log1p, exp
            t = -n_ * log1p(r_)               # (1+r)^(-n) = exp(t)
            if t > 700.0:                     # expのオーバーフロー回避
                return None
            inv = exp(t)                      # (1+r)^(-n)
            denom = 1.0 - inv
            if abs(denom) < 1e-15:
                return None
            return ((L_ - B_ * inv) * r_) / denom

        # --- 各ケース ---
        if solve == "payment":
            if n <= 0 or L <= 0:
                flash("借入金額と返済年数は正の値を指定してください。", "danger")
                return redirect(url_for("page_loan"))
            p = pmt_from(L, r_m, n, B)
            if p is None or (r_m == 0.0 and L < B):
                flash("計算が不安定か、条件が不成立です（残存元本が大きすぎる等）。", "warning")
            else:
                result = {"solve": solve, "monthly_payment": round(p, 2), "n": n}
                # グラフ用
                L_eff, PMT_eff, r_eff, n_eff = L, p, r_m, n

        elif solve == "amount":
            if n <= 0 or PMT <= 0:
                flash("月額返済額と返済年数は正の値を指定してください。", "danger")
                return redirect(url_for("page_loan"))
            if r_m == 0.0:
                # L = PMT*n + B
                L_req = PMT * n + B
            else:
                if 1.0 + r_m <= 0.0:
                    flash("金利が不正です。", "danger")
                    return redirect(url_for("page_loan"))
                from math import log1p, exp
                t = -n * log1p(r_m)
                if t > 700.0:
                    flash("計算が不安定です。入力値を見直してください。", "warning")
                    return redirect(url_for("page_loan"))
                inv = exp(t)  # (1+r)^(-n)
                # L = B*inv + PMT * (1 - inv)/r
                L_req = B * inv + PMT * (1.0 - inv) / r_m
            result = {"solve": solve, "loan_amount": round(L_req, 2), "n": n}
            # グラフ用（解いた L_req で系列描画）
            L_eff, PMT_eff, r_eff, n_eff = L_req, PMT, r_m, n

        #-----------------------------------------------------------------------------------------------------------
        elif solve == "years":
            if L <= 0 or PMT <= 0:
                flash("借入金額と月額返済額は正の値を指定してください。", "danger")
                return redirect(url_for("page_loan"))

            if r_m == 0.0:
                # B = L - PMT*n  ⇒  n = (L - B)/PMT
                if L <= B:
                    flash("無利子では残存元本が大きすぎます（返済が成立しません）。", "warning")
                    return redirect(url_for("page_loan"))
                n_real = (L - B) / PMT
            else:
                if 1.0 + r_m <= 0.0:
                    flash("金利が不正です。", "danger")
                    return redirect(url_for("page_loan"))

                from math import log, log1p, isfinite

                # (1+i)^n = (B - PMT/i) / (L - PMT/i)
                i = r_m
                denom = (L - PMT / i)   # 分母が負でも比が正なら解は成立
                if abs(denom) < 1e-15:
                    flash("返済条件が成立しません（PMTが金利相当と一致）。", "warning")
                    return redirect(url_for("page_loan"))

                rhs = (B - PMT / i) / denom  # 右辺 = (1+i)^n
                if rhs <= 0.0:
                    flash("その条件では返済年数の解が見つかりません。パラメータを見直してください。", "warning")
                    return redirect(url_for("page_loan"))

                n_real = log(rhs) / log1p(i)   # n = ln(rhs) / ln(1+i)
                if (not isfinite(n_real)) or n_real < 0:
                    flash("計算が不安定です。入力値を見直してください。", "warning")
                    return redirect(url_for("page_loan"))

            n_req = int(round(n_real))
            result = {"solve": solve, "months": n_req, "years": round(n_req / 12.0, 3)}

            # グラフ用（解いた n_req で系列描画）
            L_eff, PMT_eff, r_eff, n_eff = L, PMT, r_m, n_req
        #-------------------------------------------------------------------------------------------------------------------
        elif solve == "rate":
            if L <= 0 or PMT <= 0 or n <= 0:
                flash("借入金額・返済年数・月額返済額は正の値を指定してください。", "danger")
                return redirect(url_for("page_loan"))

            # r=0 のときの必要PMT（基準）
            pmt_r0 = (L - B) / n
            if r_m == 0.0 and abs(PMT - pmt_r0) < 1e-12:
                # 既に年率0%の解
                result = {"solve": solve, "monthly_rate_pct": 0.0, "annual_rate_pct": 0.0}
                # グラフ用
                L_eff, PMT_eff, r_eff, n_eff = L, PMT, 0.0, n
                return render_template("loan.html", result=result)

            def safe_f(r):
                v = pmt_from(L, r, n, B)
                if v is None or not isfinite(v):
                    return None
                return v - PMT

            # 0付近から両側にブラケット探索（オーバーフローを避けた格子点）
            grid = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
            vals = []
            for r in grid:
                fval = safe_f(r)
                if fval is not None:
                    vals.append((r, fval))

            # まずゼロヒットを確認
            r_sol = None
            for r, fval in vals:
                if abs(fval) < 1e-12:
                    r_sol = r
                    break

            # 隣接点での符号変化を探す
            if r_sol is None:
                bracket = None
                for i in range(len(vals) - 1):
                    r1, f1 = vals[i]
                    r2, f2 = vals[i + 1]
                    if f1 * f2 <= 0:
                        bracket = (r1, r2)
                        break

                if bracket is None:
                    flash("与えられた条件では金利の解が見つかりません。（月額・年数・残存元本の組合せを見直してください）", "warning")
                    return render_template("loan.html", result=None)

                lo, hi = bracket
                r_sol = bisection_solve(lambda x: safe_f(x), lo, hi, tol=1e-12, max_iter=300)

            if r_sol is None:
                flash("解を特定できませんでした。入力値の整合性を見直してください。", "warning")
            else:
                annual_pct = (((1.0 + r_sol) ** 12) - 1.0) * 100.0
                result = {
                    "solve": solve,
                    "monthly_rate_pct": round(r_sol * 100.0, 6),
                    "annual_rate_pct": round(annual_pct, 6),
                }
                # グラフ用（解いた r_sol）
                L_eff, PMT_eff, r_eff, n_eff = L, PMT, r_sol, n
        else:
            flash("解く対象（どれを求めるか）を選択してください。", "danger")

        # ---- 共通：グラフ用データを作成して result に添付 ----
        if result is not None:
            try:
                if n_eff > 0 and L_eff is not None and PMT_eff is not None and r_eff is not None:
                    series_m = build_balance_series_loan(L_eff, PMT_eff, r_eff, n_eff)
                    series_y = downsample_yearly(series_m)
                    labels_y = [f"{i}年" for i in range(len(series_y))]
                    result["chart_labels"] = labels_y
                    # ローン残高はマイナス方向に振れることがあるため、見栄え上0未満は0に丸める
                    result["chart_data"]   = [round(x if x >= 0 else 0.0, 2) for x in series_y]
                else:
                    result["chart_labels"] = []
                    result["chart_data"]   = []
            except Exception:
                result["chart_labels"] = []
                result["chart_data"]   = []

    return render_template("loan.html", result=result)

#---------------------------------------------------------------------------------------------------------------------------
# --- Drawdown (annuity decumulation) Solver page ---
# --- Drawdown (annuity decumulation) Solver page ---
@app.route("/drawdown", methods=["GET", "POST"])
def page_drawdown():
    """
    現在の貯蓄（PV）を運用しながら毎月取り崩し（WD）たとき、
    5つ（PV / WD / 残存額B / 年率annual / 取崩年数years）のうち1つを求める。
    支払タイミング: 'end'（月末/後払い） or 'begin'（月初/先払い）

    基本式（期末払い）:
      B = PV*(1+r)^n - WD * AF(r,n,end)
      AF(r,n,end)   = ((1+r)^n - 1)/r
      AF(r,n,begin) = AF(r,n,end) * (1+r)
      r=0 のとき AF = n
    """
    from math import isfinite, log1p, exp, log

    result = None

    # ---- ヘルパ ----
    def annuity_factor(r_, n_, due_begin_):
        """AF(r,n,due): 将来価値換算の年金係数。None は計算不能域。"""
        if n_ <= 0:
            return 0.0
        if r_ == 0.0:
            af = float(n_)
        else:
            if 1.0 + r_ <= 0.0:
                return None
            t = n_ * log1p(r_)          # log((1+r)^n)
            if t > 700.0:
                return None
            af = (exp(t) - 1.0) / r_
        if due_begin_ and r_ != 0.0:
            af *= (1.0 + r_)
        return af

    def pow1pr_n(r_, n_):
        """(1+r)^n を数値安定に計算。"""
        if n_ == 0:
            return 1.0
        if 1.0 + r_ <= 0.0:
            return None
        t = n_ * log1p(r_)
        if t > 700.0:
            return None
        return exp(t)

    def build_balance_series(PV0, WD0, r_, n_, due_begin_):
        """
        月次残高の配列を返す。index=0 が初期残高（月0）。
        end（後払い） : S_{k+1} = S_k*(1+r) - WD
        begin（先払い）: S_{k+1} = (S_k - WD)*(1+r)
        """
        series = [max(0.0, PV0)]
        S = PV0
        for _ in range(n_):
            if not due_begin_:
                S = S * (1.0 + r_) - WD0
            else:
                S = (S - WD0) * (1.0 + r_)
            series.append(S)
        return series

    def downsample_yearly(series):
        """年次（12ヶ月ごと）に間引き。0年目含む。端数月があれば末尾も追加。"""
        out = [series[i] for i in range(0, len(series), 12)]
        if (len(series) - 1) % 12 != 0:
            out.append(series[-1])
        return out

    if request.method == "POST":
        solve    = (request.form.get("solve") or "withdrawal").strip()

        PV       = parse_float(request.form.get("pv", "0"))            # 初期貯蓄
        WD       = parse_float(request.form.get("withdrawal", "0"))    # 取崩月額
        years    = parse_float(request.form.get("years", "0"))         # 取崩年数
        annual   = parse_float(request.form.get("annual", "0"))        # 年率(%)
        B        = parse_float(request.form.get("residual", "0"))      # 最終残存金額
        due_str  = (request.form.get("due") or "end").strip()
        due_begin = (due_str == "begin")

        # 軽いバリデーション（エラーでも redirect せず再描画 → 入力保持）
        if annual < -100 or annual > 100:
            flash("利回り（年率）の範囲が不正です。", "danger")
            return render_template("drawdown.html", result=None)
        if years < 0:
            flash("取崩年数は0以上で入力してください。", "danger")
            return render_template("drawdown.html", result=None)
        if B < 0:
            flash("残存金額は0以上で入力してください。", "danger")
            return render_template("drawdown.html", result=None)

        n = int(round(years * 12)) if years > 0 else 0
        r = annual / 100.0 / 12.0  # 月利

        # ---- 各ケース ----
        if solve == "withdrawal":
            if n <= 0:
                flash("取崩年数を正の値で入力してください。", "danger")
                return render_template("drawdown.html", result=None)
            X  = pow1pr_n(r, n)
            AF = annuity_factor(r, n, due_begin)
            if X is None or AF in (None, 0.0):
                flash("計算が不安定です。入力値を見直してください。", "warning")
                return render_template("drawdown.html", result=None)
            WD_req = (PV - B / X) / (AF / X) if r != 0.0 else (PV - B) / n
            result = {"solve": solve, "withdrawal": round(WD_req, 2), "n": n}

            PV_eff, WD_eff, n_eff, r_eff, due_eff = PV, WD_req, n, r, due_begin

        elif solve == "pv":
            if n <= 0:
                flash("取崩年数を正の値で入力してください。", "danger")
                return render_template("drawdown.html", result=None)
            X  = pow1pr_n(r, n)
            AF = annuity_factor(r, n, due_begin)
            if X is None or AF is None:
                flash("計算が不安定です。入力値を見直してください。", "warning")
                return render_template("drawdown.html", result=None)
            PV_req = (B + WD * AF) / X if r != 0.0 else (B + WD * n)
            result = {"solve": solve, "pv": round(PV_req, 2), "n": n}

            PV_eff, WD_eff, n_eff, r_eff, due_eff = PV_req, WD, n, r, due_begin

        elif solve == "residual":
            if n <= 0:
                flash("取崩年数を正の値で入力してください。", "danger")
                return render_template("drawdown.html", result=None)
            X  = pow1pr_n(r, n)
            AF = annuity_factor(r, n, due_begin)
            if X is None or AF is None:
                flash("計算が不安定です。入力値を見直してください。", "warning")
                return render_template("drawdown.html", result=None)
            B_req = PV * X - WD * AF if r != 0.0 else (PV - WD * n)
            result = {"solve": solve, "residual": round(B_req, 2), "n": n}

            PV_eff, WD_eff, n_eff, r_eff, due_eff = PV, WD, n, r, due_begin
        #----------------------------------------------------------------------------------------------------------------
        elif solve == "years":
            # 取崩して B に到達するまでの「月数 n」を解く
            # モデル：期末払い(annuity-immediate)、期首払いのときは due_begin=True
            # 一般式（r>0）：
            #   B = (PV - A) * (1+r)^n + A
            # ここで A = WD * ((1+r) if due_begin else 1) / r
            # したがって (1+r)^n = (B - A) / (PV - A) で、rhs>0 が解の必要十分条件（分母の符号は不問）

            if WD <= 0:
                flash("取崩月額は正の値で入力してください。", "danger")
                return render_template("drawdown.html", result=None)

            from math import log, log1p, isfinite

            if r == 0.0:
                # 金利ゼロ：PV - WD*n = B  →  n = (PV - B)/WD
                n_real = (PV - B) / WD
                if n_real < 0 or not isfinite(n_real):
                    flash("その条件では到達できません。", "warning")
                    return render_template("drawdown.html", result=None)
                n_req = max(0, int(round(n_real)))
            else:
                if 1.0 + r <= 0.0:
                    flash("金利が不正です。", "danger")
                    return render_template("drawdown.html", result=None)

                # A: 年金現価係数の転置項（期首払い対応）
                A = WD * ((1.0 + r) if due_begin else 1.0) / r

                denom = (PV - A)               # 分母が負でもOK（比が正なら可）
                if abs(denom) < 1e-15:
                    flash("条件が特異です（PMT が金利相当と一致）。", "warning")
                    return render_template("drawdown.html", result=None)

                rhs = (B - A) / denom          # = (1+r)^n
                if rhs <= 0.0 or not isfinite(rhs):
                    flash("その条件では到達できません（パラメータを見直してください）。", "warning")
                    return render_template("drawdown.html", result=None)

                n_real = log(rhs) / log1p(r)   # n = ln(rhs)/ln(1+r)
                if n_real < 0 or not isfinite(n_real):
                    flash("計算が不安定です。入力値を見直してください。", "warning")
                    return render_template("drawdown.html", result=None)

                n_req = max(0, int(round(n_real)))

            result = {"solve": solve, "months": n_req, "years": round(n_req / 12.0, 3)}

            # グラフ用（解いた n_req で系列描画）
            PV_eff, WD_eff, n_eff, r_eff, due_eff = PV, WD, n_req, r, due_begin

        #-----------------------------------------------------------------------------------------------------------------
        elif solve == "rate":
            if WD <= 0 or n <= 0:
                flash("取崩月額と取崩年数は正の値を入力してください。", "danger")
                return render_template("drawdown.html", result=None)

            WD_r0 = (PV - B) / n
            if abs(WD - WD_r0) < 1e-12:
                result = {"solve": solve, "monthly_rate_pct": 0.0, "annual_rate_pct": 0.0, "n": n}
                PV_eff, WD_eff, n_eff, r_eff, due_eff = PV, WD, n, 0.0, due_begin
                # ↓ グラフ生成へ
            else:
                def WD_from(PV_, r_, n_, B_, due_begin_):
                    X  = pow1pr_n(r_, n_)
                    AF = annuity_factor(r_, n_, due_begin_)
                    if X is None or AF in (None, 0.0):
                        return None
                    return (PV_ * X - B_) / AF if r_ != 0.0 else (PV_ - B_) / n_

                def f(r_):
                    v = WD_from(PV, r_, n, B, due_begin)
                    if v is None or not isfinite(v):
                        return None
                    return v - WD

                # 0 近傍から安全にブラケット探索
                grid = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
                vals = []
                for r_try in grid:
                    fv = f(r_try)
                    if fv is not None:
                        vals.append((r_try, fv))

                r_sol = None
                for r_try, fv in vals:
                    if abs(fv) < 1e-12:
                        r_sol = r_try
                        break
                bracket = None
                if r_sol is None:
                    for i in range(len(vals) - 1):
                        r1, f1 = vals[i]
                        r2, f2 = vals[i + 1]
                        if f1 * f2 <= 0:
                            bracket = (r1, r2)
                            break

                if r_sol is None and bracket is None:
                    flash("与えられた条件では利回りの解が見つかりません。", "warning")
                    return render_template("drawdown.html", result=None)

                if r_sol is None:
                    lo, hi = bracket
                    r_sol = bisection_solve(lambda x: f(x), lo, hi, tol=1e-12, max_iter=300)

                if r_sol is None:
                    flash("解を特定できませんでした。入力値の整合性を見直してください。", "warning")
                    return render_template("drawdown.html", result=None)

                annual_pct = (((1.0 + r_sol) ** 12) - 1.0) * 100.0
                result = {
                    "solve": solve,
                    "monthly_rate_pct": round(r_sol * 100.0, 6),
                    "annual_rate_pct": round(annual_pct, 6),
                    "n": n,
                }
                PV_eff, WD_eff, n_eff, r_eff, due_eff = PV, WD, n, r_sol, due_begin

        else:
            flash("解く対象（どれを求めるか）を選択してください。", "danger")
            return render_template("drawdown.html", result=None)

        # ---- 共通：グラフ用データを作成して result に添付 ----
        try:
            series_m = build_balance_series(PV_eff, WD_eff, r_eff, n_eff, due_eff)
            series_y = downsample_yearly(series_m)
            labels_y = [f"{i}年" for i in range(len(series_y))]
            # 末尾ラベルを「最終」にする場合は以下に変更：
            # if (len(series_m) - 1) % 12 != 0 and len(series_y) >= 2:
            #     labels_y[-1] = "最終"
            result["chart_labels"] = labels_y
            result["chart_data"]   = [round(x, 2) for x in series_y]
        except Exception:
            # 作図失敗は致命的でないので、グラフは無視して結果のみ表示
            result["chart_labels"] = []
            result["chart_data"]   = []

    return render_template("drawdown.html", result=result)



# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
