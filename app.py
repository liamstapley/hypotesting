import re
import numpy as np
import streamlit as st
from scipy import stats

# =============================
# Page config + styling
# =============================
st.set_page_config(page_title="StatCheck", page_icon="📊", layout="centered")

st.markdown(
    """
    <style>
      /* Page background */
      .stApp { background: #F7F9FC; }

      /* Make room for fixed header */
      .block-container {
        padding-top: 5.25rem;
        padding-bottom: 2.5rem;
        max-width: 920px;
      }

      /* Hide Streamlit chrome */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* Top navbar */
      .topbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 64px;
        display: flex;
        align-items: center;
        padding: 0 22px;
        background: rgba(255,255,255,0.92);
        border-bottom: 1px solid rgba(20,20,20,0.08);
        backdrop-filter: blur(10px);
        z-index: 1000;
      }
      .topbar-title {
        font-weight: 750;
        font-size: 1.05rem;
        letter-spacing: -0.01em;
        color: rgba(20,20,20,0.9);
      }
      .topbar-subtitle {
        margin-left: 10px;
        font-size: 0.92rem;
        color: rgba(20,20,20,0.55);
      }

      /* Section card */
      .card {
        background: #FFFFFF;
        border: 1px solid rgba(20,20,20,0.08);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(20, 20, 20, 0.05);
        margin-bottom: 14px;
      }

      /* Section title */
      .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: rgba(20,20,20,0.78);
        margin-bottom: 0.65rem;
      }

      /* Hint text */
      .hint {
        margin-top: -0.35rem;
        margin-bottom: 0.65rem;
        color: rgba(20,20,20,0.55);
        font-size: 0.88rem;
      }

      /* Badge */
      .badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 650;
        font-size: 0.9rem;
        border: 1px solid rgba(20,20,20,0.12);
        background: rgba(255,255,255,0.85);
      }

      /* Slightly tighten katex spacing */
      .katex-display { margin: 0.35rem 0 !important; }

      /* Widget labels */
      label,
      .stNumberInput label,
      .stSelectbox label,
      .stTextArea label {
        color: rgba(20, 20, 20, 0.85) !important;
        font-weight: 600;
      }

        /* Help / hint text */
        .stMarkdown,
        .stCaption,
        .stTooltipContent {
        color: rgba(20, 20, 20, 0.65) !important;
        }

        /* Tabs: unselected */
        button[data-baseweb="tab"] {
        color: rgba(20, 20, 20, 0.65) !important;
        font-weight: 600;
        }

        /* Tabs: selected */
        button[data-baseweb="tab"][aria-selected="true"] {
        color: rgba(20, 20, 20, 0.95) !important;
        border-bottom: 3px solid #ff4b4b !important;
        }

        /* Section titles */
        .section-title {
        color: rgba(20, 20, 20, 0.85) !important;
        }
    </style>

    <div class="topbar">
      <div class="topbar-title">StatCheck</div>
      <div class="topbar-subtitle">Hypothesis tests for means (auto Z/T)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# Utilities
# =============================
def parse_sample(text: str) -> np.ndarray:
    if not text or not text.strip():
        raise ValueError("Sample input is empty.")
    tokens = re.split(r"[,\s]+", text.strip())
    vals = [float(t) for t in tokens if t != ""]
    if not vals:
        raise ValueError("No numeric values found.")
    return np.array(vals, dtype=float)

def alt_symbol_latex(alt: str) -> str:
    return {"two-sided": r"\neq", "greater": ">", "less": "<"}[alt]

def alt_label(alt: str) -> str:
    return {"two-sided": "Two-sided (≠)", "greater": "Right-tailed (>)", "less": "Left-tailed (<)"}[alt]

def critical_values(dist, alpha: float, alt: str):
    if alt == "two-sided":
        return dist.ppf(alpha / 2), dist.ppf(1 - alpha / 2)
    if alt == "greater":
        return dist.ppf(1 - alpha)
    return dist.ppf(alpha)

def rejection_region_text(crit, alt: str, stat_symbol: str):
    if alt == "two-sided":
        lo, hi = crit
        return f"Reject H₀ if {stat_symbol} ≤ {lo:.4f} or {stat_symbol} ≥ {hi:.4f}"
    if alt == "greater":
        return f"Reject H₀ if {stat_symbol} ≥ {crit:.4f}"
    return f"Reject H₀ if {stat_symbol} ≤ {crit:.4f}"

def reject_from_stat(stat: float, crit, alt: str) -> bool:
    if alt == "two-sided":
        lo, hi = crit
        return stat <= lo or stat >= hi
    if alt == "greater":
        return stat >= crit
    return stat <= crit

def p_value(dist, stat: float, alt: str) -> float:
    if alt == "two-sided":
        return dist.sf(abs(stat)) * 2
    if alt == "greater":
        return dist.sf(stat)
    return dist.cdf(stat)

def preview_stats(x: np.ndarray):
    n = int(x.size)
    xbar = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if n >= 2 else float("nan")
    return n, xbar, s

# =============================
# Page intro
# =============================
st.markdown(
    '<div class="card">'
    '<div class="section-title">How to use</div>'
    '<div class="hint">Choose a test, paste your sample(s), set α and the alternative, then run. '
    'Rule: use Z only when <b>both</b> sample sizes are > 40; otherwise use T.</div>'
    '</div>',
    unsafe_allow_html=True,
)

# =============================
# Global test settings
# =============================
st.markdown(
    '<div class="section-title">Test settings</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)
with c1:
    alpha = st.number_input(
        "Significance level (α)",
        min_value=0.0001,
        max_value=0.5,
        value=0.05,
        step=0.01,
        format="%.4f",
        help="Common choices: 0.10, 0.05, 0.01",
    )
with c2:
    alt = st.selectbox(
        "Alternative hypothesis (Hₐ)",
        ["two-sided", "greater", "less"],
        format_func=alt_label,
        help="Two-sided tests for 'different'. Right-tailed tests for 'greater'. Left-tailed tests for 'less'.",
    )

st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Tabs: 1 mean vs 2 means
# =============================
tab1, tab2 = st.tabs(["1 Mean (one sample)", "2 Means (independent samples)"])

# -----------------------------
# 1-sample
# -----------------------------
with tab1:
    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint">Paste one sample. Enter μ₀ (the value you are testing the mean against).</div>', unsafe_allow_html=True)

    with st.form("form_1mean", clear_on_submit=False):
        mu0 = st.number_input(
            "Hypothesized mean (μ₀)",
            value=0.0,
            step=1.0,
            help="Example: μ₀ = 50 if you are testing whether the true mean could be 50.",
        )

        sample_text = st.text_area(
            "Sample values (numbers only)",
            height=130,
            placeholder="Enter numbers separated by commas, spaces, or new lines.\nExample:\n12 15 14 10 9 11",
            help="Allowed separators: comma, space, newline.",
        )

        submitted = st.form_submit_button("Run test", type="primary", use_container_width=True)

    if submitted:
        try:
            x = parse_sample(sample_text)
            n, xbar, s = preview_stats(x)
            if n < 2:
                st.error("Need at least 2 observations.")
                st.stop()

            se = s / np.sqrt(n)
            if se == 0:
                st.error("Standard error is zero (all values identical).")
                st.stop()

            use_z = (n > 40)
            stat_symbol = "z" if use_z else "t"
            stat = (xbar - mu0) / se

            dist = stats.norm() if use_z else stats.t(df=n - 1)
            pv = p_value(dist, stat, alt)
            crit = critical_values(dist, alpha, alt)
            reject = reject_from_stat(stat, crit, alt)

            st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

            a, b, c, d = st.columns(4)
            a.metric("n", n)
            b.metric("x̄", f"{xbar:.6g}")
            c.metric("s", f"{s:.6g}")
            d.metric("Test", stat_symbol.upper())

            st.markdown("**Hypotheses**")
            st.latex(rf"H_0: \mu = {mu0}")
            st.latex(rf"H_a: \mu {alt_symbol_latex(alt)} {mu0}")

            st.markdown("**Rejection region**")
            st.write(rejection_region_text(crit, alt, stat_symbol))

            st.markdown("**Test statistic and p-value**")
            st.latex(rf"{stat_symbol} = {stat:.4f}")
            st.write(f"p-value = **{pv:.6g}**")

            st.markdown("**Conclusion**")
            label = "Reject H₀" if reject else "Fail to reject H₀"
            st.markdown(f'<span class="badge">{label} (α = {alpha})</span>', unsafe_allow_html=True)

            if alt == "two-sided":
                st.write(f"There is {'strong' if reject else 'insufficient'} statistical evidence that the true mean differs from μ₀.")
            elif alt == "greater":
                st.write(f"There is {'strong' if reject else 'insufficient'} statistical evidence that the true mean is greater than μ₀.")
            else:
                st.write(f"There is {'strong' if reject else 'insufficient'} statistical evidence that the true mean is less than μ₀.")

            st.caption("Rule used: Z only when n > 40. (Z uses sample SD as σ approximation.)")

        except Exception as e:
            st.error(f"Input error: {e}")

# -----------------------------
# 2-sample (independent only)
# -----------------------------
with tab2:
    st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint">Paste two independent samples. Δ₀ is the hypothesized difference (μ₁ − μ₂). Usually 0.</div>', unsafe_allow_html=True)

    with st.form("form_2mean", clear_on_submit=False):
        delta0 = st.number_input(
            "Hypothesized difference (Δ₀ = μ₁ − μ₂)",
            value=0.0,
            step=1.0,
            help="Most common: Δ₀ = 0 (testing whether the means are different).",
        )

        left, right = st.columns(2)
        with left:
            sample1_text = st.text_area(
                "Sample 1 values",
                height=130,
                placeholder="Example:\n12 15 14 10 9 11",
                help="Allowed separators: comma, space, newline.",
            )
        with right:
            sample2_text = st.text_area(
                "Sample 2 values",
                height=130,
                placeholder="Example:\n8 7 9 6 10 7",
                help="Allowed separators: comma, space, newline.",
            )

        submitted2 = st.form_submit_button("Run test", type="primary", use_container_width=True)

    if submitted2:
        try:
            x1 = parse_sample(sample1_text)
            x2 = parse_sample(sample2_text)

            n1, xbar1, s1 = preview_stats(x1)
            n2, xbar2, s2 = preview_stats(x2)

            if n1 < 2 or n2 < 2:
                st.error("Each sample must have at least 2 observations.")
                st.stop()

            # Welch standard error (independent samples)
            se = np.sqrt((s1**2) / n1 + (s2**2) / n2)
            if se == 0:
                st.error("Standard error is zero.")
                st.stop()

            use_z = (n1 > 40 and n2 > 40)  # confirmed rule
            stat_symbol = "z" if use_z else "t"
            stat = ((xbar1 - xbar2) - delta0) / se

            if use_z:
                dist = stats.norm()
            else:
                num = (s1**2 / n1 + s2**2 / n2) ** 2
                den = ((s1**2 / n1) ** 2) / (n1 - 1) + ((s2**2 / n2) ** 2) / (n2 - 1)
                df = num / den
                dist = stats.t(df=df)

            pv = p_value(dist, stat, alt)
            crit = critical_values(dist, alpha, alt)
            reject = reject_from_stat(stat, crit, alt)

            st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

            a, b, c, d = st.columns(4)
            a.metric("n₁", n1)
            b.metric("x̄₁", f"{xbar1:.6g}")
            c.metric("n₂", n2)
            d.metric("x̄₂", f"{xbar2:.6g}")

            st.markdown("**Hypotheses**")
            st.latex(rf"H_0: \mu_1 - \mu_2 = {delta0}")
            st.latex(rf"H_a: \mu_1 - \mu_2 {alt_symbol_latex(alt)} {delta0}")

            st.markdown("**Rejection region**")
            st.write(rejection_region_text(crit, alt, stat_symbol))

            st.markdown("**Test statistic and p-value**")
            st.latex(rf"{stat_symbol} = {stat:.4f}")
            st.write(f"p-value = **{pv:.6g}**")

            st.markdown("**Conclusion**")
            label = "Reject H₀" if reject else "Fail to reject H₀"
            st.markdown(f'<span class="badge">{label} (α = {alpha})</span>', unsafe_allow_html=True)

            if alt == "two-sided":
                st.write(f"There is {'strong' if reject else 'insufficient'} statistical evidence that the two true means are different.")
            elif alt == "greater":
                st.write(f"There is {'strong' if reject else 'insufficient'} statistical evidence that μ₁ − μ₂ is greater than Δ₀.")
            else:
                st.write(f"There is {'strong' if reject else 'insufficient'} statistical evidence that μ₁ − μ₂ is less than Δ₀.")

            st.caption("Independent samples only (Welch). Rule used: Z only when both n₁ and n₂ > 40.")

        except Exception as e:
            st.error(f"Input error: {e}")