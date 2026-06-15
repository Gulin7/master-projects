import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation import (
    simulate_black_scholes_euler,
    simulate_black_scholes_book,
    compute_path_statistics,
    mu_constant,
    mu_time_varying,
    sigma_constant,
    sigma_time_varying,
)


# PAGE CONFIG
st.set_page_config(
    page_title="Black-Scholes Simulation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stMetric {
            background-color: #1e2130;
            padding: 10px;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# SIDEBAR
st.sidebar.title("⚙️ Simulation Parameters")

st.sidebar.header("Simulation Scheme")
scheme_choice = st.sidebar.selectbox(
    "Choose simulation method",
    [
        "Euler-Maruyama",
        "Book Algorithm 5.1 (Exponential)",
    ],
)

if scheme_choice == "Euler-Maruyama":
    simulate_func = simulate_black_scholes_euler
    scheme_name = "Euler-Maruyama"
else:
    simulate_func = simulate_black_scholes_book
    scheme_name = "Book Algorithm 5.1 (Exponential)"

st.sidebar.header("Market Parameters")
S0 = st.sidebar.number_input("Initial Price S₀", 10.0, 1000.0, 100.0, step=5.0)
T = st.sidebar.slider("Time Horizon T (years)", 0.1, 5.0, 1.0, 0.1)
n_steps = st.sidebar.slider("Time Steps N", 50, 1000, 252, 50)
n_paths = st.sidebar.slider("Simulated Paths", 100, 5000, 500, 100)
seed = int(st.sidebar.number_input("Random Seed", 0, 9999, 42))

st.sidebar.header("Drift Function μ(t, S)")
mu_choice = st.sidebar.selectbox(
    "μ function",
    ["Constant", "Time-Varying"],
)

if mu_choice == "Constant":
    mu0 = st.sidebar.slider("μ₀", -0.2, 0.5, 0.05, 0.01)
    mu_func = mu_constant(mu0)
else:
    mu0 = st.sidebar.slider("μ₀ base drift", -0.1, 0.4, 0.05, 0.01)
    mu_amp = st.sidebar.slider("Amplitude A", 0.0, 0.3, 0.1, 0.01)
    mu_freq = st.sidebar.slider("Frequency f", 0.5, 4.0, 1.0, 0.5)
    mu_func = mu_time_varying(mu0, mu_amp, mu_freq)

st.sidebar.header("Volatility Function σ(t, S)")
sigma_choice = st.sidebar.selectbox(
    "σ function",
    ["Constant", "Time-Varying"],
)

if sigma_choice == "Constant":
    sigma0 = st.sidebar.slider("σ₀", 0.01, 0.8, 0.2, 0.01)
    sigma_func = sigma_constant(sigma0)
else:
    sigma0 = st.sidebar.slider("σ₀ base volatility", 0.05, 0.5, 0.2, 0.01)
    sigma_amp = st.sidebar.slider("Amplitude", 0.0, 0.9, 0.3, 0.05)
    sigma_freq = st.sidebar.slider("Frequency", 0.5, 4.0, 1.0, 0.5)
    sigma_func = sigma_time_varying(sigma0, sigma_amp, sigma_freq)

paths_to_show = st.sidebar.slider("Paths to display", 10, 200, 50, 10)


# RUN SIMULATION
time_grid, paths = simulate_func(
    S0=S0,
    T=T,
    n_steps=n_steps,
    n_paths=n_paths,
    mu_func=mu_func,
    sigma_func=sigma_func,
    seed=seed,
)

stats = compute_path_statistics(paths, time_grid)
S_T = paths[:, -1]


# MAIN PAGE
st.title("📈 Black-Scholes Stock Price Simulation")
st.markdown(
    f"**Simulation scheme:** `{scheme_name}`  \n"
    f"**Current drift:** `{mu_func.__name__}`  \n"
    f"**Current volatility:** `{sigma_func.__name__}`"
)

col1, col2, col3 = st.columns(3)
col1.metric("Initial Price S₀", f"${S0:.2f}")
col2.metric("Mean Final Price", f"${stats['mean'].iloc[-1]:.2f}")
col3.metric("Std. Dev. Final Price", f"${stats['std'].iloc[-1]:.2f}")

tab1, tab2, tab3 = st.tabs(["📉 Price Paths", "📊 Terminal Distribution", "📐 Statistics"])


# TAB 1 — PRICE PATHS
with tab1:
    st.subheader(f"Simulated Paths (showing {min(paths_to_show, n_paths)} of {n_paths})")

    fig1 = go.Figure()

    for i in range(min(paths_to_show, n_paths)):
        fig1.add_trace(
            go.Scatter(
                x=time_grid,
                y=paths[i],
                mode="lines",
                line=dict(width=0.8, color="rgba(100,180,255,0.30)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig1.add_trace(
        go.Scatter(
            x=time_grid,
            y=stats["mean"],
            mode="lines",
            line=dict(color="gold", width=2.5),
            name="Mean",
        )
    )

    fig1.add_trace(
        go.Scatter(
            x=time_grid,
            y=stats["median"],
            mode="lines",
            line=dict(color="tomato", width=2, dash="dash"),
            name="Median",
        )
    )

    fig1.update_layout(
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Stock Price",
        height=550,
    )

    st.plotly_chart(fig1, use_container_width=True)

    if scheme_choice == "Euler-Maruyama":
        st.info(
            """
**SDE simulated:**

dS = μ(t,S)·S·dt + σ(t,S)·S·dW

**Euler-Maruyama step:**

S_(i+1) = S_i + μ(t_i,S_i)·S_i·Δt + σ(t_i,S_i)·S_i·√Δt·Z_i, where Z_i ~ N(0,1)
"""
        )
    else:
        st.info(
            """
**Book-style exponential update:**

S_(i+1) = S_i · exp((μ(t_i,S_i) - 0.5·σ(t_i,S_i)^2)·Δt + σ(t_i,S_i)·√Δt·Z_i)

where Z_i ~ N(0,1)
"""
        )


# TAB 2 — TERMINAL DISTRIBUTION
with tab2:
    st.subheader("Distribution of Terminal Price S(T)")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Histogram(
            x=S_T,
            nbinsx=50,
            marker_color="rgba(0,180,255,0.75)",
            marker_line_color="white",
            marker_line_width=0.3,
            name="S(T)",
        )
    )

    fig2.add_vline(x=S0, line_dash="dash", line_color="gold", annotation_text=f"S₀ = {S0}")
    fig2.add_vline(
        x=np.mean(S_T),
        line_dash="dot",
        line_color="lime",
        annotation_text=f"Mean = {np.mean(S_T):.2f}",
    )

    fig2.update_layout(
        template="plotly_dark",
        xaxis_title="Terminal Price S(T)",
        yaxis_title="Frequency",
        height=500,
    )

    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"${np.mean(S_T):.2f}")
    c2.metric("Median", f"${np.median(S_T):.2f}")
    c3.metric("5th Percentile", f"${np.percentile(S_T, 5):.2f}")
    c4.metric("95th Percentile", f"${np.percentile(S_T, 95):.2f}")


# TAB 3 — STATISTICS
with tab3:
    st.subheader("Statistics Over Time")

    fig3 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Mean and Median", "Standard Deviation"),
    )

    fig3.add_trace(
        go.Scatter(
            x=time_grid,
            y=stats["mean"],
            mode="lines",
            line=dict(color="gold", width=2),
            name="Mean",
        ),
        row=1,
        col=1,
    )

    fig3.add_trace(
        go.Scatter(
            x=time_grid,
            y=stats["median"],
            mode="lines",
            line=dict(color="tomato", width=2, dash="dash"),
            name="Median",
        ),
        row=1,
        col=1,
    )

    fig3.add_trace(
        go.Scatter(
            x=time_grid,
            y=stats["std"],
            mode="lines",
            line=dict(color="orange", width=2),
            name="Std Dev",
        ),
        row=1,
        col=2,
    )

    fig3.update_layout(
        template="plotly_dark",
        height=450,
    )

    fig3.update_xaxes(title_text="Time", row=1, col=1)
    fig3.update_xaxes(title_text="Time", row=1, col=2)
    fig3.update_yaxes(title_text="Price", row=1, col=1)
    fig3.update_yaxes(title_text="Std Dev", row=1, col=2)

    st.plotly_chart(fig3, use_container_width=True)


st.divider()
st.markdown(
    """
<div style='text-align:center; color:gray; font-size:13px'>
    Black-Scholes-Type Simulation · Euler-Maruyama and Book Algorithm 5.1
</div>
""",
    unsafe_allow_html=True,
)


