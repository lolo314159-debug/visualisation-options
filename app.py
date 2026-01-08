import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go

# --- MOTEUR DE CALCUL (Black-Scholes) ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 1e-6: # Proche de l'expiration
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == "call":
        return S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    else:
        return K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Option Master Pro", layout="wide")
st.title("📈 Analyseur de Stratégies d'Options")

# --- SIDEBAR : PARAMÈTRES MARCHÉ ---
with st.sidebar:
    st.header("⚙️ Paramètres Marché")
    S0 = st.number_input("Prix du Sous-jacent (S)", value=100.0, step=1.0)
    vol = st.slider("Volatilité Implicite (%)", 5, 100, 25) / 100
    rate = st.slider("Taux d'intérêt (%)", 0.0, 10.0, 2.0) / 100
    
    st.divider()
    st.header("⏳ Évolution Temporelle")
    days_to_expiry = st.number_input("Jours totaux", value=45)
    days_passed = st.slider("Jours écoulés", 0, int(days_to_expiry), 0)
    
    T_init = days_to_expiry / 365
    T_current = (days_to_expiry - days_passed) / 365

# --- INTERFACE PRINCIPALE : CONSTRUCTION STRATÉGIE ---
st.subheader("🛠️ Construction de la Stratégie")
col1, col2, col3, col4 = st.columns(4)

with col1: leg1_type = st.selectbox("Type (L1)", ["Call", "Put"], key="t1")
with col2: leg1_strike = st.number_input("Strike (L1)", value=105.0, key="s1")
with col3: leg1_qty = st.number_input("Quantité (L1)", value=1, help="Positif = Achat, Négatif = Vente")
with col4: leg1_price = st.number_input("Prix payé/reçu (L1)", value=2.5)

# Ajout d'une deuxième jambe optionnelle
expander = st.expander("Ajouter une deuxième jambe (Spread/Straddle)")
with expander:
    c1, c2, c3, c4 = st.columns(4)
    leg2_active = st.checkbox("Activer Leg 2")
    leg2_type = c1.selectbox("Type (L2)", ["Call", "Put"], key="t2")
    leg2_strike = c2.number_input("Strike (L2)", value=95.0, key="s2")
    leg2_qty = c3.number_input("Quantité (L2)", value=0, key="q2")
    leg2_price = c4.number_input("Prix payé/reçu (L2)", value=1.0, key="p2")

# --- CALCULS ---
S_range = np.linspace(S0 * 0.7, S0 * 1.3, 200)

def get_strategy_pnl(S_vec, time_remaining, is_expiry=False):
    pnl = []
    for s in S_vec:
        # Leg 1
        val1 = (s - leg1_strike if leg1_type == "Call" else leg1_strike - s) if is_expiry else black_scholes(s, leg1_strike, time_remaining, rate, vol, leg1_type.lower())
        pnl_leg1 = (val1 - leg1_price) * leg1_qty
        
        # Leg 2
        pnl_leg2 = 0
        if leg2_active:
            val2 = (s - leg2_strike if leg2_type == "Call" else leg2_strike - s) if is_expiry else black_scholes(s, leg2_strike, time_remaining, rate, vol, leg2_type.lower())
            pnl_leg2 = (val2 - leg2_price) * leg2_qty
            
        pnl.append(pnl_leg1 + pnl_leg2)
    return pnl

pnl_now = get_strategy_pnl(S_range, T_current)
pnl_expiry = get_strategy_pnl(S_range, 0, is_expiry=True)

# --- GRAPHIQUE ---
fig = go.Figure()

# Zone Profit/Perte (Couleur)
fig.add_trace(go.Scatter(x=S_range, y=pnl_now, name="Profil Actuel", line=dict(color='#00d1b2', width=4)))
fig.add_trace(go.Scatter(x=S_range, y=pnl_expiry, name="À l'expiration", line=dict(color='#ff3860', dash='dash')))

fig.update_layout(
    title="Profil de Profit et Perte (PnL)",
    xaxis_title="Prix du Sous-jacent",
    yaxis_title="Profit / Perte ($)",
    hovermode="x unified",
    template="plotly_dark"
)
fig.add_hline(y=0, line_color="white", line_opacity=0.5)
fig.add_vline(x=S0, line_color="yellow", line_dash="dot", annotation_text="Prix Actuel")

st.plotly_chart(fig, use_container_width=True)

# --- GRÉCOQUES (Moyenne sur la stratégie) ---
st.info(f"💡 Analyse : À J+{days_passed}, l'impact du temps (Theta) rapproche votre courbe de la ligne pointillée rouge.")
