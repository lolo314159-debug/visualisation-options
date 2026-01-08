import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go

# --- MOTEUR DE CALCUL (Black-Scholes) ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 1e-6:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    else:
        return K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)

# --- CONFIGURATION ---
st.set_page_config(page_title="Option Strategy Pro", layout="wide")
st.title("🦅 Visualiseur Multi-Jambes (Iron Condor & Spreads)")

with st.sidebar:
    st.header("⚙️ Paramètres Marché")
    S0 = st.number_input("Prix du Sous-jacent", value=100.0)
    vol = st.slider("Volatilité Implicite (%)", 5, 100, 20) / 100
    rate = st.slider("Taux d'intérêt (%)", 0.0, 10.0, 2.0) / 100
    
    st.divider()
    days_to_expiry = st.number_input("Jours à l'échéance", value=45)
    days_passed = st.slider("Passage du temps (jours)", 0, int(days_to_expiry), 0)
    
    T_init = days_to_expiry / 365
    T_current = (days_to_expiry - days_passed) / 365

# --- CONSTRUCTION DES JAMBES ---
st.subheader("🛠️ Configuration des Jambes")
legs = []
cols = st.columns(4)

for i in range(4):
    with cols[i]:
        st.markdown(f"**Jambe {i+1}**")
        active = st.checkbox("Activer", value=(i==0), key=f"active_{i}")
        if active:
            side = st.selectbox("Action", ["Achat (Long)", "Vente (Short)"], key=f"side_{i}")
            opt_type = st.selectbox("Type", ["Call", "Put"], key=f"type_{i}")
            strike = st.number_input("Strike", value=100.0 + (i*5 if i<2 else -i*5), key=f"strike_{i}")
            price = st.number_input("Prix", value=2.0, key=f"price_{i}")
            
            # Conversion pour le calcul
            qty = 1 if side == "Achat (Long)" else -1
            legs.append({"type": opt_type.lower(), "k": strike, "p": price, "q": qty})

# --- CALCUL DU PNL ---
S_range = np.linspace(S0 * 0.6, S0 * 1.4, 250)

def calculate_pnl(S_vec, t_rem, is_expiry=False):
    total_pnl = np.zeros_like(S_vec)
    for leg in legs:
        leg_pnl = []
        for s in S_vec:
            if is_expiry:
                val = max(0, s - leg['k']) if leg['type'] == "call" else max(0, leg['k'] - s)
            else:
                val = black_scholes(s, leg['k'], t_rem, rate, vol, leg['type'])
            
            # PnL = (Valeur actuelle - Prix payé/reçu) * Quantité
            leg_pnl.append((val - leg['p']) * leg['q'])
        total_pnl += np.array(leg_pnl)
    return total_pnl

pnl_now = calculate_pnl(S_range, T_current)
pnl_expiry = calculate_pnl(S_range, 0, is_expiry=True)

# --- GRAPHIQUE ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=S_range, y=pnl_now, name="Profil Actuel (T+t)", line=dict(color='#00d1b2', width=3)))
fig.add_trace(go.Scatter(x=S_range, y=pnl_expiry, name="À l'échéance", line=dict(color='#ff3860', dash='dash')))

fig.add_hline(y=0, line_color="white", opacity=0.5)
fig.add_vline(x=S0, line_color="yellow", line_dash="dot", opacity=0.5, annotation_text="Prix actuel")

fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    yaxis_title="Profit / Perte ($)",
    xaxis_title="Prix du Sous-jacent",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

# --- RÉCAPITULATIF ---
if legs:
    net_cost = sum(l['p'] * l['q'] for l in legs)
    st.sidebar.metric("Crédit/Débit Net", f"{-net_cost:.2f} $", help="Négatif = Débit (payé), Positif = Crédit (reçu)")
