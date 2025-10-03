import streamlit as st
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------
# LLM Helper (OpenAI + Ollama)
# -------------------
class LLMHelper:
    def __init__(self, provider="openai", model="gpt-4o-mini", api_key=None, host="http://localhost:11434"):
        """
        provider: "openai" or "ollama"
        model: model name (OpenAI or Ollama model name)
        api_key: OpenAI key (ignored for Ollama)
        host: Ollama server URL (default http://localhost:11434)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.host = host

        if provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                self.client = None
        elif provider == "ollama":
            try:
                import requests
                self.requests = requests
            except ImportError:
                self.requests = None
        else:
            raise ValueError("Unsupported provider: choose 'openai' or 'ollama'")

    def ask(self, prompt: str) -> str:
        if self.provider == "openai":
            if not self.client:
                return "⚠️ OpenAI client not available. Install `openai`."
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}]
            )
            return response.choices[0].message.content

        elif self.provider == "ollama":
            if not self.requests:
                return "⚠️ Ollama requires `requests` library. Install it first."
            try:
                r = self.requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "system", "content": prompt}],
                        "stream": False
                    }
                )
                if r.status_code == 200:
                    data = r.json()
                    return data["message"]["content"]
                else:
                    return f"⚠️ Ollama request failed: {r.text}"
            except Exception as e:
                return f"⚠️ Ollama error: {e}"

        return "⚠️ No valid LLM client configured."


# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Coconut Price Calculator + AI", layout="centered")
st.title("🌴 Coconut Fair Price Calculator with AI ")

# Inputs
copra_price = st.number_input("Copra Price (₹/kg)", min_value=0.0, value=200.0, step=1.0)
husk_price = st.number_input("Husk Price (₹/Piece)", min_value=0.0, value=2.0, step=0.5)
outturn = st.number_input("Outturn % (kg of copra per 1000kg coconuts)", min_value=0.0, value=29.0, step=1.0)
include_husk = st.checkbox("Include Husk Value in Calculation?", value=True)
de_husk_labour = st.number_input("De-husking Labour Cost (₹/1000 piece)", min_value=0.0, value=700.0, step=50.0)
Kalam_labour = st.number_input("Kalam Labour Cost (₹/1000 piece)", min_value=0.0, value=600.0, step=50.0)
Thotti_price = st.number_input("Thotti Price (₹/Kg)", min_value=15.0, value=25.0, step=1.0)
coconut_weight = st.number_input("Avg. Weight of One Coconut (kg)", min_value=0.3, value=0.48, step=0.05)


buy_price_kg = st.number_input("Planned Buying Price (₹/kg)", min_value=0.0, value=0.0, step=1.0)
buy_price_piece = st.number_input("Planned Buying Price (₹/piece)", min_value=0.0, value=0.0, step=1.0)

mode = st.radio("🤖 Select AI Mode", ["Normal Mode", "DeepThink Mode"])
explain_mode = "Short"
if mode == "DeepThink Mode":
    explain_mode = st.radio("📖 Explanation Style", ["Short", "Detailed", "Auto"])

# -------------------
# Core Calculation
# -------------------
copra_value = outturn * copra_price*10
husk_value = (1000 * husk_price / coconut_weight) if include_husk else 0
de_husk_cost = (de_husk_labour / coconut_weight)
kalam_cost = (Kalam_labour / coconut_weight)
Thotti_value = Thotti_price * outturn *10
total_value = (copra_value + husk_value - (de_husk_cost + kalam_cost) + Thotti_value)
fair_price_per_kg = total_value / 1000
fair_price_per_piece = fair_price_per_kg * coconut_weight

# -------------------
# Tables
# -------------------
st.subheader("💰 Estimated Fair Price Summary")
df = pd.DataFrame({
    "Basis": ["Per kg", "Per piece"],
    "Fair Price (₹)": [round(fair_price_per_kg, 2), round(fair_price_per_piece, 2)],
    "Unit": ["kg", "piece"]
})
st.table(df)

# -------------------
# Rule-based AI
# -------------------
def rule_based_ai(user_query: str) -> str:
    q = user_query.strip().lower()
    numbers = re.findall(r"\d+\.?\d*", q)

    if "piece" in q and not numbers:
        return f"👉 Fair Buying Price per coconut = **₹{fair_price_per_piece:.2f}/piece**"
    if "fair price" in q or ("kg" in q and not numbers):
        return f"👉 Fair Buying Price per kg = **₹{fair_price_per_kg:.2f}/kg**"

    if numbers:
        planned_price = float(numbers[0])
        if "piece" in q:
            diff = fair_price_per_piece - planned_price
            profit_loss = diff * (1000 / coconut_weight)
            return f"At ₹{planned_price:.2f}/piece → Profit/Loss: ₹{profit_loss:,.2f} per 1000kg batch."
        else:
            diff = fair_price_per_kg - planned_price
            profit_loss = diff * 1000
            return f"At ₹{planned_price:.2f}/kg → Profit/Loss: ₹{profit_loss:,.2f} per 1000kg."
    return "🤔 Ask about fair price or profit/loss."

# -------------------
# GPT-powered AI
# -------------------
# choose provider here: "openai" or "ollama"
llm = LLMHelper(provider="openai", model="gpt-4o-mini")
# llm = LLMHelper(provider="ollama", model="llama3")

def gpt_ai(user_query: str) -> str:
    style = "Short"
    if explain_mode == "Auto":
        if any(w in user_query.lower() for w in ["profit", "loss", "buy", "sell", "copra price", "required"]):
            style = "Detailed"
    elif explain_mode == "Detailed":
        style = "Detailed"

    instructions = """
    You are an AI assistant for coconut trading.
    Rules:
    - Fair Price/kg = ( copra_value + husk_value - (de_husk_cost + kalam_cost) + Thotti_value) ÷ 1000
    - Fair Price/piece = Fair Price/kg × coconut_weight
    - Profit/Loss per 1000kg = (FairPrice/kg - BuyingPrice/kg) × 1000
    - Profit/Loss per 1000kg batch (piece) = (FairPrice/piece - BuyingPrice/piece) × (1000 ÷ coconut_weight)
    If asked 'what copra price is needed', reverse the formula.
    """

    if style == "Detailed":
        instructions += " Explain step by step before giving final result."
    else:
        instructions += " Give only the final result briefly."

    prompt = f"""
    {instructions}

    ### Current Values:
    - Outturn = {outturn}
    - Copra Price = ₹{copra_price}
    - Husk Value = ₹{husk_value}
    - de_husk_cost = ₹{de_husk_cost}
    - kalam_cost = ₹{kalam_cost}
    - Thotti_value = ₹{Thotti_value}
    - Fair Price = ₹{fair_price_per_kg:.2f}/kg OR ₹{fair_price_per_piece:.2f}/piece
    - Coconut Weight = {coconut_weight} kg

    ### User Question:
    {user_query}
    """
    return llm.ask(prompt)

# -------------------
# Auto Chart for DeepThink
# -------------------
def auto_chart(planned_price: float, basis: str):
    if planned_price <= 0:
        return
    if basis == "kg":
        fair = fair_price_per_kg
        diff = fair - planned_price
        profit_loss = diff * 1000
    else:
        fair = fair_price_per_piece
        diff = fair - planned_price
        profit_loss = diff * (1000 / coconut_weight)

    # Price comparison
    fig, ax = plt.subplots()
    ax.bar(["Fair Price", "Buying Price"], [fair, planned_price])
    ax.set_title(f"{basis.upper()} Price Comparison")
    st.pyplot(fig)

    # Profit/Loss
    fig2, ax2 = plt.subplots()
    ax2.bar(["Profit/Loss"], [profit_loss], color="green" if profit_loss >= 0 else "red")
    ax2.set_title(f"{basis.upper()} Profit/Loss (per 1000kg batch)")
    st.pyplot(fig2)

# -------------------
# Chat
# -------------------
st.subheader("💬 Coconut AI Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask a question...")

if user_query:
    if mode == "Normal Mode":
        reply = rule_based_ai(user_query)
    else:
        reply = gpt_ai(user_query)
        if any(w in user_query.lower() for w in ["profit", "loss", "buy", "sell"]):
            nums = re.findall(r"\d+\.?\d*", user_query)
            if nums:
                planned_price = float(nums[0])
                if "piece" in user_query.lower():
                    auto_chart(planned_price, "piece")
                else:
                    auto_chart(planned_price, "kg")

    st.session_state.chat_history.append(("You", user_query))
    st.session_state.chat_history.append(("AI", reply))

for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{'🧑' if sender=='You' else '🤖'} {sender}:** {msg}")
