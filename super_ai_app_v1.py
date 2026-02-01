import streamlit as st
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
import io
# For OpenAI LLM (your GemmaChatLLM should be defined as in your latest version)
import openai 
MAX_HISTORY = 5


# --- Appearance: Custom CSS for modern chat-like UI ---
def inject_custom_css():
    css = """
    <style>
    body {background: linear-gradient(135deg, #eaf1fb 0%, #f5f7fa 100%);}
    .chat-container {background: #fff; border-radius: 20px; padding: 1.5rem; margin: 2rem auto 1rem auto; box-shadow: 0 6px 32px rgba(79,139,249,0.08);}
    .bubble-user {background: linear-gradient(90deg, #f0f0f0 0%, #b2b2b2 100%); color: #222; border-radius: 18px 18px 4px 18px; padding: 0.85rem 1.3rem; margin-bottom:6px; max-width: 75%; float: right; clear: both;}
    .bubble-ai {background: linear-gradient(90deg, #f0f0f0 0%, #b2d1ff 100%); color: #222; border-radius: 18px 18px 18px 4px; padding: 0.85rem 1.3rem; margin-bottom: 12px; max-width: 75%; float: left; clear: both;}
    .chat-timestamp {display:block; font-size:0.82rem; color:#888;text-align:right; font-family:monospace;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
inject_custom_css()

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: clear/download chat ---
with st.sidebar:
    st.header("🛠️ Labour charges")
    de_husk_labour = st.number_input("De-husking Labour Cost (₹/1000 piece)", min_value=0.0, value=700.0, step=50.0)
    Kalam_labour = st.number_input("Kalam Labour Cost (₹/1000 piece)", min_value=0.0, value=600.0, step=50.0)
    Thotti_price = st.number_input("Thotti Price (₹/Kg)", min_value=15.0, value=25.0, step=1.0)
    st.header("🤖 AI Settings")
    mode = st.radio("🤖 Select AI Mode", ["Normal Mode", "DeepThink Mode"])
    explain_mode = "Short"
    if mode == "DeepThink Mode":
        explain_mode = st.radio("📖 Explanation Style", ["Short", "Detailed", "Auto"])

    st.header("Chat Options")
    
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")
    if st.session_state.get("chat_history", []):
        chat_text = ""
        for u, a in st.session_state.chat_history:
            chat_text += f"You: {u}\nAI: {a}\n\n"
        chat_bytes = io.BytesIO(chat_text.encode("utf-8"))
        st.download_button(
            label="⬇️ Download Chat",
            data=chat_bytes,
            file_name="chat_history.txt",
            mime="text/plain"
        )
    st.markdown(
        "<hr><div style='text-align:center;font-size:0.95rem; color:#8aa;'>Made with ❤️ Happy Trading </div>",
        unsafe_allow_html=True
    )

# --- Streamlit Page and Inputs ---
st.set_page_config(page_title="Coconut Price Calculator + AI", layout="wide")
st.title("🌴 Coconut Fair Price Calculator with AI ")

# Inputs (core logic unchanged)
copra_price = st.number_input("Copra Price (₹/kg)", min_value=0.0, value=200.0, step=1.0)
husk_price = st.number_input("Husk Price (₹/Piece)", min_value=0.0, value=2.0, step=0.5)
outturn = st.number_input("Outturn % (kg of copra per 1000kg coconuts)", min_value=0.0, value=29.0, step=1.0)
include_husk = st.checkbox("Include Husk Value in Calculation?", value=True)

coconut_weight = st.number_input("Avg. Weight of One Coconut (kg)", min_value=0.3, value=0.48, step=0.05)

buy_price_kg = st.number_input("Planned Buying Price (₹/kg)", min_value=0.0, value=0.0, step=1.0)
buy_price_piece = st.number_input("Planned Buying Price (₹/piece)", min_value=0.0, value=0.0, step=1.0)


# --- Core Calculation ---
copra_value = outturn * copra_price * 10
husk_value = (1000 * husk_price / coconut_weight) if include_husk else 0
de_husk_cost = de_husk_labour / coconut_weight
kalam_cost = Kalam_labour / coconut_weight
Thotti_value = Thotti_price * outturn * 10
total_value = (copra_value + husk_value - (de_husk_cost + kalam_cost) + Thotti_value)
fair_price_per_kg = total_value / 1000
fair_price_per_piece = fair_price_per_kg * coconut_weight

# --- Summary Table ---
st.subheader("💰 Estimated Fair Price Summary")
df = pd.DataFrame({
    "Basis": ["Per kg", "Per piece"],
    "Fair Price (₹)": [round(fair_price_per_kg, 2), round(fair_price_per_piece, 2)],
    "Unit": ["kg", "piece"]
})
st.table(df)

# --- Rule-based AI ---
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

# --- OpenAI Chat LLM ---
class GemmaChatLLM:
    def __init__(self, model_id="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY") or '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'
        openai.api_key = api_key
        #self.client = OpenAI(api_key=api_key)
        self.model_id = model_id

    def _call(self, prompt: str) -> str:
        try:
            resp = openai.ChatCompletion.create(
                model=self.model_id,
                messages=prompt,
                max_tokens=800,
                temperature=0.7,
            )
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            return f"Error from OpenAI API: {e}"

llm = GemmaChatLLM(model_id="gpt-4o-mini")

# --- GPT-powered AI ---
def gpt_ai(user_query: str) -> str:
    style = "Short"
    if explain_mode == "Auto":
        if any(w in user_query.lower() for w in ["profit", "loss", "buy", "sell", "copra price", "required"]):
            style = "Detailed"
    elif explain_mode == "Detailed":
        style = "Detailed"

    instructions = """
    You are a precise coconut pricing and profit calculation assistant.

    Always follow these instructions:

    1. Use the variables provided under "Current Values" for all calculations. Do NOT use any external, invented, or arbitrary numbers. If a variable value is missing, say so and ask for it.
    2. You can answer only about coconut pricing, fair price calculations, profit/loss based on buy prices, or inversion to get required copra price.
    3. When showing a calculation, clearly state the formula and substitute values from "Current Values".
    4. If asked 'what copra price is needed,' invert the formula step by step using the provided values.
    5. Output the **final numeric answer**, and, if "Detailed" is requested, first show each relevant calculation step on its own line, then the box answer.
    6. When not sure, say "Sorry, I can only answer about coconut pricing and related maths using the provided values."

    ### Calculation Reference

    - **Fair Price per kg** = (copra_value + husk_value - (de_husk_cost + kalam_cost) + Thotti_value) / 1000
    - **copra_value** = outturn × copra_price × 10
    - **Fair Price per piece** = Fair Price per kg × coconut_weight
    - **Profit/Loss per 1000kg** = (Fair Price per kg - Buying Price per kg) × 1000
    - **Profit/Loss per 1000kg (piece)** = (Fair Price per piece - Buying Price per piece) × (1000 / coconut_weight)
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
    ----
    ALWAYS use the above formulas and values only; ALWAYS display the answer as a clear numeric result. Say "Not applicable" for other types of questions, clearly mentions if its profit or Loss.
    """

    # Assemble messages: system, then full chat, then latest user input
    messages = [{"role": "system", "content": system_instructions}]

    for user, ai in st.session_state.get("chat_history", []):
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": ai})
    # Add current user input as last message
    messages.append({"role": "user", "content": user_query})


    return llm._call(messages)

# --- Chart function (unchanged) ---
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

# --- Modern Chat UI (replace old input/chat logic with this) ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.subheader("💬 Chat with Coconut AI")
AVATAR_USER = '<span style="font-size:2rem;vertical-align:middle;">🧑‍💻</span>' # Or a modern SVG image or emoji
AVATAR_AI = '<span style="font-size:3rem;vertical-align:middle;">🤖</span>'      # Or a modern SVG image or emoji
# Show chat bubbles
for idx, (user, ai) in enumerate(st.session_state.chat_history):
    st.markdown(
        f"<div class='bubble-user'>{AVATAR_USER} {user}<div class='chat-timestamp'>You</div></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='bubble-ai'>{AVATAR_AI} {ai}<div class='chat-timestamp'>AI</div></div>",
        unsafe_allow_html=True
    )

# Modern chat input at the bottom
#with st.form(key="chat_input_form", clear_on_submit=True):
user_query = st.chat_input("Type your question", key="user_input")
#submitted = st.form_submit_button("Send")
if  user_query:
    if mode == "Normal Mode":
        reply = rule_based_ai(user_query)
    else:
        with st.spinner("Thinking..."):
            reply = gpt_ai(user_query)
            if any(w in user_query.lower() for w in ["profit", "loss", "buy", "sell"]):
                nums = re.findall(r"\d+\.?\d*", user_query)
                if nums:
                    planned_price = float(nums[0])
                    if "piece" in user_query.lower():
                        auto_chart(planned_price, "piece")
                    else:
                        auto_chart(planned_price, "kg")
    st.session_state.chat_history.append((user_query, reply))
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)