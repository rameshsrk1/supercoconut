import streamlit as st
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from pydantic import PrivateAttr
from langchain.llms.base import LLM
from huggingface_hub import InferenceClient
import datetime

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


# --- Custom Wrapper using InferenceClient.chat.completions ---
class GemmaChatLLM(LLM):
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"  # Change to your preferred model
    temperature: float = 0.7
    max_tokens: int = 1024

    _client: InferenceClient = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = InferenceClient(
            provider="together",
            api_key=st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Use chat completions instead of text generation
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        #return response.choices[0].message['content']
        # Extract only the generated message (i.e., the answer)
        # Debug the response to inspect its structure
        print("Response from API:", response)

        try:
            # Extract content from the message object
            answer = response.choices[0].message.content.strip()
            print("Answer from Response :", answer)
            # Return the answer only
            return answer if answer else "No valid response received"
        except (AttributeError, KeyError, IndexError):
            # In case the structure doesn't match, return a helpful message
            return "Error: Unexpected response structure."
    


    @property
    def _llm_type(self) -> str:
        return "huggingface-chat"
        
    def ask(self, prompt: str) -> str:
        """Wrapper so it behaves like LLMHelper.ask"""
        return self._call(prompt)


# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Coconut Price Calculator + AI", layout="centered")
st.title("🌴 Coconut Buying Price Calculator with AI Chat")

# Inputs
copra_price = st.number_input("Copra Price (₹/kg)", min_value=0.0, value=180.0, step=5.0)
husk_price = st.number_input("Husk Price (₹/Piece)", min_value=0.0, value=2.0, step=0.2)
outturn = st.number_input("Outturn (kg of copra per 100kg coconuts)", min_value=0.0, value=28.0, step=0.5)
include_husk = st.checkbox("Include Husk Value in Calculation?", value=True)
coconut_weight = st.number_input("Avg. Weight of One Coconut (kg)", min_value=0.3, value=.45, step=0.1)

buy_price_kg = st.number_input("Planned Buying Price (₹/kg)", min_value=0.0, value=0.0, step=1.0)
buy_price_piece = st.number_input("Planned Buying Price (₹/piece)", min_value=0.0, value=0.0, step=1.0)

mode = st.radio("🤖 Select AI Mode", ["Normal Mode", "DeepThink Mode"])
explain_mode = "Short"
if mode == "DeepThink Mode":
    explain_mode = st.radio("📖 Explanation Style", ["Short", "Detailed", "Auto"])

# -------------------
# Core Calculation
# -------------------
copra_value = 10*outturn * copra_price
husk_value = 2000 * husk_price if include_husk else 0
total_value = copra_value + husk_value
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
#llm = LLMHelper(provider="openai", model="gpt-4o-mini")
llm = GemmaChatLLM(model_id="mistralai/Mistral-7B-Instruct-v0.3")
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
    - Fair Price/kg = (outturn × copra_price + husk_value) ÷ 1000
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

# ✅ Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_box_id = "chat-box"

# CSS for WhatsApp-like chat
chat_css = f"""
<style>
.chat-box {{
    height: 450px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 10px;
    border-radius: 10px;
    background-color: #e5ddd5; /* WhatsApp background */
    display: flex;
    flex-direction: column;
}}
.msg-row {{
    display: flex;
    align-items: flex-end;
    margin-bottom: 10px;
}}
.user-row {{
    justify-content: flex-end;
}}
.ai-row {{
    justify-content: flex-start;
}}
.profile-icon {{
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background-color: #ccc;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    margin: 0 6px;
}}
.chat-bubble {{
    padding: 8px 12px;
    border-radius: 12px;
    max-width: 70%;
    word-wrap: break-word;
    font-size: 15px;
    position: relative;
}}
.user-msg {{
    background-color: #dcf8c6;
    border-bottom-right-radius: 0;
    text-align: left;
}}
.ai-msg {{
    background-color: #fff;
    border-bottom-left-radius: 0;
    text-align: left;
}}
.timestamp {{
    font-size: 11px;
    color: gray;
    margin-top: 2px;
    text-align: right;
}}
</style>
"""
st.markdown(chat_css, unsafe_allow_html=True)

# Build chat HTML
chat_html = f"<div id='{chat_box_id}' class='chat-box'>"
for sender, msg, ts in st.session_state.chat_history:
    time_str = ts.strftime("%H:%M")
    if sender == "You":
        chat_html += f"""
        <div class='msg-row user-row'>
            <div class='chat-bubble user-msg'>
                {msg}
                <div class='timestamp'>{time_str}</div>
            </div>
            <div class='profile-icon'>🧑</div>
        </div>
        """
    else:
        chat_html += f"""
        <div class='msg-row ai-row'>
            <div class='profile-icon'>🤖</div>
            <div class='chat-bubble ai-msg'>
                {msg}
                <div class='timestamp'>{time_str}</div>
            </div>
        </div>
        """
chat_html += "</div>"

# Render chat
st.markdown(chat_html, unsafe_allow_html=True)

# Auto-scroll script
scroll_js = f"""
<script>
    var chatBox = document.getElementById('{chat_box_id}');
    if (chatBox) {{
        chatBox.scrollTop = chatBox.scrollHeight;
    }}
</script>
"""
st.markdown(scroll_js, unsafe_allow_html=True)

# Spacer
st.markdown("---")

# Chat input
user_query = st.text_input("Type a message...", key="chat_input")

if user_query:
    # AI response
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

    # Store messages with timestamp
    now = datetime.datetime.now()
    st.session_state.chat_history.append(("You", user_query, now))
    st.session_state.chat_history.append(("AI", reply, now))

    # Clear input & rerun
    st.session_state["chat_input"] = ""
    st.experimental_rerun()
