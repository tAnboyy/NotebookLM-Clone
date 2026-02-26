
import gradio as gr

import os
import secrets
import urllib.parse

import requests
import streamlit as st


def greet(name):
    return "Hello " + name + "!!. Testing Hugging Face deployment!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()


HF_LOGIN_PAGE = "https://huggingface.co/login"
HF_AUTH_URL = "https://huggingface.co/oauth/authorize"
HF_TOKEN_URL = "https://huggingface.co/oauth/token"
HF_SCOPE = "read:org"  # request the minimum scope needed for metadata

HF_CLIENT_ID = os.environ.get("HF_OAUTH_CLIENT_ID")
HF_CLIENT_SECRET = os.environ.get("HF_OAUTH_CLIENT_SECRET")
HF_REDIRECT_URI = os.environ.get("HF_OAUTH_REDIRECT_URI", "http://localhost:8501/")

st.set_page_config(page_title="Login to Hugging Face", page_icon="📓", layout="centered")
st.markdown(
    """
    <style>
    .hf-panel {
        background: linear-gradient(180deg, #050913 0%, #131a31 100%);
        border-radius: 18px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.45);
        max-width: 640px;
        margin: 40px auto;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .hf-button {
        background: linear-gradient(90deg, #244ce7, #a33bf0);
        color: #fff;
        font-weight: 600;
        font-size: 18px;
        padding: 14px 26px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
        border: none;
        text-decoration: none;
    }
    .hf-button:hover {
        transform: translateY(-1px);
    }
    .hf-button span {
        margin-left: 4px;
    }
    .hf-note {
        color: #b9c8f5;
        font-size: 14px;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hf-panel">
        <div class="hf-note">
            Sign in to Hugging Face, then authorize the NotebookLM experience.
        </div>
        <a class="hf-button" href="https://huggingface.co/login" target="_blank" rel="noreferrer">
            <span>🧑‍💻</span>
            <span>Sign in with Hugging Face</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

if not (HF_CLIENT_ID and HF_CLIENT_SECRET):
    st.warning(
        "Set HF_OAUTH_CLIENT_ID and HF_OAUTH_CLIENT_SECRET in the environment before running this app."
    )
    st.stop()

if "hf_oauth_state" not in st.session_state:
    st.session_state["hf_oauth_state"] = secrets.token_urlsafe(16)

state = st.session_state["hf_oauth_state"]
params = {
    "client_id": HF_CLIENT_ID,
    "response_type": "code",
    "scope": HF_SCOPE,
    "redirect_uri": HF_REDIRECT_URI,
    "state": state,
}

auth_url = f"{HF_AUTH_URL}?{urllib.parse.urlencode(params)}"

st.markdown(
    "Use this OAuth link to grant the app access to Hugging Face. After consenting, you will be redirected back to the redirect URI with a `code` parameter."
)
st.markdown(f"[Open Hugging Face to authenticate]({auth_url})")

code = st.text_input("Paste the `code` from the redirect URL here", key="hf_oauth_code")

if st.button("Exchange code for token"):
    if not code:
        st.error("Provide the `code` query parameter returned by the redirect.")
    else:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": HF_REDIRECT_URI,
        }
        try:
            token_resp = requests.post(
                HF_TOKEN_URL,
                data=data,
                auth=(HF_CLIENT_ID, HF_CLIENT_SECRET),
                timeout=15,
            )
            token_resp.raise_for_status()
            token = token_resp.json()
            st.session_state["hf_session"] = token
            st.success("Authentication completed. Keep this session token secured.")
        except requests.RequestException as exc:
            st.error(f"Unable to fetch token: {exc}")

if "hf_session" in st.session_state:
    st.subheader("Authenticated session")
    st.json(st.session_state["hf_session"])

st.caption("Tokens are stored only in the Streamlit session. For production, persist tokens in encrypted storage and rotate them regularly.")

