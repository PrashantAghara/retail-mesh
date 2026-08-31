import streamlit as st
from api_client import stream_chat, synthesize_speech, transcribe_audio

st.set_page_config(page_title="RetailMesh Assistant", page_icon="🛒", layout="centered")

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 16px; padding: 4px 8px; }
    .step-status {
        font-size: 0.85rem; color: #6b7280; font-style: italic;
        display: flex; align-items: center; gap: 8px; padding: 4px 0;
    }
    .step-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #3b82f6;
        animation: pulse 1.2s infinite;
    }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

    /* Anchor the mic toggle to sit at the same bottom bar as the chat input.
       Streamlit's centered layout maxes out around 752px — adjust `left` if
       your window width/zoom makes this drift; it's a visual-only overlay,
       not a native part of the input bar (Streamlit doesn't support that). */
    div[data-testid="stButton"]:has(button[kind="secondary"][aria-label="mic-toggle"]) {
        position: fixed;
        bottom: 14px;
        left: calc(50% - 360px);
        z-index: 1000;
    }
    div[data-testid="stButton"]:has(button[kind="secondary"][aria-label="mic-toggle"]) button {
        border-radius: 50%;
        width: 44px;
        height: 44px;
        padding: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛒 RetailMesh Assistant")
st.caption(
    "Ask about orders, products, or policies — attach a shelf photo, or tap 🎙️ to talk."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_open" not in st.session_state:
    st.session_state.voice_open = False


# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], width=200)
        if msg.get("is_voice") and msg.get("audio"):
            st.audio(
                msg["audio"], format="audio/wav"
            )  # no autoplay on history — only the live turn autoplays
        else:
            st.write(msg["text"])


def run_query(query, image_bytes=None, image_name=None, is_voice=False):
    user_display_text = query

    if is_voice:
        pending_audio = st.session_state.pop("_pending_audio")
        with st.spinner("Transcribing…"):
            user_display_text = transcribe_audio(pending_audio)

    st.session_state.messages.append(
        {
            "role": "user",
            "text": user_display_text,
            "image": image_bytes,
        }
    )
    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=200)
        st.write(user_display_text)

    with st.chat_message("assistant"):
        step_placeholder = st.empty()

        def on_step(label):
            step_placeholder.markdown(
                f'<div class="step-status"><div class="step-dot"></div>{label}</div>',
                unsafe_allow_html=True,
            )

        result = stream_chat(user_display_text, image_bytes, image_name, on_step)
        step_placeholder.empty()

        response_text = (
            result["response"]
            if result
            else "Something went wrong — no response received."
        )

        response_audio = None
        if is_voice:
            with st.spinner("Generating voice reply…"):
                response_audio = synthesize_speech(response_text)
            st.audio(response_audio, format="audio/wav", autoplay=True)
        else:
            st.write(response_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": response_text,
            "audio": response_audio,
            "is_voice": is_voice,
        }
    )


# --- Mic toggle, CSS-anchored to the chat input's bottom bar ---
if st.button("🎙️", key="mic_toggle", help="Record a voice question"):
    st.session_state.voice_open = not st.session_state.voice_open

if st.session_state.voice_open:
    audio_value = st.audio_input("Record your question", label_visibility="collapsed")
    if audio_value is not None:
        st.session_state["_pending_audio"] = audio_value.getvalue()
        st.session_state.voice_open = False
        run_query(None, is_voice=True)
        st.rerun()

# --- Native chat input with built-in "+" file attach ---
chat_value = st.chat_input(
    "Message RetailMesh…",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
)

if chat_value and chat_value.text:
    image_bytes, image_name = None, None
    if chat_value.files:
        uploaded = chat_value.files[0]
        image_bytes = uploaded.getvalue()
        image_name = uploaded.name
    run_query(chat_value.text, image_bytes, image_name)
