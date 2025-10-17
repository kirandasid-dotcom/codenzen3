import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tempfile
import os
import requests
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

# === IBM Watson TTS Setup ===
# Replace these with your IBM Watson TTS API key and URL
IBM_TTS_APIKEY = "YOUR_IBM_WATSON_TTS_APIKEY"
IBM_TTS_URL = "YOUR_IBM_WATSON_TTS_URL"

TTS_VOICES = {
    "Lisa (Female)": "en-US_LisaV3Voice",
    "Michael (Male)": "en-US_MichaelV3Voice",
    "Allison (Female)": "en-US_AllisonV3Voice"
}

# === Load IBM Granite 3.2 2B model from Hugging Face ===
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "ibm-granite/granite-3.2-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    return pipe

pipe = load_model()

# === Generate rewritten text with prompt chaining ===
def generate_tone_adaptive_text(input_text: str, tone: str) -> str:
    # Step 1: Basic rewrite prompt
    prompt_1 = (
        f"Rewrite the following text in a {tone.lower()} tone while preserving its original meaning:\n\n{input_text}\n\nRewritten:"
    )
    out_1 = pipe(prompt_1, max_new_tokens=300, temperature=0.7, do_sample=True)[0]["generated_text"]
    rewritten_1 = out_1.split("Rewritten:")[-1].strip()

    # Step 2: Enhance stylistic quality prompt chaining
    prompt_2 = (
        f"Improve the following text to make the {tone.lower()} tone more expressive, fluent, and natural while keeping the meaning:\n\n{rewritten_1}\n\nEnhanced:"
    )
    out_2 = pipe(prompt_2, max_new_tokens=300, temperature=0.7, do_sample=True)[0]["generated_text"]
    rewritten_2 = out_2.split("Enhanced:")[-1].strip()

    return rewritten_2

# === IBM Watson Text-to-Speech conversion ===
def ibm_tts(text: str, voice: str):
    url = f"{IBM_TTS_URL}/v1/synthesize"
    headers = {
        "Content-Type": "application/json",
        "Accept": "audio/mp3"
    }
    auth = ("apikey", IBM_TTS_APIKEY)
    data = {
        "text": text,
        "voice": voice,
        "accept": "audio/mp3"
    }
    response = requests.post(url, json=data, headers=headers, auth=auth)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error(f"TTS API error: {response.status_code} - {response.text}")
        return None

# === Streamlit UI ===
st.set_page_config(page_title="EchoVerse", page_icon="🔊", layout="wide")
st.title("🔊 EchoVerse — AI Audiobook Creator")

st.markdown(
    """
    EchoVerse transforms your text into expressive audio narrations with customizable tones and voices.
    Upload a text file or paste your text, select the tone and voice, and generate natural-sounding audiobooks.
    """
)

# --- Input text section ---
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📄 Upload a .txt file", type=["txt"])
    pasted_text = st.text_area("✍️ Or paste your text here", height=220)

input_text = ""
if uploaded_file:
    input_text = uploaded_file.read().decode("utf-8")
elif pasted_text:
    input_text = pasted_text

# Display original text
if input_text:
    with col2:
        st.subheader("Original Text")
        st.write(input_text)

# --- Tone selection ---
tone = st.selectbox("🎭 Select narration tone", ["Neutral", "Suspenseful", "Inspiring"])

# --- Voice selection ---
voice_name = st.selectbox("🗣️ Select voice", list(TTS_VOICES.keys()))
voice_code = TTS_VOICES[voice_name]

# --- Generate button ---
if st.button("✨ Generate Narration"):
    if not input_text.strip():
        st.warning("Please upload or paste some text to continue.")
    else:
        with st.spinner("🪄 Rewriting text..."):
            rewritten_text = generate_tone_adaptive_text(input_text, tone)

        # Show side-by-side comparison
        st.subheader("Side-by-Side Text Comparison")
        left, right = st.columns(2)
        left.markdown("**Original Text:**")
        left.write(input_text)
        right.markdown(f"**{tone} Tone Adapted Text:**")
        right.write(rewritten_text)

        st.session_state["rewritten_text"] = rewritten_text

        # Convert to speech
        with st.spinner("🔊 Synthesizing speech..."):
            audio_bytes_io = ibm_tts(rewritten_text, voice_code)

        if audio_bytes_io:
            audio_bytes = audio_bytes_io.read()
            st.audio(audio_bytes, format="audio/mp3")

            st.download_button(
                label="⬇️ Download Narration (MP3)",
                data=audio_bytes,
                file_name="echoverse_narration.mp3",
                mime="audio/mp3"
            )
