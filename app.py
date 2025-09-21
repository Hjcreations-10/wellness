import streamlit as st
import tempfile
import os
import random
import time
from streamlit_mic_recorder import mic_recorder
from PIL import Image, ImageDraw, ImageFont

# Import Google Cloud client libraries
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import dialogflow_v2beta1 as dialogflow
from google.cloud import texttospeech_v1 as tts
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# --- PAGE CONFIG & SESSION STATE INIT ---
st.set_page_config(page_title="Journey to Wellness Prototype", layout="centered")
for key, default_value in {
    "page": "home",
    "credits": 0,
    "user_input": "",
    "user_original": "",
    "chat_history": [],
    "moods": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- GOOGLE CLOUD CONFIGURATION ---
# Replace with your Google Cloud Project details
PROJECT_ID = "your-gcp-project-id"
DIALOGFLOW_SESSION_ID = "streamlit-session"
LOCATION = "us-central1"

# Initialize Google Cloud clients
speech_client = speech.SpeechClient()
dialogflow_client = dialogflow.SessionsClient()
tts_client = tts.TextToSpeechClient()
vertexai.init(project=PROJECT_ID, location=LOCATION)
generative_model = GenerativeModel("gemini-pro")

# --- UTILITY & MODEL FUNCTIONS ---
def transcribe_audio_gcp(audio_path):
    """Transcribes audio using Google's Speech-to-Text API."""
    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    
    response = speech_client.recognize(config=config, audio=audio)
    if response.results:
        return response.results[0].alternatives[0].transcript
    return ""

def get_sentiment_vertex_ai(text: str):
    """Performs sentiment analysis using Vertex AI."""
    # This is a placeholder for a more advanced NLP model on Vertex AI.
    # For a prototype, you could use a simple sentiment analysis model.
    # The current code uses a keyword-based approach as a fallback.
    return get_simple_sentiment(text)

def get_dialogflow_response(user_input: str) -> str:
    """Gets a response from a Dialogflow agent."""
    session_path = dialogflow_client.session_path(PROJECT_ID, DIALOGFLOW_SESSION_ID)
    text_input = dialogflow.TextInput(text=user_input, language_code="en-US")
    query_input = dialogflow.QueryInput(text=text_input)
    
    try:
        response = dialogflow_client.detect_intent(
            session=session_path,
            query_input=query_input
        )
        return response.query_result.fulfillment_text
    except Exception as e:
        st.error(f"Dialogflow error: {e}")
        return get_simple_bot_response(user_input)

def speak_text_tts_gcp(text: str):
    """Converts text to speech using Google's Cloud TTS API."""
    input_text = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=tts.SsmlVoiceGender.NEUTRAL
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
    
    response = tts_client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config
    )
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    with open(temp_file.name, "wb") as f:
        f.write(response.audio_content)
    return temp_file.name

def create_mansion_story_vertex_ai(user_text: str) -> str:
    """Generates a personalized story using Vertex AI."""
    prompt = f"Write a short, inspiring hero's journey story based on the user's reflection: '{user_text}'. The story should focus on resilience and triumph, ending with them reaching a 'Mansion of Inner Peace'."
    response = generative_model.generate_content(prompt)
    return response.text

def create_image_from_text(story_text: str):
    """Generates an image from text using the Pillow library."""
    # This function remains as a simple, no-API fallback for video generation.
    # For a full implementation, you would use a Cloud Video API.
    width, height = 1280, 720
    bg_color = (18, 24, 37)
    text_color = "white"
    
    img = Image.new('RGB', (width, height), color=bg_color)
    d = ImageDraw.Draw(img)
    
    try:
        font_path = "arial.ttf"
        font = ImageFont.truetype(font_path, 30)
    except IOError:
        font = ImageFont.load_default()
    
    lines = []
    max_width = width - 100
    words = story_text.split(' ')
    current_line = ""
    for word in words:
        if d.textlength(current_line + ' ' + word, font=font) < max_width:
            current_line += ' ' + word
        else:
            lines.append(current_line.strip())
            current_line = word
    lines.append(current_line.strip())
    story_wrapped = "\n".join(lines)
    
    text_width, text_height = d.textlength(story_wrapped, font=font), d.textlength(story_wrapped, font=font)
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    d.text((x, y), story_wrapped, font=font, fill=text_color)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(temp_file.name)
    return temp_file.name

# --- PAGE COMPONENTS ---
def reflection_box():
    """Allows users to input text or record audio for reflection."""
    st.subheader("💬 Reflect — how are you feeling?")
    typed_text = st.text_area("Write your thoughts:", value=st.session_state.get("user_input", ""), height=150)
    audio_data = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", just_once=False, key="mic_recorder")

    reflection_text = typed_text.strip()
    
    if audio_data and audio_data.get("bytes"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_data["bytes"])
            audio_path = f.name
        
        transcribed_text = transcribe_audio_gcp(audio_path)
        os.remove(audio_path)
        
        if transcribed_text:
            st.success("🎙️ Transcribed: " + transcribed_text)
            reflection_text = transcribed_text
            st.session_state.user_input = reflection_text
            st.session_state.user_original = reflection_text
            st.experimental_rerun()

    if typed_text != st.session_state.user_input and typed_text:
        st.session_state.user_input = typed_text
        st.session_state.user_original = typed_text
        st.experimental_rerun()

def chatbot_in_game():
    """The main chatbot interface for the game pages."""
    st.subheader("🧠 Chat with the Wellness Bot")
    reflection_box()

    st.markdown("### Conversation")
    for sender, msg in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f'<div style="background:#DCF8C6;padding:8px;border-radius:10px;margin:6px">🗣️ {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:#F1F0F0;padding:8px;border-radius:10px;margin:6px">🤖 {msg}</div>', unsafe_allow_html=True)

    if st.button("➡ Send to Bot"):
        user_input = st.session_state.get("user_input", "").strip()
        if len(user_input) < 5:
            st.warning("Please share a bit more before sending.")
        else:
            sentiment = get_sentiment_vertex_ai(user_input)
            reply = get_dialogflow_response(user_input)
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("bot", reply))
            st.session_state.moods.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "text": user_input, "sentiment": sentiment
            })
            st.session_state.user_input = ""
            st.experimental_rerun()
            
    if st.session_state.moods:
        st.markdown("### Mood Tracker")
        for m in reversed(st.session_state.moods[-5:]):
            st.write(f"**{m['sentiment']}** — {m['time']}")
            st.caption(m["text"])

    if st.session_state.credits >= 3:
        if st.button("➡ Continue to Mansion"):
            st.session_state.page = "mansion"
            st.experimental_rerun()
    else:
        st.info(f"✨ Earn at least 3 credits from games to unlock your Mansion story. (Current: {st.session_state.credits})")

# --- GAME PAGE LOGIC ---
def train_game():
    st.header("🚆 Train: Object Hunt")
    st.write("Find the hidden objects using hints.")
    st.image("https://gamersunite.s3.amazonaws.com/uploads/june-s-journey-hidden-object-mystery-game/1531636021150", width='stretch')
    objects = {"butterfly": "A small creature with wings", "wheel": "Round part found on vehicles", "horseshoe": "Lucky, horse related"}
    hidden_objects = random.sample(list(objects.keys()), 3)
    
    with st.form("train_form"):
        for i, obj in enumerate(hidden_objects):
            st.write(f"Hint {i+1}: {objects[obj]}")
            st.text_input(f"Guess for Hint {i+1}:", key=f"train_guess_{i}")
        submitted = st.form_submit_button("Check Answers ✅")

    if submitted:
        score = sum(1 for i, obj in enumerate(hidden_objects) if st.session_state[f"train_guess_{i}"].strip().lower() == obj)
        st.session_state.credits += score
        st.success(f"Found {score}/3 objects. Total credits: {st.session_state.credits}")
        st.experimental_rerun()
    chatbot_in_game()

def car_game():
    st.header("🚗 Car: Puzzle Rush")
    puzzles = [("What has keys but can’t open locks?", "piano"), ("I’m tall when I’m young, and short when I’m old. What am I?", "candle")]
    question, answer = random.choice(puzzles)
    
    with st.form("car_form"):
        ans = st.text_input(question)
        submitted = st.form_submit_button("Submit Answer ✅")
        
    if submitted:
        if ans.strip().lower() == answer:
            st.session_state.credits += 1
            st.success("Correct! 🎉")
        else:
            st.error(f"Wrong — the answer was: {answer}")
        st.info(f"Credits: {st.session_state.credits}")
        st.experimental_rerun()
    chatbot_in_game()

def bus_game():
    st.header("🚌 Bus: Word Hunt")
    st.markdown("Find the missing words (bus stops). You can ask for hints if you need them!")
    st.image("https://puzzlestoplay.com/wp-content/uploads/2021/01/bus-driver-word-search-puzzle-photo-506x675.jpg", width='stretch')
    words = {"route": "From bottom, line 6", "driver": "Top, line 3", "ticket": "Center, line 8"}

    if st.button("💡 Show Hints"):
        for w, h in words.items():
            st.info(f"Hint: {h}")

    with st.form("bus_form"):
        for i, word in enumerate(words.keys()):
            st.text_input(f"Guess #{i+1}:", key=f"bus_guess_{i}")
        submitted = st.form_submit_button("Check Answers ✅")

    if submitted:
        score = sum(1 for i, word in enumerate(words.keys()) if st.session_state[f"bus_guess_{i}"].strip().lower() == word)
        st.session_state.credits += score
        st.success(f"Found {score}/{len(words)} — Total credits: {st.session_state.credits}")
        st.experimental_rerun()
    chatbot_in_game()

# --- MANSION PAGE ---
def mansion_page():
    st.title("🏰 Mansion — Your Success Story")
    st.image("https://img.freepik.com/premium-photo/free-photo-luxurious-mansion-with-lush-garden-beautiful-sunset-background_650144-557.jpg?w=2000", width='stretch')
    user_text = st.session_state.get("user_original", "")
    
    if not user_text:
        st.warning("No reflection found from your journey.")
    else:
        if st.button("🏰 Generate Story Image"):
            story = create_mansion_story_vertex_ai(user_text)
            with st.spinner("Rendering your story image..."):
                image_path = create_image_from_text(story)
                st.image(image_path)
                st.success("Your story is complete! 🌟")
                os.remove(image_path)
    if st.button("🔁 Restart Journey"):
        st.session_state.clear()
        st.experimental_rerun()

# --- HOME PAGE ---
def home_page():
    st.title("🚦 Start Your Journey")
    st.markdown("Choose your mode of travel:")
    st.image("https://wallpaperaccess.com/full/4515512.jpg", width='stretch')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚆 Train"):
            st.session_state.page = "train"
            st.experimental_rerun()
    with col2:
        if st.button("🚌 Bus"):
            st.session_state.page = "bus"
            st.experimental_rerun()
    with col3:
        if st.button("🚗 Car"):
            st.session_state.page = "car"
            st.experimental_rerun()

# --- ROUTER ---
pages = {
    "home": home_page,
    "train": train_game,
    "car": car_game,
    "bus": bus_game,
    "mansion": mansion_page,
}
pages[st.session_state.page]()



