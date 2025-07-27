import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="AI File Assistant", layout="centered", initial_sidebar_state="auto")

# Inject custom CSS for dark theme and glowing UI
st.markdown("""
    <style>
    /* Overall background container with blur animation */
    .animated-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        overflow: hidden;
        background: radial-gradient(circle at 20% 20%, #00ffff33, transparent 60%),
                    radial-gradient(circle at 80% 30%, #ff00ff33, transparent 60%),
                    radial-gradient(circle at 50% 80%, #00ff9933, transparent 60%);
        animation: moveBlobs 20s infinite ease-in-out;
        filter: blur(60px);
    }

    @keyframes moveBlobs {
        0% {
            background-position: 20% 20%, 80% 30%, 50% 80%;
        }
        50% {
            background-position: 30% 30%, 70% 40%, 60% 70%;
        }
        100% {
            background-position: 20% 20%, 80% 30%, 50% 80%;
        }
    }

    body {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stApp {
        background-color: transparent !important;
    }

    .block-container {
        padding: 2rem;
    }

    .modern-header {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Glowing box style */
    .glow-box {
        border: 2px solid #00ffff;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1.5rem;
        box-shadow: 0 0 20px #00ffff55;
        background-color: #1e1e1e;
    }

    /* Glowing prompt text area */
    textarea {
        background-color: #121212 !important;
        color: #FFFFFF !important;
        border: 2px solid #00ffff !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px #00ffff55 !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }

    textarea::placeholder {
        color: #AAAAAA !important;
    }

    /* File uploader glowing style */
    section[data-testid="stFileUploader"] {
        border: 2px solid #00ffff;
        padding: 1rem;
        border-radius: 12px;
        background-color: #1e1e1e;
        box-shadow: 0 0 20px #00ffff55;
        margin-top: 1.5rem;
    }

    /* Sidebar transparent and blurred style */
    [data-testid="stSidebar"] {
        background-color: rgba(18, 18, 18, 0.5) !important; /* Semi-transparent black */
        backdrop-filter: blur(10px); /* Adds a blur effect */
        box-shadow: none !important; /* Removes default shadow */
    }

    /* Sidebar content style */
    [data-testid="stSidebar"] > div {
        background-color: transparent !important;
    }
    </style>
    <div class="animated-bg"></div>
""", unsafe_allow_html=True)

# Title with updated style
st.markdown('<h1 class="modern-header">AI Report Making Assistant</h1>', unsafe_allow_html=True)

# Initialize session state for prompt history if not already present
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

# Sidebar for accessing previous prompts
st.sidebar.title("🧾 Prompt History")
if st.session_state.prompt_history:
    for idx, p in enumerate(reversed(st.session_state.prompt_history[-10:])):
        if st.sidebar.button(f"{p}", key=f"load_{idx}"):
            st.session_state.active_prompt = p
else:
    st.sidebar.markdown("_No previous prompts yet._")

# Spacing for aesthetics
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

# Spacing for aesthetics
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Prompt Input
prompt = st.text_area("Enter your prompt related to the uploaded file", placeholder="Type your prompt here...")

# Button to send the prompt
if st.button("Submit Prompt"):
    if not prompt:
        st.error("Please enter a prompt before submitting.")
    else:
        # Add the prompt to the history
        st.session_state.prompt_history.append(prompt)
        st.success("Prompt submitted successfully! 🚀")
        st.markdown("<br>", unsafe_allow_html=True)

# If both file and prompt are provided
if uploaded_file and prompt:
    st.markdown('<div class="glow-box">', unsafe_allow_html=True)

    try:
        # Read uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Show file preview
        st.subheader("📄 File Preview:")
        st.dataframe(df.head(), use_container_width=True)

        # Show prompt
        st.subheader("🧠 Prompt Response:")
        st.markdown(f"**Your Prompt:** `{prompt}`")
        st.markdown("🚧 *(You can implement prompt-based processing here)*")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        
    st.markdown('</div>', unsafe_allow_html=True)
