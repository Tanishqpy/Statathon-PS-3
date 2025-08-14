import streamlit as st
import pandas as pd
import requests
import json
import time
import io
import base64
from datetime import datetime
import uuid


# --- CONFIG ---
BACKEND_URL = "http://127.0.0.1:8000/process"

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
if "active_prompt" not in st.session_state:
    st.session_state.active_prompt = ""
if "just_submitted" not in st.session_state:
    st.session_state.just_submitted = False
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "download_clicked" not in st.session_state:
    st.session_state.download_clicked = False
if "state_management" not in st.session_state:
    st.session_state.state_management = {
        "last_submitted_prompt": None,
        "history_updated": False
    }
if "processed_data_cache" not in st.session_state:
    st.session_state.processed_data_cache = {}

# Define PDF download button function before using it
def add_pdf_download_button(task_id):
    """Adds a download button for the PDF report."""
    if task_id:
        pdf_url = f"{BACKEND_URL.replace('/process', '')}/download/{task_id}/pdf"
        
        # Create a button that triggers the download
        st.markdown("### 📄 Download Statistical Report as PDF")
        
        # Create HTML for a custom styled button
        button_html = f'''
        <a href="{pdf_url}" target="_blank" style="text-decoration: none;">
            <div style="
                background-color: #ff4b4b;
                color: white;
                padding: 14px 20px;
                margin: 8px 0;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                font-size: 16px;
                text-align: center;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                box-shadow: 0 0 15px #ff4b4b55;
            ">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white" style="margin-right: 8px;">
                    <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
                </svg>
                Download PDF Report
            </div>
        </a>
        '''
        
        st.markdown(button_html, unsafe_allow_html=True)
        # Also add a direct URL for users who prefer to copy/paste
        st.markdown(f"Or use this direct link: [Statistical Report PDF]({pdf_url})")

# Check if we're in download mode
query_params = st.query_params
if "download" in query_params:
    download_id = query_params.get("download", [""])[0]
    download_format = query_params.get("format", ["csv"])[0]
    
    if download_id and download_id in st.session_state.processed_data_cache:
        # Handle download without affecting history
        data = st.session_state.processed_data_cache[download_id]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        
        if download_format == "csv":
            csv_data = df.to_csv(index=False)
            st.download_button(
                "Click again to download CSV",
                data=csv_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
            st.markdown("Download should start automatically. If not, click the button above.")
            
            # Auto-download script
            st.markdown("""
            <script>
                document.querySelector('[data-testid="stDownloadButton"]').click();
                setTimeout(() => { window.history.back(); }, 1000);
            </script>
            """, unsafe_allow_html=True)
            
        elif download_format == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='ProcessedData')
            excel_data = output.getvalue()
            
            st.download_button(
                "Click again to download Excel",
                data=excel_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.markdown("Download should start automatically. If not, click the button above.")
            
            # Auto-download script
            st.markdown("""
            <script>
                document.querySelector('[data-testid="stDownloadButton"]').click();
                setTimeout(() => { window.history.back(); }, 1000);
            </script>
            """, unsafe_allow_html=True)
            
        st.button("Return to Report", on_click=lambda: st.query_params.clear())
        
    else:
        st.error("Download data not found. Please return to the main page and try again.")
        if st.button("Return to Main Page"):
            st.query_params.clear()
            st.rerun()
            
    # Stop execution for the download page
    st.stop()

# Sidebar for accessing previous prompts
st.sidebar.title("🧾 Prompt History")
if st.session_state.prompt_history:
    for idx, p in enumerate(reversed(st.session_state.prompt_history[-10:])):
        if st.sidebar.button(f"{p}", key=f"load_{idx}"):
            st.session_state.active_prompt = p
else:
    st.sidebar.markdown("_No previous prompts yet._")

# Spacing for aesthetics
st.markdown("<br>" * 4, unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

# Spacing for aesthetics
st.markdown("<br>" * 2, unsafe_allow_html=True)

# Prompt Input
prompt = st.text_area(
    "Enter your prompt related to the uploaded file",
    placeholder="Type your prompt here...",
    key="active_prompt"
)

if st.button("Submit Prompt"):
    if not uploaded_file:
        st.error("Please upload a file before submitting.")
    elif not prompt:
        st.error("Please enter a prompt before submitting.")
    else:
        # Store the current prompt for history tracking
        current_prompt = prompt
        
        # Only add to history if this is a new submission (not a rerun due to download)
        if (current_prompt != st.session_state.state_management["last_submitted_prompt"] or 
            not st.session_state.state_management["history_updated"]):
            
            # Add to history if not already there
            if current_prompt not in st.session_state.prompt_history:
                st.session_state.prompt_history.append(current_prompt)
            
            # Mark that we've updated history for this prompt
            st.session_state.state_management["last_submitted_prompt"] = current_prompt
            st.session_state.state_management["history_updated"] = True
        
        with st.spinner("🚀 Sending data to the AI for processing..."):
            try:
                # Prepare payload for the backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"prompt": prompt}

                # Send request to FastAPI backend to start processing
                response = requests.post(BACKEND_URL, files=files, data=data)

                if response.status_code == 200:
                    task_data = response.json()
                    task_id = task_data["task_id"]
                    
                    # Create a container with fixed height for logs
                    st.markdown("""
                    <style>
                    .fixed-height-container {
                        height: 400px;
                        overflow-y: auto;
                        border: 2px solid #00ffff;
                        border-radius: 10px;
                        padding: 10px;
                        background-color: #121212;
                        margin-bottom: 20px;
                        box-shadow: 0 0 15px #00ffff55;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Create status header and log container
                    status_container = st.empty()
                    log_container = st.empty()  # Changed to st.empty() so we can clear it later
                    
                    # Poll for updates
                    status = "processing"
                    log_content = ""
                    while status == "processing":
                        # Update status message
                        status_container.markdown("⚙️ **Processing:** AI is analyzing your data...")
                        
                        # Get status update
                        status_response = requests.get(f"{BACKEND_URL.replace('/process', '')}/status/{task_id}")
                        status_data = status_response.json()
                        
                        # Update logs in fixed-height container
                        log_content = "\n".join(status_data["logs"])
                        log_container.markdown(f"""
                        <div class="fixed-height-container">
                            <pre style="color: #00ffff; margin: 0;">{log_content}</pre>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Check if processing is done
                        status = status_data["status"]
                        
                        if status == "processing":
                            time.sleep(1)  # Wait a second before next poll
                    
                    # Display the results in a glowing box with report and download options
                    if status == "completed":
                        # Clear the real-time log display to avoid duplication
                        log_container.empty()
                        
                        # Set the processing_complete flag to prevent session reset
                        st.session_state.processing_complete = True
                        
                        status_container.success("✅ Processing complete!")
                        
                        # Display execution time if available
                        if "execution_time" in status_data:
                            st.info(f"⏱️ Processing time: {status_data['execution_time']}")
                        
                        # Display the logs in a collapsible section
                        with st.expander("📋 Processing Logs", expanded=False):
                            st.markdown(f"""
                            <div class="fixed-height-container">
                                <pre style="color: #00ffff; margin: 0;">{log_content}</pre>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display the AI-generated report prominently
                        st.markdown('<div class="glow-box">', unsafe_allow_html=True)
                        st.subheader("🧠 AI Report:")
                        
                        if "report" in status_data.get("summary", {}):
                            st.markdown(status_data["summary"]["report"])
                        else:
                            st.write("No report was generated.")
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add options to view detailed statistics
                        with st.expander("📊 Detailed Statistics", expanded=False):
                            # Show summary sections in collapsible expanders
                            if status_data.get("summary", {}):
                                summary_data = status_data["summary"]
                                
                                # Define sections to exclude from display
                                excluded_sections = ["narrator_summary", "narration", "prompt_used", "report", "processed_data"]
                                
                                # Organize sections into expandable components
                                for section, content in summary_data.items():
                                    # Skip excluded sections
                                    if section in excluded_sections:
                                        continue
                                        
                                    # Format section name for display
                                    section_display = section.replace('_', ' ').title()
                                    
                                    # Display section content
                                    with st.expander(f"{section_display}", expanded=False):
                                        try:
                                            if isinstance(content, (dict, list)):
                                                st.json(content)
                                            else:
                                                st.code(str(content))
                                        except Exception as e:
                                            st.error(f"Error displaying content: {str(e)}")
                        
                        # Add download buttons for processed data
                        if "processed_data" in status_data.get("summary", {}):
                            st.subheader("📥 Download Processed Data")
                            
                            # Create two columns for the download buttons
                            col1, col2 = st.columns(2)
                            
                            # Backend download URLs
                            csv_url = f"{BACKEND_URL.replace('/process', '')}/download/{task_id}/csv"
                            excel_url = f"{BACKEND_URL.replace('/process', '')}/download/{task_id}/excel"
                            
                            # Create download buttons with direct links to FastAPI
                            with col1:
                                st.markdown(f'''
                                <a href="{csv_url}" target="_blank">
                                    <button style="
                                        background-color: #4CAF50;
                                        border: none;
                                        color: white;
                                        padding: 15px 32px;
                                        text-align: center;
                                        text-decoration: none;
                                        display: inline-block;
                                        font-size: 16px;
                                        margin: 4px 2px;
                                        cursor: pointer;
                                        border-radius: 12px;
                                        width: 100%;
                                    ">
                                        Download as CSV
                                    </button>
                                </a>
                                ''', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f'''
                                <a href="{excel_url}" target="_blank">
                                    <button style="
                                        background-color: #008CBA;
                                        border: none;
                                        color: white;
                                        padding: 15px 32px;
                                        text-align: center;
                                        text-decoration: none;
                                        display: inline-block;
                                        font-size: 16px;
                                        margin: 4px 2px;
                                        cursor: pointer;
                                        border-radius: 12px;
                                        width: 100%;
                                    ">
                                        Download as Excel
                                    </button>
                                </a>
                                ''', unsafe_allow_html=True)
                            
                            # Add PDF report download button
                            st.markdown("<br>", unsafe_allow_html=True)
                            add_pdf_download_button(task_id)
                        
                        # Add a button to clear results and prepare for a new analysis
                        if "processed_data" in status_data.get("summary", {}):
                            if st.button("Start New Analysis"):
                                # Clear cached data
                                for key in ["csv_data", "excel_data"]:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                
                                # Reset all session state flags
                                st.session_state.just_submitted = False
                                st.session_state.processing_complete = False
                                st.session_state.download_clicked = False
                                
                                # Rerun the app to refresh
                                st.rerun()
                    else:
                        # Processing failed
                        status_container.error(f"❌ Error during processing")

                        # Show logs for debugging
                        with st.expander("📋 Error Logs", expanded=True):
                            st.markdown(f"""
                            <div class="fixed-height-container">
                                <pre style="color: #ff5555; margin: 0;">{log_content}</pre>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error(f"Error from server: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    # Reset the submitted flag only when the submit button is not pressed AND
    # we're not in a download state AND we're not displaying results
    if (not st.session_state.get("processing_complete", False) and 
        not st.session_state.get("download_clicked", False)):
        st.session_state.just_submitted = False
    
    # Reset download flag after handling it
    if st.session_state.get("download_clicked", False):
        st.session_state.download_clicked = False
