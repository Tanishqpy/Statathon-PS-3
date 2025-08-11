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

# Inject custom CSS for styling
st.markdown("""
    <style>
    /* Header styling - match original design */
    header[data-testid="stHeader"] {
        background-color: #24344D!important;
        background-image: none !important;
        box-shadow: none !important;
        height: 3rem !important;
    }

    /* Remove colored strip above header */
    [data-testid="stAppViewContainer"] > header::before {
        content: none !important;
    }

    /* Header buttons visibility - white for dark header */
    header[data-testid="stHeader"] button, 
    header[data-testid="stHeader"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* Page background */
    [data-testid="stAppViewContainer"] {
        background-color:#e8edf3;
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }

    /* Sidebar positioning - over header */
    [data-testid="stSidebar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        height: 100% !important;
        z-index: 999999 !important;
        background-color: rgba(248, 249, 250, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1) !important;
    }

    /* Sidebar content styling */
    [data-testid="stSidebar"] > div {
        background-color: transparent !important;
        backdrop-filter: none !important;
        box-shadow: none !important;
    }

    /* Main content styling */
    .block-container {
        padding: 2rem;
    }

    .modern-header {
        background: linear-gradient(135deg, #377bf2, #723beb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Enhanced file uploader styling */
    section[data-testid="stFileUploader"] {
        border: 3px dashed #4CAF50;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    section[data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }

    /* Text area styling */
    textarea {
        background-color: #475569 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 1rem !important;
        font-size: 1rem !important;
    }

    /* Placeholder text */
    textarea::placeholder {
        color: #cbd5e1 !important;
    }

    /* File uploader button */
    section[data-testid="stFileUploader"] div[role="button"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        border: none !important;
    }

    section[data-testid="stFileUploader"] div[role="button"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Fixed height container for logs */
    .fixed-height-container {
        height: 400px;
        overflow-y: auto;
        border: 2px solid #24344D;
        border-radius: 10px;
        padding: 10px;
        background-color: #FFFFFF;
        margin-bottom: 20px;
    }

    /* File info cards */
    .file-info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    /* Success message styling */
    .success-message {
        background: linear-gradient(90deg, #00d4aa 0%, #00b894 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }

    /* Upload zone styling */
    .upload-zone {
        border: 3px dashed #4CAF50;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        margin: 20px 0;
        transition: all 0.3s ease;
    }

    .upload-zone:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-color: #667eea;
    }
    
    </style>
""", unsafe_allow_html=True)

# Title with updated style
st.markdown('<h1 class="modern-header" style="padding-top:40px; font-weight:bold;">AI Report Making Assistant</h1>',
            unsafe_allow_html=True)
st.markdown("""
<p style="padding-top:0px; color:#64748b; text-align:center;">
    Transform your data into comprehensive reports with the power of AI.<br>
    Upload your files and let our assistant create detailed insights.
</p>
""", unsafe_allow_html=True)
#generate ai report button
st.markdown("""
<style>
/* Target the primary generate button by its label text */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #377bf2, #723beb) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.8rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    box-shadow: 0 4px 15px rgba(55, 123, 242, 0.4) !important;
    transition: all 0.3s ease !important;
}

/* Hover effect */
div.stButton > button:first-child:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(55, 123, 242, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)

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
if "uploaded_file_info" not in st.session_state:
    st.session_state.uploaded_file_info = None

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
st.sidebar.title("📊 Prompt History")
if st.session_state.prompt_history:
    for idx, p in enumerate(reversed(st.session_state.prompt_history[-10:])):
        if st.sidebar.button(f"{p[:50]}..." if len(p) > 50 else f"{p}", key=f"load_{idx}"):
            st.session_state.active_prompt = p
else:
    st.sidebar.markdown("_No previous prompts yet._")

# Add file upload statistics in sidebar if file is uploaded
if st.session_state.uploaded_file_info:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Current File")
    info = st.session_state.uploaded_file_info
    st.sidebar.markdown(f"**Name:** {info['name']}")
    st.sidebar.markdown(f"**Size:** {info['size']}")
    st.sidebar.markdown(f"**Type:** {info['type']}")
    if 'rows' in info:
        st.sidebar.markdown(f"**Rows:** {info['rows']:,}")
        st.sidebar.markdown(f"**Columns:** {info['columns']}")

# Spacing for aesthetics
st.markdown("<br>", unsafe_allow_html=True)

# File validation function
def validate_file(file):
    """Validate uploaded file"""
    errors = []
    warnings = []
    
    # Size check (100MB limit)
    if file.size > 100 * 1024 * 1024:
        errors.append("File size exceeds 100MB limit")
    elif file.size > 50 * 1024 * 1024:
        warnings.append("Large file detected. Processing may take longer.")
    
    # Content validation
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, nrows=5)  # Read only first 5 rows for validation
            file.seek(0)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file, nrows=5)
            file.seek(0)
        else:
            errors.append("Unsupported file format")
            return errors, warnings
        
        if df.empty:
            errors.append("File appears to be empty")
        elif df.shape[1] < 1:
            errors.append("File must have at least 1 column")
        
    except Exception as e:
        errors.append(f"File format error: {str(e)[:100]}...")
    
    return errors, warnings

# Enhanced File Upload Section with Working Drag & Drop
st.markdown("""
<style>
/* Enhanced drag and drop styling */
.stFileUploader > div > div {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
    border: 3px dashed #4CAF50 !important;
    border-radius: 20px !important;
    padding: 40px !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
}

.stFileUploader > div > div:hover {
    border-color: #667eea !important;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
}

/* Style the drag text */
.stFileUploader > div > div > div {
    color: #667eea !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}

/* Style the browse button */
.stFileUploader button {
    background: linear-gradient(135deg, #377bf2, #723beb) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 12px 30px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    margin-top: 15px !important;
}

.stFileUploader button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Hide the default upload icon */
.stFileUploader > div > div > svg {
    display: none !important;
}

/* Custom upload area content */
.upload-header {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    text-align: center;
    border: 1px solid rgba(102, 126, 234, 0.2);
}
</style>

<div class="upload-header">
    <h3 style="color: #667eea; margin: 0 0 10px 0; font-size: 1.5rem;">🚀 Upload Your Data File</h3>
    <p style="color: #6b7280; margin: 0 0 10px 0;">Drag and drop your file or click browse button below</p>
    <div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-top: 10px;">
        <span style="background: #4CAF50; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;">📊 CSV</span>
        <span style="background: #2196F3; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;">📈 Excel</span>
    </div>
    <p style="color: #9ca3af; font-size: 0.85rem; margin: 10px 0 0 0;">Maximum file size: 100MB</p>
</div>
""", unsafe_allow_html=True)

# Enhanced file uploader that actually supports drag & drop
uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=['csv', 'xlsx',],
    help="💡 You can drag files directly into the area above or click 'Browse files' button",
    accept_multiple_files=False
)

# Alternative: URL upload option
with st.expander("🌐 Or load data from URL", expanded=False):
    url_input = st.text_input(
        "Enter file URL",
        placeholder="https://example.com/data.csv",
        help="Direct link to CSV, Excel, or JSON file"
    )
    
    if url_input and st.button("📥 Load from URL"):
        try:
            with st.spinner("🔄 Loading data from URL..."):
                if url_input.endswith('.csv'):
                    df_preview = pd.read_csv(url_input)
                    st.session_state['url_data'] = url_input
                    st.session_state['url_df'] = df_preview
                elif url_input.endswith(('.xlsx', '.xls')):
                    df_preview = pd.read_excel(url_input)
                    st.session_state['url_data'] = url_input
                    st.session_state['url_df'] = df_preview
                elif url_input.endswith('.json'):
                    df_preview = pd.read_json(url_input)
                    st.session_state['url_data'] = url_input
                    st.session_state['url_df'] = df_preview
                else:
                    st.error("❌ Unsupported URL format. Please use CSV, Excel, or JSON.")
                    st.stop()
            
            st.success(f"✅ Data loaded successfully from URL!")
            
            # Show preview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Rows", f"{df_preview.shape[0]:,}")
            with col2:
                st.metric("📈 Columns", df_preview.shape[1])
            with col3:
                st.metric("🔍 Preview", "Ready")
            
            with st.expander("👁️ Data Preview", expanded=False):
                st.dataframe(df_preview.head(10))
                
        except Exception as e:
            st.error(f"❌ Error loading from URL: {str(e)}")

# Process uploaded file
current_file = None
if uploaded_file is not None:
    current_file = uploaded_file
    file_source = "upload"
elif 'url_data' in st.session_state:
    current_file = st.session_state['url_data']
    file_source = "url"

if current_file is not None:
    if file_source == "upload":
        # Validate the uploaded file
        errors, warnings = validate_file(uploaded_file)
        
        # Show validation results
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
            st.stop()
        
        if warnings:
            for warning in warnings:
                st.warning(f"⚠️ {warning}")
        
        # File successfully validated
        st.markdown('<div class="success-message">✅ File uploaded and validated successfully!</div>', 
                   unsafe_allow_html=True)
        
        # Load and analyze the file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.stop()
            
        # Store file info in session state
        st.session_state.uploaded_file_info = {
            'name': uploaded_file.name,
            'size': f"{uploaded_file.size/1024:.1f} KB" if uploaded_file.size < 1024*1024 else f"{uploaded_file.size/(1024*1024):.1f} MB",
            'type': uploaded_file.type,
            'rows': df.shape[0],
            'columns': df.shape[1]
        }
        
    else:  # URL source
        df = st.session_state['url_df']
        st.session_state.uploaded_file_info = {
            'name': current_file.split('/')[-1],
            'size': f"{len(str(df))/1024:.1f} KB",
            'type': 'URL',
            'rows': df.shape[0],
            'columns': df.shape[1]
        }
    
    # Display file statistics in an attractive format
    st.markdown("### 📋 File Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("📈 Columns", df.shape[1])
    with col3:
        st.metric("❓ Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        memory_usage = df.memory_usage(deep=True).sum()
        if memory_usage < 1024*1024:
            memory_str = f"{memory_usage/1024:.1f} KB"
        else:
            memory_str = f"{memory_usage/(1024*1024):.1f} MB"
        st.metric("💾 Memory Usage", memory_str)
    
    # Data preview in expandable section
    with st.expander("👁️ Data Preview & Column Information", expanded=False):
        tab1, tab2 = st.tabs(["🔍 Data Preview", "📊 Column Info"])
        
        with tab1:
            st.markdown("**First 10 rows:**")
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            # Column information
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)

# Spacing for aesthetics
st.markdown("<br>" * 2, unsafe_allow_html=True)

# Prompt Input with enhanced styling
st.markdown("### 💬 Analysis Prompt")
st.markdown("*Describe what insights or analysis you'd like to generate from your data*")

prompt = st.text_area(
    "Enter your prompt related to the uploaded file",
    placeholder="Example: 'Create a comprehensive analysis of sales trends, identify top performers, and provide actionable insights for improving revenue.'",
    key="active_prompt",
    height=100
)

# Enhanced Submit Button
if st.button("🚀 Generate AI Report", type="primary"):
    if current_file is None:
        st.error("📁 Please upload a file or load data from URL a before submitting.")
    elif not prompt:
        st.error("💬 Please enter a prompt before submitting.")
    else:
        # Store the current prompt for history tracking
        current_prompt = prompt

        # Only add to history if this is a new submission
        if (current_prompt != st.session_state.state_management["last_submitted_prompt"] or
                not st.session_state.state_management["history_updated"]):

            # Add to history if not already there
            if current_prompt not in st.session_state.prompt_history:
                st.session_state.prompt_history.append(current_prompt)

            # Mark that we've updated history for this prompt
            st.session_state.state_management["last_submitted_prompt"] = current_prompt
            st.session_state.state_management["history_updated"] = True

        with st.spinner("🤖 AI is analyzing your data and generating insights..."):
            try:
                # Prepare payload for the backend
                if file_source == "upload":
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                else:  # URL source
                    # For URL, we need to send the URL as metadata
                    files = {"file": ("url_data.csv", st.session_state['url_df'].to_csv(index=False).encode(), "text/csv")}
                
                data = {"prompt": prompt}

                # Send request to FastAPI backend to start processing
                response = requests.post(BACKEND_URL, files=files, data=data)

                if response.status_code == 200:
                    task_data = response.json()
                    task_id = task_data["task_id"]

                    # Create status header and log container
                    status_container = st.empty()
                    log_container = st.empty()

                    # Poll for updates
                    status = "processing"
                    log_content = ""
                    while status == "processing":
                        # Update status message with animated emoji
                        status_container.markdown("⚙️ **Processing:** AI is analyzing your data... 🔄")

                        # Get status update
                        status_response = requests.get(f"{BACKEND_URL.replace('/process', '')}/status/{task_id}")
                        status_data = status_response.json()

                        # Update logs in fixed-height container
                        log_content = "\n".join(status_data["logs"])
                        log_container.markdown(f"""
                        <div class="fixed-height-container">
                            <pre style="color: #00ffff; margin: 0; font-family: 'Courier New', monospace;">{log_content}</pre>
                        </div>
                        """, unsafe_allow_html=True)

                        # Check if processing is done
                        status = status_data["status"]

                        if status == "processing":
                            time.sleep(1)

                    # Display the results
                    if status == "completed":
                        # Clear the real-time log display
                        log_container.empty()

                        # Set the processing_complete flag
                        st.session_state.processing_complete = True

                        status_container.success("🎉 Analysis Complete! Your AI report is ready!")

                        # Show execution time
                        if "execution_time" in status_data:
                            st.info(f"⏱️ Processing completed in: **{status_data['execution_time']}**")

                        # Display processing logs in collapsible section
                        with st.expander("📋 Processing Logs", expanded=False):
                            st.markdown(f"""
                            <div class="fixed-height-container">
                                <pre style="color: #00ffff; margin: 0; font-family: 'Courier New', monospace;">{log_content}</pre>
                            </div>
                            """, unsafe_allow_html=True)

                        # Display the AI-generated report prominently
                        st.markdown("---")
                        st.markdown("## 🧠 AI-Generated Report")

                        if "report" in status_data.get("summary", {}):
                            # Create an attractive report container
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                       padding: 2rem; border-radius: 15px; border-left: 5px solid #667eea; 
                                       box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 1rem 0;">
                            """, unsafe_allow_html=True)
                            st.markdown(status_data["summary"]["report"])
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ No report was generated. Please check the processing logs.")

                        # Detailed Statistics Section
                        with st.expander("📊 Detailed Analytics & Statistics", expanded=False):
                            if status_data.get("summary", {}):
                                summary_data = status_data["summary"]

                                # Exclude certain sections from display
                                excluded_sections = ["narrator_summary", "narration", "prompt_used", "report", "processed_data"]

                                # Display each section
                                for section, content in summary_data.items():
                                    if section in excluded_sections:
                                        continue

                                    section_display = section.replace('_', ' ').title()

                                    with st.expander(f"📈 {section_display}", expanded=False):
                                        try:
                                            if isinstance(content, (dict, list)):
                                                st.json(content)
                                            else:
                                                st.code(str(content), language="text")
                                        except Exception as e:
                                            st.error(f"Error displaying content: {str(e)}")

                        # Download Section
                        if "processed_data" in status_data.get("summary", {}):
                            st.markdown("---")
                            st.markdown("## 📥 Download Your Results")

                            # Create download buttons with enhanced styling
                            col1, col2 = st.columns(2)

                            # Backend download URLs
                            csv_url = f"{BACKEND_URL.replace('/process', '')}/download/{task_id}/csv"
                            excel_url = f"{BACKEND_URL.replace('/process', '')}/download/{task_id}/excel"

                            with col1:
                                st.markdown(f'''
                                <a href="{csv_url}" target="_blank" style="text-decoration: none;">
                                    <div style="
                                        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                                        border: none;
                                        color: white;
                                        padding: 20px;
                                        text-align: center;
                                        border-radius: 15px;
                                        width: 100%;
                                        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
                                        transition: all 0.3s ease;
                                        cursor: pointer;
                                        margin: 10px 0;
                                    ">
                                        <h4 style="margin: 0; color: white;">📊 Download CSV</h4>
                                        <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9);">Spreadsheet format</p>
                                    </div>
                                </a>
                                ''', unsafe_allow_html=True)

                            with col2:
                                st.markdown(f'''
                                <a href="{excel_url}" target="_blank" style="text-decoration: none;">
                                    <div style="
                                        background: linear-gradient(135deg, #008CBA 0%, #007B9A 100%);
                                        border: none;
                                        color: white;
                                        padding: 20px;
                                        text-align: center;
                                        border-radius: 15px;
                                        width: 100%;
                                        box-shadow: 0 4px 15px rgba(0, 140, 186, 0.4);
                                        transition: all 0.3s ease;
                                        cursor: pointer;
                                        margin: 10px 0;
                                    ">
                                        <h4 style="margin: 0; color: white;">📈 Download Excel</h4>
                                        <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9);">Advanced formatting</p>
                                    </div>
                                </a>
                                ''', unsafe_allow_html=True)

                        # New Analysis Button
                        st.markdown("---")
                        if st.button("🔄 Start New Analysis", type="secondary"):
                            # Clear cached data and reset states
                            for key in ["csv_data", "excel_data", "url_data", "url_df", "uploaded_file_info"]:
                                if key in st.session_state:
                                    del st.session_state[key]

                            # Reset flags
                            st.session_state.just_submitted = False
                            st.session_state.processing_complete = False
                            st.session_state.download_clicked = False
                            st.session_state.active_prompt = ""

                            st.rerun()
                    else:
                        # Processing failed
                        status_container.error("❌ Processing failed. Please check the error logs below.")

                        with st.expander("🔍 Error Logs", expanded=True):
                            st.markdown(f"""
                            <div class="fixed-height-container">
                                <pre style="color: #ff5555; margin: 0; font-family: 'Courier New', monospace;">{log_content}</pre>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Server Error ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("🔌 **Connection Error:** Unable to connect to the AI backend server.")
                st.markdown("""
                **Troubleshooting steps:**
                1. Ensure the backend server is running on `http://127.0.0.1:8000`
                2. Check if the server is accessible
                3. Verify your network connection
                4. Contact support if the issue persists
                """)
            except requests.exceptions.RequestException as e:
                st.error(f"🌐 **Network Error:** {str(e)}")
            except Exception as e:
                st.error(f"⚠️ **Unexpected Error:** {str(e)}")
                st.markdown("Please try again or contact support if the problem continues.")
else:
    # Reset flags when not submitting
    if (not st.session_state.get("processing_complete", False) and
            not st.session_state.get("download_clicked", False)):
        st.session_state.just_submitted = False

    # Reset download flag
    if st.session_state.get("download_clicked", False):
        st.session_state.download_clicked = False

# Add footer with tips and information
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
           padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center;">
    <h4 style="color: #667eea; margin-bottom: 1rem;">💡 Pro Tips for Better Results</h4>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; text-align: left;">
        <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h5 style="color: #4CAF50; margin: 0 0 0.5rem 0;">📊 Data Quality</h5>
            <p style="margin: 0; color: #6b7280; font-size: 0.9rem;">Ensure your data is clean with proper column headers and consistent formatting for best analysis results.</p>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h5 style="color: #008CBA; margin: 0 0 0.5rem 0;">💬 Clear Prompts</h5>
            <p style="margin: 0; color: #6b7280; font-size: 0.9rem;">Be specific about the insights you want. Mention key metrics, timeframes, and analysis types.</p>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h5 style="color: #FF9800; margin: 0 0 0.5rem 0;">🔍 File Size</h5>
            <p style="margin: 0; color: #6b7280; font-size: 0.9rem;">For faster processing, keep files under 100MB. Larger files are supported but may take longer.</p>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h5 style="color: #008CBA; margin: 0 0 0.5rem 0;">⏱️ Processing Time</h5>
            <p style="margin: 0; color: #6b7280; font-size: 0.9rem;">Complex analyses on large datasets may take 2-5 minutes. Monitor the real-time logs to track progress and identify any potential issues during processing.</p>
        </div>   
    </div>
</div>
""", unsafe_allow_html=True)

# Add version and support information
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 2rem;">
    <p></p>
    <p>Supported formats: CSV, Excel (.xlsx/.xls)| Maximum file size: 100MB</p>
</div>
""", unsafe_allow_html=True)
