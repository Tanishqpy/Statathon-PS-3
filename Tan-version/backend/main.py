from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
# Change relative imports to absolute imports
from processor import process_data  # Remove the dot
from models import ModelInference   # Remove the dot
from fastapi.responses import JSONResponse, StreamingResponse, Response, FileResponse
import pandas as pd
import io
import time
import uuid
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from contextlib import asynccontextmanager
from io import BytesIO, StringIO
import os
from dotenv import load_dotenv
import requests
import json
from report_generator import create_statistical_report, create_basic_report

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/generate-report")


load_dotenv()  # Load environment variables from .env file

# Set API key manually if not in environment (for development only)
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB9xNj839DkZ3y4NgVG9pZjnofWRBjb4dE"  # Replace with your actual key

# Global dictionary to store processing status and logs
processing_tasks = {}

# Global model instance
model_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance
    print("Loading model...")
    # Use a valid Google Gemini model name
    model_instance = ModelInference(model_name="gemini-1.5-flash-latest", use_web_search=True)
    yield
    print("Cleaning up model...")
    model_instance = None

# Custom middleware to limit upload size
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 50 MB max upload
        if request.headers.get("content-length") and int(request.headers["content-length"]) > 50 * 1024 * 1024:
            return Response("File too large (max 50 MB)", status_code=413)
        return await call_next(request)

# Create FastAPI app
app = FastAPI(
    title="Data Analysis API",
    description="API for analyzing data files and generating explanations",
    version="1.0.0",
    lifespan=lifespan
)

@app.on_event("startup")
async def startup_event():
    """Run at application startup"""
    print("Application ready!")

# Add the upload size limit middleware
app.add_middleware(LimitUploadSizeMiddleware)

# Allow frontend to communicate with backend (CORS setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background task to process data
def process_data_task(task_id: str, contents: bytes, filename: str, prompt: str, model: ModelInference):
    try:
        logs = processing_tasks[task_id]["logs"]
        logs.append("Starting data processing...")
        logs.append(f"🔍 Analyzing file: {filename}")
        
        # Determine file type and parse
        if filename.endswith(".csv"):
            logs.append("Detected CSV file. Parsing with pandas...")
            df = pd.read_csv(io.StringIO(contents.decode("utf-8"))
                             )
            logs.append(f"CSV parsing complete. DataFrame shape: {df.shape}")
        elif filename.endswith((".xls", ".xlsx")):
            logs.append("Detected Excel file. Parsing with pandas...")
            df = pd.read_excel(io.BytesIO(contents))
            logs.append(f"Excel parsing complete. DataFrame shape: {df.shape}")
        else:
            logs.append(f"Unsupported file type: {filename}")
            processing_tasks[task_id]["status"] = "failed"
            processing_tasks[task_id]["error"] = "Unsupported file type."
            return

        # For large datasets, apply smart sampling
        original_shape = df.shape
        sampled_df = df
        is_sampled = False
        
        # If dataset is large, use a sample for analysis
        if df.shape[0] > 100000:
            # Very large dataset - take 10% sample
            sampled_df = df.sample(frac=0.1, random_state=42)
            is_sampled = True
            logs.append(f"📊 Dataset is very large ({df.shape[0]} rows). Working with a 10% sample for analysis.")
        elif df.shape[0] > 10000:
            # Large dataset - take 30% sample
            sampled_df = df.sample(frac=0.3, random_state=42)
            is_sampled = True
            logs.append(f"📊 Dataset is large ({df.shape[0]} rows). Working with a 30% sample for analysis.")
            
        # Process the data with the logging hook
        logs.append("🔄 Starting data processing pipeline...")
        
        def log_callback(message):
            processing_tasks[task_id]["logs"].append(message)
        
        # Use sampled data for analysis, but apply changes to full dataset
        df_clean, summary, narration = process_data(sampled_df, prompt, model, log_callback)
        
        # If we used sampling, note that in the logs
        if is_sampled:
            logs.append(f"Note: Analysis was performed on a sample of {sampled_df.shape[0]} rows. Cleaning recommendations apply to the full dataset.")
        
        logs.append("✅ Data processing complete")
        processing_tasks[task_id]["processed_data"] = summary["processed_data"]  
        processing_tasks[task_id]["column_names"] = df_clean.columns.tolist()
        processing_tasks[task_id]["summary"] = summary
        processing_tasks[task_id]["status"] = "completed"
        
    except Exception as e:
        import traceback
        processing_tasks[task_id]["logs"].append(f"❌ ERROR: {str(e)}")
        processing_tasks[task_id]["logs"].append(f"Stack trace: {traceback.format_exc()}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)
    
    finally:
        # Calculate execution time
        execution_time = time.time() - processing_tasks[task_id]["start_time"]
        processing_tasks[task_id]["execution_time"] = f"{execution_time:.2f} seconds"
        processing_tasks[task_id]["logs"].append(f"⏱️ Total processing time: {execution_time:.2f} seconds")

@app.post("/generate-report/{task_id}")
async def generate_report(task_id: str, request: Request):
    """
    Trigger n8n workflow to generate a PDF report for the processed data
    """
    # Check if task exists and is completed
    if task_id not in processing_tasks:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    
    task_info = processing_tasks[task_id]
    if task_info["status"] != "completed":
        return JSONResponse(
            status_code=400, 
            content={"error": "Data processing not completed yet"}
        )
    
    try:
        # Generate PDF buffer first
        pdf_buffer = create_statistical_report(task_info)
        
        # Option 1: Use direct PDF download if requested
        use_direct = request.query_params.get("direct", "false").lower() == "true"
        if use_direct:
            return StreamingResponse(
                pdf_buffer, 
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=statistical_report_{task_id}.pdf"}
            )
        
        # Option 2: Use n8n webhook (original functionality)
        # Prepare data for the report
        report_data = {
            "task_id": task_id,
            "summary": task_info["summary"],
            "processed_data": task_info["processed_data"],
            "column_names": task_info["column_names"],
            "execution_time": task_info["execution_time"],
        }
        
        # Send data to n8n webhook
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=report_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code >= 400:
            return JSONResponse(
                status_code=response.status_code,
                content={"error": f"n8n webhook error: {response.text}"}
            )
        
        # Parse the response from n8n
        try:
            n8n_response = response.json()
            return {
                "message": "Report generation initiated",
                "report_details": n8n_response
            }
        except json.JSONDecodeError:
            # If n8n doesn't return JSON, return the raw response
            return {
                "message": "Report generation initiated",
                "raw_response": response.text
            }
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate report: {str(e)}"}
        )

@app.get("/")
def read_root():
    return {"message": "FastAPI backend is working!"}

@app.post("/process")
async def process_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), prompt: str = Form(...)):
    # Create a unique ID for this task
    task_id = str(uuid.uuid4())
    
    # Store task status
    processing_tasks[task_id] = {
        "status": "processing",
        "logs": [],
        "start_time": time.time()
    }
    
    # Read file contents
    contents = await file.read()
    
    # Start background task to process the file
    background_tasks.add_task(process_data_task, task_id, contents, file.filename, prompt, model_instance)
    
    return {"task_id": task_id}

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in processing_tasks:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    
    task_info = processing_tasks[task_id]
    response = {
        "status": task_info["status"],
        "logs": task_info["logs"]
    }
    
    # If task is completed or failed, include all information
    if task_info["status"] in ["completed", "failed"]:
        response["execution_time"] = task_info["execution_time"]
        if task_info["status"] == "completed":
            response["summary"] = task_info["summary"]
            response["processed_data"] = task_info["processed_data"]
            response["column_names"] = task_info["column_names"]
        else:
            response["error"] = task_info["error"]
    
    return response

# Optional: Clean up old tasks periodically
@app.get("/cleanup")
async def cleanup_old_tasks():
    current_time = time.time()
    # Remove tasks older than 1 hour
    for task_id in list(processing_tasks.keys()):
        if current_time - processing_tasks[task_id]["start_time"] > 3600:  # 1 hour in seconds
            del processing_tasks[task_id]
    return {"message": f"Cleanup complete. {len(processing_tasks)} active tasks remaining."}

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    
    # Send immediate acknowledgment
    await websocket.send_json({"status": "connected", "message": "WebSocket connection established"})
    
    try:
        # Check if task exists
        if task_id not in processing_tasks:
            await websocket.send_json({"status": "error", "message": "Task not found"})
            await websocket.close()
            return
        
        # Send initial logs
        task_info = processing_tasks[task_id]
        await websocket.send_json({
            "status": task_info["status"],
            "logs": task_info["logs"]
        })
        
        # Send waiting message
        await websocket.send_json({
            "status": "processing",
            "message": "Waiting for AI model response (this may take 5-10 seconds)..."
        })
        
        # Keep track of the last log index we sent
        last_sent_log_index = len(task_info["logs"]) - 1
        
        # Keep connection open and check for updates every 100ms
        while True:
            await asyncio.sleep(0.1)  # Check every 100ms
            
            # Get latest task info
            task_info = processing_tasks[task_id]
            current_logs = task_info["logs"]
            
            # If there are new logs, send them
            if len(current_logs) > last_sent_log_index + 1:
                new_logs = current_logs[last_sent_log_index + 1:]
                await websocket.send_json({
                    "status": task_info["status"],
                    "new_logs": new_logs
                })
                last_sent_log_index = len(current_logs) - 1
            
            # If task is completed or failed, send final response and close
            if task_info["status"] in ["completed", "failed"]:
                final_response = {
                    "status": task_info["status"],
                    "execution_time": task_info["execution_time"]
                }
                
                if task_info["status"] == "completed":
                    final_response["summary"] = task_info["summary"]
                    final_response["processed_data"] = task_info["processed_data"]
                    final_response["column_names"] = task_info["column_names"]
                else:
                    final_response["error"] = task_info["error"]
                    
                await websocket.send_json(final_response)
                await websocket.close()
                break
                
    except WebSocketDisconnect:
        print(f"WebSocket client disconnected: {task_id}")

@app.get("/download/{task_id}/csv")
async def download_csv(task_id: str):
    """Download processed data as CSV"""
    if task_id not in processing_tasks or "processed_data" not in processing_tasks[task_id]:
        return Response(content="Data not found", status_code=404)
    
    try:
        # Get data from processing task
        data = processing_tasks[task_id]["processed_data"]
        column_names = processing_tasks[task_id]["column_names"]
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=column_names)
        
        # Convert to CSV
        output = StringIO()
        df.to_csv(output, index=False)
        
        # Create response
        response = StreamingResponse(
            iter([output.getvalue()]), 
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=processed_data_{task_id}.csv"
        
        return response
    except Exception as e:
        return Response(content=f"Error generating CSV: {str(e)}", status_code=500)

@app.get("/download/{task_id}/excel")
async def download_excel(task_id: str):
    """Download processed data as Excel file"""
    if task_id not in processing_tasks or "processed_data" not in processing_tasks[task_id]:
        return Response(content="Data not found", status_code=404)
    
    try:
        # Get data from processing task
        data = processing_tasks[task_id]["processed_data"]
        column_names = processing_tasks[task_id]["column_names"]
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=column_names)
        
        # Convert to Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='ProcessedData')
        output.seek(0)
        
        # Create response
        response = StreamingResponse(
            output, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=processed_data_{task_id}.xlsx"
        
        return response
    except Exception as e:
        return Response(content=f"Error generating Excel: {str(e)}", status_code=500)

@app.get("/download/{task_id}/pdf")
async def download_pdf(task_id: str):
    """Download processed data as a formatted PDF report"""
    if task_id not in processing_tasks or "processed_data" not in processing_tasks[task_id]:
        # Generate error report
        pdf_buffer = create_basic_report(task_id, "Data not found or processing incomplete")
        return StreamingResponse(
            pdf_buffer, 
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=error_report_{task_id}.pdf"}
        )
    
    try:
        # Generate statistical report with better error handling
        pdf_buffer = create_statistical_report(processing_tasks[task_id])
        
        # Create response
        return StreamingResponse(
            pdf_buffer, 
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=statistical_report_{task_id}.pdf"}
        )
    except Exception as e:
        import traceback
        error_message = f"Error generating PDF: {str(e)}\n\nTraceback: {traceback.format_exc()[:500]}"
        print(error_message)  # Log the full error
        # Generate error report with more detailed error message
        pdf_buffer = create_basic_report(task_id, error_message)
        return StreamingResponse(
            pdf_buffer, 
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=error_report_{task_id}.pdf"}
        )
