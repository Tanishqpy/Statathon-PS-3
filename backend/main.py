from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from processor import process_data
from fastapi.responses import JSONResponse
import pandas as pd
import io

app = FastAPI()

# Allow frontend to communicate with backend (CORS setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "FastAPI backend is working!"}

@app.post("/process")
async def process_file(prompt: str = Form(...), file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Handle both CSV and Excel
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

        # Pass to processor
        result = process_data(df, prompt)

        return {"message": "Processed successfully", "summary": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
