to run the webapp -> streamlit run ./frontend/app.py



to run the server -> cd ./backend
                  -> uvicorn main:app --reload

## Statistical Report PDF Generation Guide

### How to Get Your Task ID

The task ID is required to generate and download PDF reports. Here's how to get it:

1. **From API Response:** When you upload a file using the `/process` endpoint, the API returns a JSON response containing your task ID:
   ```json
   {
     "task_id": "0a1b2c3d-4e5f-6789-abcd-ef0123456789"
   }
   ```

2. **From the Frontend:** If using the Streamlit frontend, the task ID is typically stored after file processing. Look for:
   - A success message displaying your task ID
   - Task ID in the URL parameters
   - Check the browser console for API responses

3. **Using the Status Endpoint:** If processing has already started, you can view all active tasks:
   ```bash
   curl http://localhost:8000/cleanup
   ```
   The response includes a count of active tasks. You may need to check your browser network tab to find recent task IDs.

### Method 1: Direct Download via Browser

1. After uploading and processing your data, get your `task_id` from the API response
2. Open your browser and navigate to:
   ```
   http://localhost:8000/download/{task_id}/pdf
   ```
   Replace `{task_id}` with your actual task ID
3. The browser will automatically download the PDF report

### Method 2: Using the API Endpoints

#### Option A: Direct PDF Download via API

```bash
# Using curl
curl -X GET "http://localhost:8000/download/{task_id}/pdf" --output report.pdf

# Using wget
wget -O report.pdf "http://localhost:8000/download/{task_id}/pdf"
```

#### Option B: Using the Generate Report Endpoint

For direct PDF download:
```bash
curl -X POST "http://localhost:8000/generate-report/{task_id}?direct=true" --output report.pdf
```

For n8n integration (returns JSON response):
```bash
curl -X POST "http://localhost:8000/generate-report/{task_id}"
```

### Method 3: Integration in Frontend

Add a button in your frontend application:

```javascript
// Example code for a download button in frontend
function downloadPDF(taskId) {
  window.location.href = `http://localhost:8000/download/${taskId}/pdf`;
}

// HTML Button example
<button onclick="downloadPDF('your-task-id')">Download PDF Report</button>
```

### Report Content

The PDF report includes:
- Dataset overview and statistics
- AI-generated analysis and insights
- Data visualizations (histograms, charts)
- Data cleaning summary
- Statistical summaries of numeric and categorical columns

### Troubleshooting

- If you receive a 404 error, verify your task ID is correct
- If you receive a 400 error, ensure data processing is complete
- For error reports, check the PDF content which will include error details

#### Common PDF Generation Errors

1. **"Style 'Heading1' already defined in stylesheet"**
   
   This error occurs when the PDF generator attempts to redefine an existing style.
   
   **Solution for developers:**
   - In the backend PDF generation code, ensure styles are only defined once
   - Check if there are multiple instances of style definitions for headings
   - Wrap style definitions in a conditional check to prevent redefinition:
     ```python
     # Example fix in the PDF generation code
     if 'Heading1' not in stylesheet.byName:
         stylesheet.add(ParagraphStyle(name='Heading1', fontSize=16, bold=True))
     ```
   
   **Temporary workaround:**
   - Try generating the report again with a new task ID
   - Download data in CSV/Excel format instead and generate reports locally
   - If the problem persists, contact the administrator with the error details
