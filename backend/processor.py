# backend/processor.py

def process_data(df, prompt):
    """
    This is a placeholder for real data cleaning and ML.
    For now, it just returns basic info and echoes the prompt.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "user_prompt": prompt,
        "sample_data": df.head().to_dict(orient="records"),
    }
    return summary
