from models import ModelInference
import pandas as pd
from typing import Dict, List, Any, Optional
import threading
import time
import queue

class Narrator:
    """
    Uses a language model to provide human-like explanations of data processing steps
    """
    def __init__(self, model: Optional[ModelInference] = None):
        self.model = model  # Use the same model instance as the processor
        self.explanation_queue = queue.Queue()
        self.max_wait_time = 2.0  # Maximum seconds to wait for model response
        
    def explain_dataset(self, df: pd.DataFrame) -> str:
        """Generate a human-like description of the dataset"""
        rows, cols = df.shape
        dtypes = df.dtypes.value_counts().to_dict()
        num_numeric = sum(count for dtype, count in dtypes.items() if pd.api.types.is_numeric_dtype(dtype))
        num_categorical = sum(count for dtype, count in dtypes.items() if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype))
        
        # Start with fallback explanation in case model is slow
        fallback = self._fallback_dataset_description(df)
        
        if self.model and self.model.initialized:
            prompt = f"""
            Write a brief, friendly paragraph describing this dataset. Use simple, non-technical language:
            
            - Dataset has {rows} rows and {cols} columns
            - There are {num_numeric} numeric columns and {num_categorical} text/categorical columns
            - Column names: {', '.join(df.columns.tolist())}
            
            Your explanation (1-2 sentences, friendly tone):
            """
            
            # Try to get model response with timeout
            return self._get_model_response_with_timeout(prompt, fallback)
        else:
            return fallback
    
    def _get_model_response_with_timeout(self, prompt, fallback):
        """Get model response with timeout to avoid blocking"""
        if not self.model or not self.model.initialized:
            return fallback
            
        # Clear queue
        while not self.explanation_queue.empty():
            try:
                self.explanation_queue.get_nowait()
            except queue.Empty:
                break
                
        # Start a thread to get the model response
        def get_response():
            try:
                response = self.model.pipe(prompt, return_full_text=False)[0]['generated_text']
                self.explanation_queue.put(response.strip())
            except Exception as e:
                self.explanation_queue.put(None)
        
        # Start thread
        thread = threading.Thread(target=get_response)
        thread.daemon = True
        thread.start()
        
        # Wait for response with timeout
        try:
            start_time = time.time()
            response = self.explanation_queue.get(timeout=self.max_wait_time)
            if response:
                return response
            return fallback
        except queue.Empty:
            # If timeout, return fallback immediately
            return fallback + " (Model response timeout)"
    
    def _fallback_dataset_description(self, df: pd.DataFrame) -> str:
        """Generate a simple description without using the model"""
        rows, cols = df.shape
        return f"I'm looking at your dataset with {rows} rows and {cols} columns. Let's see what we can learn from it!"
    
    def explain_column_classification(self, col_name: str, classification: Dict) -> str:
        """Explain why a column was classified a certain way"""
        fallback = self._fallback_column_explanation(col_name, classification)
        
        if self.model and self.model.initialized:
            col_type = classification["type"]
            reason = classification["reason"]
            is_special = classification["is_special"]
            
            prompt = f"""
            Explain to a non-technical person why the column '{col_name}' was classified as {col_type}.
            
            Classification details:
            - Type: {col_type}
            - Is special column: {is_special}
            - Reason: {reason}
            
            Your explanation (1 short, friendly sentence):
            """
            
            return self._get_model_response_with_timeout(prompt, fallback)
        else:
            return fallback
    
    def _fallback_column_explanation(self, col_name: str, classification: Dict) -> str:
        """Generate a simple column explanation without using the model"""
        col_type = classification["type"]
        
        if col_type == "identifier":
            return f"I noticed that '{col_name}' looks like an ID column, so I'll treat it carefully and not flag any values as outliers."
        elif col_type == "temporal":
            return f"'{col_name}' seems to contain time or date information, so I won't look for outliers here."
        elif col_type == "categorical_as_number":
            return f"'{col_name}' contains numbers that represent categories rather than measurements, so outlier detection isn't appropriate."
        elif col_type == "geographic":
            return f"'{col_name}' appears to contain geographic codes or coordinates, which I'll preserve as-is."
        else:
            return f"'{col_name}' contains regular numeric values that I'll analyze for potential outliers."
    
    def explain_missing_value_strategy(self, col_name: str, strategy: str, value: Any) -> str:
        """Explain the strategy used to fill missing values"""
        fallback = self._fallback_missing_explanation(col_name, strategy, value)
        
        if self.model and self.model.initialized:
            prompt = f"""
            Explain to a non-technical person why we filled missing values in column '{col_name}' using the {strategy} method (value: {value}).
            
            Your explanation (1 short, friendly sentence):
            """
            
            return self._get_model_response_with_timeout(prompt, fallback)
        else:
            return fallback
    
    def _fallback_missing_explanation(self, col_name: str, strategy: str, value: Any) -> str:
        """Generate a simple explanation for missing value handling"""
        if strategy == "median":
            return f"For missing values in '{col_name}', I used the middle value ({value}) to fill in the gaps."
        elif strategy == "mode":
            return f"When values were missing in '{col_name}', I used the most common value ({value}) as a replacement."
        else:
            return f"I filled in missing values in '{col_name}' with {value}."
    
    def explain_outliers(self, col_name: str, count: int, percentage: float) -> str:
        """Explain outliers found in a column"""
        fallback = self._fallback_outlier_explanation(col_name, count, percentage)
        
        if self.model and self.model.initialized:
            prompt = f"""
            Explain to a non-technical person what it means that we found {count} outliers ({percentage:.1f}%) in column '{col_name}'.
            
            Your explanation (1 short, friendly sentence):
            """
            
            return self._get_model_response_with_timeout(prompt, fallback)
        else:
            return fallback
    
    def _fallback_outlier_explanation(self, col_name: str, count: int, percentage: float) -> str:
        """Generate a simple explanation for outliers"""
        if percentage < 1:
            return f"I found a few unusual values ({count}) in '{col_name}' that might be errors or special cases."
        elif percentage < 5:
            return f"About {percentage:.1f}% of values in '{col_name}' are outliers that stand out from the typical pattern."
        else:
            return f"There are quite a few unusual values in '{col_name}' ({count} values, or {percentage:.1f}%) that don't follow the normal pattern."
    
    def summarize_changes(self, original_shape: tuple, final_shape: tuple, changes_made: Dict) -> str:
        """Summarize all the changes made to the dataset"""
        rows_removed = original_shape[0] - final_shape[0]
        num_filled_cols = len(changes_made.get("missing_values_filled", {}))
        num_outlier_cols = len(changes_made.get("outliers_removed", {}))
        
        fallback = self._fallback_summary(original_shape, final_shape, rows_removed, num_filled_cols, num_outlier_cols)
        
        if self.model and self.model.initialized:
            prompt = f"""
            Summarize for a non-technical person what cleaning we performed on the dataset:
            
            - Started with {original_shape[0]} rows and {original_shape[1]} columns
            - Ended with {final_shape[0]} rows and {final_shape[1]} columns
            - Filled missing values in {num_filled_cols} columns
            - Removed outliers from {num_outlier_cols} columns
            - Total rows removed: {rows_removed}
            
            Your explanation (2-3 friendly sentences, avoid technical terms):
            """
            
            return self._get_model_response_with_timeout(prompt, fallback)
        else:
            return fallback
    
    def _fallback_summary(self, original_shape: tuple, final_shape: tuple, 
                         rows_removed: int, num_filled_cols: int, num_outlier_cols: int) -> str:
        """Generate a simple summary without using the model"""
        summary = f"I've cleaned your dataset! "
        
        if num_filled_cols > 0:
            summary += f"I filled in missing information in {num_filled_cols} columns. "
            
        if rows_removed > 0:
            percentage = (rows_removed / original_shape[0]) * 100
            if percentage < 1:
                summary += f"I removed a small number of unusual values ({rows_removed} rows). "
            elif percentage < 5:
                summary += f"I removed some outliers that might have skewed your analysis ({rows_removed} rows, about {percentage:.1f}%). "
            else:
                summary += f"I found and removed a significant number of outliers ({rows_removed} rows, about {percentage:.1f}%). "
        
        summary += f"Your data is now cleaner and ready for analysis!"
        return summary