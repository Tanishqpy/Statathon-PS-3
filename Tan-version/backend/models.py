import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Optional, Any, Tuple
import time  # Add this import
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import urllib.parse
import random

# For local model inference
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: Hugging Face transformers not available. Will use rule-based fallback.")

class ModelInference:
    """Handles local LLM inference for data analysis tasks"""
    

    def __init__(self, model_name="gemini-1.5-flash-latest", device=None, use_web_search=True):
        """Initialize with Google Gemini API instead of local model"""
        
        # Add a check to ensure a valid Gemini model name is used
        if "/" in model_name or "phi" in model_name:
            corrected_model = "gemini-1.5-flash-latest"
            print(f"⚠️ Warning: Invalid model '{model_name}' provided for Gemini API.")
            print(f"✅ Automatically switching to a valid model: '{corrected_model}'")
            self.model_name = corrected_model
        else:
            self.model_name = model_name

        self.initialized = False
        self.use_web_search = use_web_search
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        # Cache configuration (keep existing cache functionality)
        self.cache_file = os.path.join(os.path.dirname(__file__), "column_cache.json")
        self.column_cache = self._load_column_cache()
        
        # Check if API key is available
        if not self.api_key:
            print("⚠️ Warning: GOOGLE_API_KEY environment variable is not set.")
            print("The model will not be available. Please set your API key.")
        else:
            self.initialized = True
            # Use self.model_name to print the corrected name
            print(f"✅ Google Gemini API ({self.model_name}) initialized successfully")
        
    def pipe(self, prompt_or_prompts, return_full_text=False, max_new_tokens=150):
        """
        Make API call(s) to Google Gemini API. Handles both a single string prompt
        and a list of string prompts.
        """
        if not self.initialized or not self.api_key:
            raise ValueError("Google Gemini API is not initialized. Please set GOOGLE_API_KEY.")

        # Standardize input to always be a list of prompts
        prompts = [prompt_or_prompts] if isinstance(prompt_or_prompts, str) else prompt_or_prompts

        if not isinstance(prompts, list):
            raise TypeError(f"Input must be a string or a list of strings, but got {type(prompts)}")

        all_results = []
        for prompt in prompts:
            # Validate each prompt in the batch
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                print("❌ Error: Attempted to send an empty prompt to the Gemini API.")
                all_results.append([{"generated_text": "Error: The prompt was empty."}])
                continue

            # API endpoint, headers, and payload
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            headers = {'Content-Type': 'application/json', 'X-goog-api-key': self.api_key}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_new_tokens, "temperature": 0.2, "topP": 0.95}
            }
            
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
                response.raise_for_status()
                result = response.json()
                
                if "candidates" in result and result["candidates"]:
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    formatted_result = {"generated_text": prompt + text if return_full_text else text}
                    # Wrap in a list to match Hugging Face pipeline's list-of-lists format
                    all_results.append([formatted_result])
                else:
                    reason = result.get("promptFeedback", {}).get("blockReason", "Unknown")
                    print(f"❌ Prompt blocked or failed. Reason: {reason}")
                    all_results.append([{"generated_text": f"Content generation failed: {reason}"}])

            except requests.exceptions.RequestException as e:
                error_message = f"Error calling Gemini API: {str(e)}"
                if e.response is not None:
                    error_message += f" | Details: {e.response.text}"
                print(error_message)
                all_results.append([{"generated_text": "Error: API call failed."}])

        # Return a list of lists, consistent with Hugging Face pipelines
        return all_results

    def safe_pipe(self, prompt, max_length=1000, return_full_text=False, cpu_fallback=True):
        """
        A safer version of pipe with automatic retries
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of the generated response
            return_full_text: Whether to return the full text or just the generated part
            cpu_fallback: Not used with API model but kept for compatibility
        
        Returns:
            Generated text output in the same format as self.pipe()
        """
        try:
            # Try with the primary model
            return self.pipe(prompt, return_full_text=return_full_text, max_new_tokens=max_length)
        except Exception as e:
            print(f"Error in Gemini API call: {str(e)}")
            
            # Wait and retry once with a shorter prompt
            try:
                print("Retrying with simplified prompt...")
                time.sleep(2)  # Brief delay before retry
                
                # Create a shorter prompt if original is too long
                if len(prompt) > 1000:
                    simplified_prompt = prompt[:900] + "... [truncated]"
                else:
                    simplified_prompt = prompt
                    
                return self.pipe(simplified_prompt, return_full_text=return_full_text, max_new_tokens=max_length)
                
            except Exception as retry_error:
                print(f"Retry also failed: {str(retry_error)}")
                
                # Return a fallback response
                return [{
                    "generated_text": "I apologize, but I couldn't process that request. Please try again with a simpler prompt."
                }]

    def _load_column_cache(self) -> Dict[str, Dict]:
        """Load column classifications from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    print(f"Loaded {len(cache)} column classifications from cache")
                    return cache
            else:
                print("No column cache found, creating new cache")
                return {}
        except Exception as e:
            print(f"Error loading column cache: {e}")
            return {}
            
    def _save_to_cache(self, column_name: str, classification: Dict, reason: str = "cache") -> None:
        """Save a column classification to cache"""
        if not column_name or not classification:
            return
            
        # Reload cache first to get latest version
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.column_cache = json.load(f)
        except Exception as e:
            print(f"Warning: Could not reload cache before saving: {e}")
            
        # Normalize column name for consistent caching
        normalized_name = column_name.lower().strip()
        
        # Add metadata to cached entry
        classification["cached_at"] = datetime.now().isoformat()
        classification["original_reason"] = classification.get("reason", reason)
        classification["reason"] = "cached"
        
        # Store in memory cache
        self.column_cache[normalized_name] = classification
        
        # Try to save to disk
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.column_cache, f, indent=2)
            print(f"✅ Added '{normalized_name}' to cache ({classification['type']})")
        except Exception as e:
            print(f"❌ Error saving to cache: {e}")
            
    def _get_from_cache(self, column_name: str) -> Dict:
        """Get a column classification from cache if available"""
        normalized_name = column_name.lower().strip()
        return self.column_cache.get(normalized_name)
        
    def _save_batch_to_cache(self, classifications: Dict[str, Dict]) -> None:
        """Save multiple column classifications to cache"""
        if not classifications:
            return
            
        # First reload the cache to ensure we have the latest version
        # This prevents overwriting changes made by other processes
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.column_cache = json.load(f)
        except Exception as e:
            print(f"Warning: Could not reload cache before saving: {e}")
            
        # Add each classification to memory cache with metadata
        for column_name, classification in classifications.items():
            normalized_name = column_name.lower().strip()
            
            # Skip if it's already coming from cache
            if classification.get("reason") == "cached":
                continue
                
            # Add metadata
            classification["cached_at"] = datetime.now().isoformat()
            classification["original_reason"] = classification.get("reason", "batch")
            classification["reason"] = "cached"
            
            self.column_cache[normalized_name] = classification
            
        # Save to disk
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.column_cache, f, indent=2)
            print(f"✅ Updated cache with {len(classifications)} new classifications")
            print(f"📊 Cache now contains {len(self.column_cache)} column patterns")
        except Exception as e:
            print(f"❌ Error saving batch to cache: {e}")
            
    def classify_column(self, col_name: str, sample_values: List, stats: Dict, log_callback=None) -> Dict:
        """
        Use the model to classify a column based on its name, sample values, and statistics.
        First checks cache, then uses model or rules.
        """
        if log_callback:
            log_callback(f"🧠 Classifying column '{col_name}'...")
        
        # First check cache
        cached_classification = self._get_from_cache(col_name)
        if cached_classification:
            if log_callback:
                log_callback(f"📋 Using cached classification for '{col_name}': {cached_classification['type']}")
            return cached_classification
        
        # If not in cache, use model or rules
        if not self.initialized:
            if log_callback:
                log_callback("⚠️ Model not available, using rule-based classification")
            # Get classification using rules
            classification = self._rule_based_classify(col_name, sample_values, stats, log_callback)
            # Save to cache for future use
            self._save_to_cache(col_name, classification)
            return classification
        
        # Use the model for classification
        try:
            # Improve the column classification prompt:
            prompt = f"""Task: Classify database columns for outlier detection.

            For each column name, determine its type:

            1. "identifier" (NO outlier detection): 
               - Company names, symbols, codes, IDs
               - Text fields that identify specific entities
               - Example columns: "COMPANY NAME", "Symbol", "ID", "Customer Name"

            2. "temporal" (NO outlier detection):
               - Date/time related fields
               - Example columns: "Date", "Year", "Time", "Start Date"

            3. "categorical_as_number" (NO outlier detection):
               - Categories represented as numbers
               - Type/class designations
               - Example columns: "Type", "Category", "Class", "Security Type"

            4. "geographic" (NO outlier detection):
               - Location codes, regions, coordinates
               - Example columns: "State", "Region", "District" 

            5. "regular_numeric" (YES outlier detection):
               - Actual measurements, prices, counts that should be checked
               - Example columns: "Price", "Weight", "Score", "Count"

            Column names to classify: {col_name}
            
            Format each answer as exactly: [Column Name]: [classification_type]
            """
            
            if log_callback:
                log_callback(f"🤖 Using AI model to classify '{col_name}'")
            
            # Get response from the model
            response = self.pipe(prompt, return_full_text=False, max_new_tokens=100)[0]['generated_text']
            
            if log_callback:
                log_callback(f"🤖 AI response: {response[:100]}...")
            
            # Parse the response to extract the classification
            classification_result = {"reason": "ai_classification"}
            
            # Check for identifier
            if re.search(r'identifier|id number|key|unique', response.lower()):
                classification_result.update({
                    "is_special": True,
                    "type": "identifier",
                    "should_detect_outliers": False
                })
            # Check for temporal
            elif re.search(r'temporal|date|time|year', response.lower()):
                classification_result.update({
                    "is_special": True,
                    "type": "temporal",
                    "should_detect_outliers": False
                })
            # Check for categorical
            elif re.search(r'categorical|categor|encoded|code', response.lower()):
                classification_result.update({
                    "is_special": True,
                    "type": "categorical_as_number",
                    "should_detect_outliers": False
                })
            # Default to regular numeric
            else:
                classification_result.update({
                    "is_special": False,
                    "type": "regular_numeric",
                    "should_detect_outliers": True
                })
            
            # Save result to cache before returning
            self._save_to_cache(col_name, classification_result)
            return classification_result
            
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Model inference failed: {str(e)}")
            # Fallback to rule-based classification
            classification = self._rule_based_classify(col_name, sample_values, stats, log_callback)
            # Still save rule-based result to cache
            self._save_to_cache(col_name, classification)
            return classification

    def classify_columns_with_ai(self, column_names: List[str], log_callback=None) -> Dict[str, Dict]:
        """
        Use AI, web search and rules to classify columns based on their names.
        """
        all_classifications = {}
        columns_for_rules = []
        
        # Step 1: Check cache first for all columns
        if log_callback:
            log_callback("🔍 Checking column classification cache...")
            
        for col in column_names:
            cached_classification = self._get_from_cache(col)
            if cached_classification:
                all_classifications[col] = cached_classification
                if log_callback:
                    log_callback(f"📋 Using cached classification for '{col}': {cached_classification['type']}")
            else:
                columns_for_rules.append(col)
                
        if log_callback:
            cache_hits = len(column_names) - len(columns_for_rules)
            if cache_hits > 0:
                log_callback(f"✅ Found {cache_hits} column(s) in cache")
                
        # Skip further processing if all columns were in cache
        if not columns_for_rules:
            return all_classifications
            
        # Step 2: Apply rule-based classification
        if not self.initialized:
            if log_callback:
                log_callback("⚠️ Model not available, will use rule-based classification")
                
        # Apply rule-based classification to obvious columns
        rule_based_classifications = {}
        columns_for_ai = []
        
        # Pre-filter columns with clear patterns
        for col in columns_for_rules:
            col_lower = col.lower().strip()
            
            # Date columns
            if any(kw in col_lower for kw in ['date', 'time', 'year', 'month', 'day']):
                rule_based_classifications[col] = {
                    "is_special": True,
                    "type": "temporal",
                    "reason": "quick_rule_date",
                    "should_detect_outliers": False
                }
                if log_callback:
                    log_callback(f"🔍 Quick rule: '{col}' classified as temporal")
                    
            # ... [rest of your rule-based patterns] ...
                    
            # Send to AI only columns that don't match rules
            else:
                columns_for_ai.append(col)
        
        # Save rule-based classifications to cache
        self._save_batch_to_cache(rule_based_classifications)
        
        # Step 3: Try web search for remaining columns
        web_search_results = {}
        if columns_for_ai:
            if log_callback:
                log_callback(f"🌐 Trying web search for {len(columns_for_ai)} remaining columns...")
            
            for col in columns_for_ai[:]:  # Use slice to avoid modifying during iteration
                web_result = self.web_search_column_type(col, log_callback)
                
                if web_result:
                    web_search_results[col] = web_result
                    columns_for_ai.remove(col)
                    if log_callback:
                        log_callback(f"🌐 Web search classified '{col}' as {web_result['type']}")
        
        # Step 4: Use AI for any remaining columns
        ai_classifications = {}
        
        # Only attempt AI classification if we have columns that need it and model is available
        if columns_for_ai and self.initialized:
            if log_callback:
                log_callback(f"🤖 Using AI to classify {len(columns_for_ai)} remaining columns")
            
            for col in columns_for_ai:
                try:
                    # Create a basic prompt for column classification based on name only
                    prompt = f"Classify this database column name: '{col}'. Is it likely to be: 1. identifier (like ID), 2. temporal (date/time), 3. categorical, or 4. regular numeric data? Answer with the type only."
                    
                    response = self.pipe(prompt, return_full_text=False, max_new_tokens=20)[0]['generated_text']
                    
                    # Parse the response to determine classification
                    if re.search(r'identifier|id', response.lower()):
                        ai_classifications[col] = {
                            "is_special": True, 
                            "type": "identifier",
                            "reason": "ai_classification",
                            "should_detect_outliers": False
                        }
                    elif re.search(r'temporal|date|time', response.lower()):
                        ai_classifications[col] = {
                            "is_special": True,
                            "type": "temporal",
                            "reason": "ai_classification",
                            "should_detect_outliers": False
                        }
                    elif re.search(r'categorical|categor', response.lower()):
                        ai_classifications[col] = {
                            "is_special": True,
                            "type": "categorical_as_number",
                            "reason": "ai_classification",
                            "should_detect_outliers": False
                        }
                    else:
                        ai_classifications[col] = {
                            "is_special": False,
                            "type": "regular_numeric",
                            "reason": "ai_classification",
                            "should_detect_outliers": True
                        }
                    
                    if log_callback:
                        log_callback(f"🤖 AI classified '{col}' as {ai_classifications[col]['type']}")
                        
                except Exception as e:
                    if log_callback:
                        log_callback(f"❌ AI classification failed for '{col}': {str(e)}")
                    # Default classification if AI fails
                    ai_classifications[col] = {
                        "is_special": False,
                        "type": "regular_numeric",
                        "reason": "ai_fallback",
                        "should_detect_outliers": True
                    }
        else:
            # Use default classification for remaining columns when AI is not available
            for col in columns_for_ai:
                ai_classifications[col] = {
                    "is_special": False,
                    "type": "regular_numeric",
                    "reason": "default_no_ai",
                    "should_detect_outliers": True
                }
        
        # Step 5: Save all new classifications to cache
        all_classifications.update(rule_based_classifications)
        all_classifications.update(web_search_results)
        all_classifications.update(ai_classifications)
        
        # Cache everything we learned
        self._save_batch_to_cache({**rule_based_classifications, **web_search_results, **ai_classifications})
        
        return all_classifications

    def test_model_connectivity(self):
        """Test function to verify model connectivity"""
        if not self.initialized:
            print("❌ Model not initialized")
            return False
        
        try:
            print("🔍 Testing model connectivity...")
            test_prompt = "Complete this sentence: The quick brown fox"
            
            print(f"Sending test prompt: '{test_prompt}'")
            start_time = time.time()
            
            # Print more debug info
            print(f"Model type: Google Gemini API")
            
            # Try with max_length parameter to control response length
            response = self.pipe(test_prompt, 
                                return_full_text=False, 
                                max_new_tokens=20)[0]['generated_text']
            
            elapsed = time.time() - start_time
            print(f"✅ Model responded in {elapsed:.2f} seconds")
            print(f"Response: {response}")
            
            # Try a column classification example
            print("\nTesting column classification...")
            col_prompt = """Classify this database column:
            Column name: 'Transaction_Date'
            What type is this? Choose one:
            1. identifier
            2. temporal
            3. regular_numeric"""
            
            col_start = time.time()
            col_response = self.pipe(col_prompt, 
                                    return_full_text=False,
                                    max_new_tokens=20)[0]['generated_text']
            col_elapsed = time.time() - col_start
            print(f"✅ Classification responded in {col_elapsed:.2f} seconds")
            print(f"Classification response: {col_response}")
            
            return True
        except Exception as e:
            import traceback
            print(f"❌ Model test failed: {str(e)}")
            print(traceback.format_exc())
            return False

    def _rule_based_classify(self, col_name: str, sample_values: List, stats: Dict, log_callback=None) -> Dict:
        """Rule-based column classification when model is not available"""
        def log(message):
            if log_callback:
                log_callback(f"🔍 RULE: {message}")
        
        col_name_lower = col_name.lower().strip()
        
        # 1. Company name/title detection (always text identifiers)
        company_keywords = ['company', 'name', 'title', 'business', 'organization', 'entity', 'firm']
        if any(keyword in col_name_lower for keyword in company_keywords):
            log(f"Column '{col_name}' identified as company name/text identifier")
            return {
                "is_special": True,
                "type": "identifier",
                "reason": "rule_based_company_name",
                "should_detect_outliers": False
            }
        
        # 2. Symbol/ticker/code detection (always identifiers)
        symbol_keywords = ['symbol', 'ticker', 'code', 'stock']
        if any(keyword in col_name_lower for keyword in symbol_keywords):
            log(f"Column '{col_name}' identified as symbol/ticker identifier")
            return {
                "is_special": True,
                "type": "identifier",
                "reason": "rule_based_symbol",
                "should_detect_outliers": False
            }
        
        # 3. Type/category detection (text categories)
        type_keywords = ['type', 'category', 'class', 'security type', 'classification']
        if any(keyword in col_name_lower for keyword in type_keywords):
            log(f"Column '{col_name}' identified as categorical type")
            return {
                "is_special": True,
                "type": "categorical_as_number",
                "reason": "rule_based_type_category",
                "should_detect_outliers": False
            }
        
        # 4. Range/band detection (special text format)
        range_keywords = ['range', 'band', 'interval', 'span']
        if any(keyword in col_name_lower for keyword in range_keywords):
            log(f"Column '{col_name}' identified as range/interval")
            return {
                "is_special": True,
                "type": "identifier",
                "reason": "rule_based_range",
                "should_detect_outliers": False
            }
        
        # 5. Check for geographic/administrative columns first
        geographic_keywords = ['state', 'country', 'zip', 'postal', 'region', 'district', 
                              'province', 'county', 'city', 'territory', 'area', 'zone']
        
        # Special check for geographic identifiers
        for keyword in geographic_keywords:
            if keyword in col_name_lower:
                log(f"Column '{col_name}' identified as geographic code/id based on name")
                return {
                    "is_special": True,
                    "type": "categorical_as_number",
                    "reason": "rule_based_geographic_name",
                    "should_detect_outliers": False
                }
                
        # 6. Identifier detection - check for typical ID column names
        id_keywords = ['id', 'code', 'key', 'num', 'no', 'number', 'sl_no', 'serial']
        if any(keyword in col_name_lower for keyword in id_keywords):
            # Count unique values - if high percentage is unique, it's likely an ID
            if stats["unique"] / stats["count"] > 0.9:
                log(f"Column '{col_name}' identified as identifier based on uniqueness and name")
                return {
                    "is_special": True,
                    "type": "identifier",
                    "reason": "rule_based_unique_id",
                    "should_detect_outliers": False
                }
        
        # 7. Date/time detection
        date_keywords = ['year', 'month', 'day', 'date', 'time', 'period', 'quarter']
        if any(keyword in col_name_lower for keyword in date_keywords):
            log(f"Column '{col_name}' identified as temporal based on name")
            return {
                "is_special": True,
                "type": "temporal",
                "reason": "rule_based_name_match",
                "should_detect_outliers": False
            }
        
        # 8. Detect categorical values encoded as numbers
        # Check if the column has few unique values relative to total
        unique_ratio = stats["unique"] / stats["count"]
        if unique_ratio < 0.05 and stats["unique"] < 20:
            log(f"Column '{col_name}' identified as categorical based on low unique ratio ({unique_ratio:.2f})")
            return {
                "is_special": True,
                "type": "categorical_as_number",
                "reason": "rule_based_distribution",
                "should_detect_outliers": False
            }
        
        # 9. Check for administrative/geographic columns with specific value ranges
        # Most states/districts/regions are coded with relatively small numbers
        if 'state' in col_name_lower or 'district' in col_name_lower or 'region' in col_name_lower:
            if stats["unique"] < 50 and stats["max"] < 100:
                log(f"Column '{col_name}' identified as geographic code based on value range")
                return {
                    "is_special": True,
                    "type": "categorical_as_number", 
                    "reason": "rule_based_geographic_range",
                    "should_detect_outliers": False
                }
                
        # 10. Check for stratum/stratification columns
        if 'stratum' in col_name_lower or 'strat' in col_name_lower:
            if stats["unique"] < 100:
                log(f"Column '{col_name}' identified as stratification variable")
                return {
                    "is_special": True,
                    "type": "categorical_as_number",
                    "reason": "rule_based_stratification",
                    "should_detect_outliers": False
                }
        
        # 11. Last resort - check sample values to determine if text/categorical
        if stats and "dtype" in stats:
            # If column contains strings, it's likely an identifier or categorical
            if stats["dtype"] == "object" or stats["dtype"] == "string":
                log(f"Column '{col_name}' identified as text column based on data type")
                return {
                    "is_special": True,
                    "type": "identifier",
                    "reason": "rule_based_text_dtype",
                    "should_detect_outliers": False
                }
        
        # 12. Default to regular numeric
        log(f"Column '{col_name}' identified as regular numeric (default)")
        return {
            "is_special": False,
            "type": "regular_numeric",
            "reason": "rule_based_default",
            "should_detect_outliers": True
        }

    def web_search_column_type(self, column_name, log_callback=None):
        """Use web search to determine the likely type of a database column"""
        # Skip if web search is disabled
        if not self.use_web_search:
            return None
        
        if log_callback:
            log_callback(f"🔍 Searching for information about '{column_name}'...")
        
        # First check using dictionary/pattern matching before making API calls
        result = self._dictionary_based_column_classification(column_name, log_callback)
        if result:
            return result
        
        try:
            # Choose a search method randomly to distribute across services
            search_method = random.choice(["duckduckgo", "wikipedia", "serper"])
            
            if search_method == "duckduckgo":
                return self._search_duckduckgo(column_name, log_callback)
            elif search_method == "wikipedia":
                return self._search_wikipedia(column_name, log_callback)
            elif search_method == "serper":
                return self._search_serper(column_name, log_callback)
                
        except Exception as e:
            import traceback
            if log_callback:
                log_callback(f"❌ Search failed: {str(e)}")
            
            # Fall back to dictionary-based approach
            return self._dictionary_based_column_classification(column_name, log_callback)

    def inspect_cache(self):
        """Utility method to inspect the column cache"""
        if not self.column_cache:
            print("📭 Cache is empty")
            return
            
        print(f"📊 Cache contains {len(self.column_cache)} column patterns")
        
        # Group by classification type
        types = {}
        reasons = {}
        for col, info in self.column_cache.items():
            col_type = info.get("type", "unknown")
            types[col_type] = types.get(col_type, 0) + 1
            
            reason = info.get("original_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        
        print("\nClassification Types:")
        for t, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {t}: {count} columns")
            
        print("\nClassification Sources:")
        for r, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {r}: {count} columns")
            
        # Show some example entries
        print("\nSample Cache Entries:")
        samples = list(self.column_cache.items())[:5]
        for col, info in samples:
            print(f"  - '{col}': {info['type']} (from {info.get('original_reason', 'unknown')})")
            
        print("\nUse clear_cache() to reset the cache if needed")
        
    def clear_cache(self):
        """Utility method to clear the column cache"""
        self.column_cache = {}
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({}, f)
            print("✅ Cache cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")

    def fix_cache_entries(self):
        """Fix incorrectly classified entries in the cache"""
        corrections = {
            "company name": {
                "is_special": True,
                "type": "identifier",
                "reason": "cached",
                "should_detect_outliers": False,
                "cached_at": datetime.now().isoformat(),
                "original_reason": "manual_correction"
            },
            "security type": {
                "is_special": True,
                "type": "categorical_as_number",
                "reason": "cached",
                "should_detect_outliers": False,
                "cached_at": datetime.now().isoformat(),
                "original_reason": "manual_correction"
            },
            "symbol": {
                "is_special": True,
                "type": "identifier",
                "reason": "cached",
                "should_detect_outliers": False,
                "cached_at": datetime.now().isoformat(),
                "original_reason": "manual_correction"
            }
        }
        
        # Update the cache
        for col, correction in corrections.items():
            self.column_cache[col] = correction
        
        # Save corrected cache
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.column_cache, f, indent=2)
            print(f"✅ Fixed {len(corrections)} incorrect classifications in cache")
        except Exception as e:
            print(f"❌ Error saving fixed cache: {e}")

    def analyze_column_data_type(self, df, col_name):
        """Analyze actual data values to determine column type"""
        try:
            sample = df[col_name].dropna().head(10).tolist()
            
            # Check for text values
            if df[col_name].dtype == 'object':
                # If most values contain letters, likely text/identifier
                text_values = sum(1 for val in sample if isinstance(val, str) and any(c.isalpha() for c in val))
                if text_values > len(sample) * 0.5:
                    return "identifier"
            
            return None  # No definitive conclusion
        except:
            return None

    def _dictionary_based_column_classification(self, column_name, log_callback=None):
        """Classify column based on comprehensive dictionary of common column names"""
        col_lower = column_name.lower().strip()
        
        # Comprehensive dictionary of common column patterns
        column_patterns = {
            # Identifiers
            "identifier": [
                r"id$", r"^id", r"code$", r"^code", r"key$", r"^key", r"uuid", 
                r"name$", r"^name", r"title", r"number$", r"symbol", r"ticker",
                r"username", r"email", r"phone", r"account", r"survey.*name"
            ],
            
            # Temporal
            "temporal": [
                r"date", r"time", r"year", r"month", r"day", r"period", r"quarter",
                r"^dt_", r"^date_", r"_date$", r"_dt$", r"timestamp", r".*_days$"
            ],
            
            # Categorical
            "categorical_as_number": [
                r"type$", r"^type", r"category", r"class", r"status", r"level",
                r"grade", r"group", r"segment", r"sector", r"tier", r"flag", 
                r"rank"
            ],
            
            # Geographic
            "geographic": [
                r"state", r"country", r"city", r"region", r"district", r"province",
                r"zip", r"postal", r"address", r"location", r"area", r"territory",
                r"longitude", r"latitude", r"coord"
            ],
            
            # Survey-specific
            "survey_admin": [
                r"stratum", r"sample", r"panel", r"fsu", r"nss", r"household", 
                r"questionnaire", r"survey", r"division", r"sub_div", r"sl_no"
            ],
            
            # Regular numeric
            "regular_numeric": [
                r"amount", r"total", r"sum", r"count", r"quantity", r"qty", r"volume",
                r"price", r"cost", r"fee", r"value", r"average", r"avg", r"mean",
                r"rate", r"ratio", r"percent", r"proportion", r"number_of", r"num_"
            ]
        }
        
        # Check pattern matches
        for type_name, patterns in column_patterns.items():
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    if log_callback:
                        log_callback(f"📚 Column '{column_name}' matches pattern '{pattern}' for '{type_name}'")
                        
                    should_detect_outliers = type_name == "regular_numeric"
                    
                    # Special case for survey administration columns
                    if type_name == "survey_admin":
                        should_detect_outliers = False
                        type_name = "categorical_as_number"  # Convert to standard type
                        
                    return {
                        "is_special": not should_detect_outliers,
                        "type": type_name if type_name != "geographic" else "categorical_as_number",
                        "reason": "dictionary_pattern_match",
                        "should_detect_outliers": should_detect_outliers,
                        "confidence": 80
                    }
        
        return None

    def _search_duckduckgo(self, column_name, log_callback=None):
        """Use DuckDuckGo search API for column classification"""
        if log_callback:
            log_callback(f"🦆 Using DuckDuckGo search for '{column_name}'")
        
        # Random delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 2.0))
        
        try:
            search_term = urllib.parse.quote(f"{column_name} database column type")
            url = f"https://api.duckduckgo.com/?q={search_term}&format=json"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the text from the results
                abstract = result.get("Abstract", "")
                definition = result.get("Definition", "")
                related_topics = []
                
                for topic in result.get("RelatedTopics", []):
                    if "Text" in topic:
                        related_topics.append(topic["Text"])
                
                # Combine all text
                all_text = (abstract + " " + definition + " " + " ".join(related_topics)).lower()
                
                if log_callback:
                    log_callback(f"🦆 DuckDuckGo returned {len(all_text)} chars of information")
                
                # Use the same type indicators as before
                type_indicators = {
                    "identifier": ["identifier", "primary key", "unique", "name", "title", "code", "id column"],
                    "temporal": ["date", "time", "timestamp", "datetime", "year", "month"],
                    "categorical_as_number": ["category", "type", "class", "enum", "categorical"],
                    "geographic": ["location", "address", "country", "region", "coordinate", "lat", "lon"],
                    "regular_numeric": ["numeric", "decimal", "float", "int", "measure", "value", "amount"]
                }
                
                # Count occurrences
                type_scores = {t: 0 for t in type_indicators}
                for type_name, indicators in type_indicators.items():
                    for indicator in indicators:
                        count = all_text.count(indicator)
                        type_scores[type_name] += count
                
                # Find the type with the highest score
                best_type = max(type_scores.items(), key=lambda x: x[1])
                
                if best_type[1] > 0:
                    should_detect_outliers = best_type[0] == "regular_numeric"
                    return {
                        "is_special": not should_detect_outliers,
                        "type": best_type[0] if best_type[0] != "geographic" else "categorical_as_number",
                        "reason": "duckduckgo_search",
                        "should_detect_outliers": should_detect_outliers,
                        "confidence": min(best_type[1] * 10, 95)
                    }
        except Exception as e:
            if log_callback:
                log_callback(f"❌ DuckDuckGo search failed: {str(e)}")
        
        return None

    def _search_wikipedia(self, column_name, log_callback=None):
        """Use Wikipedia API for column research"""
        if log_callback:
            log_callback(f"📚 Using Wikipedia search for '{column_name}'")
        
        # Random delay
        time.sleep(random.uniform(0.5, 2.0))
        
        try:
            search_term = urllib.parse.quote(f"{column_name} database")
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_term}&format=json"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                search_results = result.get("query", {}).get("search", [])
                
                # Extract snippets from search results
                all_text = " ".join([item.get("snippet", "") for item in search_results])
                all_text = re.sub(r'<[^>]+>', '', all_text).lower()  # Remove HTML tags
                
                if log_callback:
                    log_callback(f"📚 Wikipedia returned {len(all_text)} chars of information")
                
                # Use the same classification logic as before
                type_indicators = {
                    "identifier": ["identifier", "primary key", "unique", "name", "title", "code", "id"],
                    "temporal": ["date", "time", "timestamp", "datetime", "year", "month"],
                    "categorical_as_number": ["category", "type", "class", "enum", "categorical"],
                    "geographic": ["location", "address", "country", "region", "coordinate", "lat", "lon"],
                    "regular_numeric": ["numeric", "decimal", "float", "int", "measure", "value", "amount"]
                }
                
                # Count occurrences
                type_scores = {t: 0 for t in type_indicators}
                for type_name, indicators in type_indicators.items():
                    for indicator in indicators:
                        count = all_text.count(indicator)
                        type_scores[type_name] += count
                
                # Find the type with the highest score
                best_type = max(type_scores.items(), key=lambda x: x[1])
                
                if best_type[1] > 0:
                    should_detect_outliers = best_type[0] == "regular_numeric"
                    return {
                        "is_special": not should_detect_outliers,
                        "type": best_type[0] if best_type[0] != "geographic" else "categorical_as_number",
                        "reason": "wikipedia_search",
                        "should_detect_outliers": should_detect_outliers,
                        "confidence": min(best_type[1] * 10, 90)
                    }
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Wikipedia search failed: {str(e)}")
        
        return None

    def _search_serper(self, column_name, log_callback=None):
        """Use Serper.dev API if API key is available"""
        # You would need to set SERPER_API_KEY in your environment variables
        api_key = os.environ.get("SERPER_API_KEY")
        if not api_key:
            return None
            
        if log_callback:
            log_callback(f"🔎 Using Serper API for '{column_name}'")
        
        try:
            url = "https://serpapi.com/search"
            
            params = {
                "engine": "google",
                "q": f"{column_name} database column type",
                "api_key": api_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract snippets from organic results
                organic_results = result.get("organic_results", [])
                all_text = " ".join([
                    item.get("title", "") + " " + item.get("snippet", "")
                    for item in organic_results
                ]).lower()
                
                if log_callback:
                    log_callback(f"🔎 Serper API returned information about '{column_name}'")
                
                # Apply the same classification logic
                # ...rest of classification logic from previous methods...
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Serper API search failed: {str(e)}")
        
        return None

    def safe_pipe(self, prompt, max_length=1000, return_full_text=False, cpu_fallback=True):
        """
        A safer version of pipe that handles CUDA errors with fallback to CPU
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of the generated response
            return_full_text: Whether to return the full text or just the generated part
            cpu_fallback: Whether to try falling back to CPU if CUDA errors occur
        
        Returns:
            Generated text output in the same format as self.pipe()
        """
        try:
            # First try with original settings
            return self.pipe(prompt, return_full_text=return_full_text)
        except Exception as e:
            print(f"Error in model inference: {str(e)}")
            
            if cpu_fallback and "CUDA" in str(e):
                print("Falling back to CPU for this inference...")
                
                # Save original device
                original_device = None
                if hasattr(self.model, 'device'):
                    original_device = self.model.device
                
                try:
                    # Move model to CPU temporarily
                    if hasattr(self.model, 'to'):
                        self.model.to('cpu')
                    
                    # Try inference on CPU
                    result = self.pipe(prompt, return_full_text=return_full_text)
                    
                    # Move model back to original device
                    if original_device and hasattr(self.model, 'to'):
                        self.model.to(original_device)
                        
                    return result
                except Exception as cpu_error:
                    # If CPU also fails, raise the original error
                    print(f"CPU fallback also failed: {str(cpu_error)}")
                    if original_device and hasattr(self.model, 'to'):
                        self.model.to(original_device)
                    raise e
            else:
                # If not CUDA error or no fallback requested, just raise the original error
                raise e

def get_column_stats(df, column_name):
    """Get basic statistics for a column"""
    series = df[column_name]
    
    try:
        stats = {
            "min": float(series.min()) if not pd.isna(series.min()) else None,
            "max": float(series.max()) if not pd.isna(series.max()) else None,
            "mean": float(series.mean()) if not pd.isna(series.mean()) else None,
            "median": float(series.median()) if not pd.isna(series.median()) else None,
            "unique": int(series.nunique()),
            "count": int(series.count()),
            "null_count": int(series.isna().sum()),
            "null_pct": float(series.isna().mean() * 100) if len(series) > 0 else 0
        }
    except Exception as e:
        stats = {
            "error": str(e),
            "unique": int(series.nunique()),
            "count": int(series.count())
        }
        
    return stats