import pandas as pd
import numpy as np
import json
from models import ModelInference
from narrator import Narrator
import threading
import time

def get_column_stats(df, column_name):
    """
    Calculates detailed statistics for a single DataFrame column.
    This function was likely removed from models.py during refactoring;
    it is better placed here in processor.py.
    """
    col = df[column_name]
    stats = {
        'name': column_name,
        'dtype': str(col.dtype),
        'missing_values': int(col.isnull().sum()),
        'missing_percentage': round(col.isnull().mean() * 100, 2)
    }

    # Ensure all data is JSON serializable
    if pd.api.types.is_numeric_dtype(col):
        stats['type'] = 'numeric'
        stats['mean'] = float(col.mean()) if pd.notna(col.mean()) else None
        stats['std_dev'] = float(col.std()) if pd.notna(col.std()) else None
        stats['min'] = float(col.min()) if pd.notna(col.min()) else None
        stats['max'] = float(col.max()) if pd.notna(col.max()) else None
        stats['25%'] = float(col.quantile(0.25)) if pd.notna(col.quantile(0.25)) else None
        stats['50%'] = float(col.quantile(0.50)) if pd.notna(col.quantile(0.50)) else None
        stats['75%'] = float(col.quantile(0.75)) if pd.notna(col.quantile(0.75)) else None
    else:
        stats['type'] = 'categorical'
        stats['unique_values'] = int(col.nunique())
        # Get top 5 value counts, ensuring keys are strings
        value_counts = col.value_counts().nlargest(5)
        stats['top_values'] = {str(k): int(v) for k, v in value_counts.items()}

    return stats

def detect_outliers(df, col, method="iqr", domain_specific=False):
    """
    More flexible outlier detection with multiple methods
    
    methods: "iqr", "zscore", "modified_zscore", "percentile", "dbscan"
    """
    if method == "iqr":
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1  # This is correct - IQR is Q3 minus Q1
        
        # Dynamic IQR multiplier based on distribution skewness
        skew = df[col].skew()
        # For highly skewed data, use higher threshold to avoid over-flagging
        iqr_multiplier = 3.0
        if abs(skew) > 2:
            iqr_multiplier = 4.0
        elif abs(skew) > 1:
            iqr_multiplier = 3.5
            
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
    elif method == "zscore":
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        
    elif method == "modified_zscore":
        # More robust to extreme outliers
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        lower_bound = median - 3.5 * mad / 0.6745
        upper_bound = median + 3.5 * mad / 0.6745
    
    elif method == "percentile":
        lower_bound = df[col].quantile(0.01)  # Bottom 1%
        upper_bound = df[col].quantile(0.99)  # Top 1%
        
    # Domain-specific handling for financial data
    if domain_specific:
        if "price" in col.lower() or "rate" in col.lower():
            # For prices, only set lower bound at 0 (can't be negative)
            lower_bound = max(0, lower_bound)
        elif "percent" in col.lower() or "ratio" in col.lower():
            # For percentages/ratios, bound between 0 and 100
            lower_bound = max(0, lower_bound)
            upper_bound = min(100, upper_bound)
    
    # Get outlier mask
    mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    return mask, lower_bound, upper_bound

def apply_weights(df, weight_column=None, log_callback=None):
    """Apply weights to a dataframe for analysis"""
    
    def log(message):
        if log_callback:
            log_callback(message)
    
    # Auto-detect weight columns if none provided
    if weight_column is None:
        potential_weight_cols = [col for col in df.columns if 
                               any(kw in col.lower() for kw in 
                                  ["weight", "multiplier", "factor", "wgt"])]
        
        if potential_weight_cols:
            weight_column = potential_weight_cols[0]
            log(f"🔢 Auto-detected weight column: '{weight_column}'")
        else:
            log("ℹ️ No weight column found or specified")
            return df, None
    
    # Validate weight column
    if weight_column not in df.columns:
        log(f"⚠️ Specified weight column '{weight_column}' not found")
        return df, None
    
    # Create normalized weights
    if df[weight_column].min() < 0:
        log(f"⚠️ Weight column '{weight_column}' contains negative values")
        return df, None
    
    # Handle zero weights
    if (df[weight_column] == 0).any():
        log(f"⚠️ Weight column '{weight_column}' contains zero values - replacing with minimum non-zero")
        min_non_zero = df[weight_column][df[weight_column] > 0].min()
        df[weight_column] = df[weight_column].replace(0, min_non_zero)
    
    # Create normalized weights that sum to sample size
    n = len(df)
    sum_weights = df[weight_column].sum()
    df['normalized_weight'] = df[weight_column] * (n / sum_weights)
    
    log(f"✅ Applied weights from '{weight_column}'. Sum of weights = {sum_weights:.2f}")
    log(f"✅ Created normalized weights (sum = {n})")
    
    # Calculate effective sample size
    ess = (df['normalized_weight'].sum() ** 2) / (df['normalized_weight'] ** 2).sum()
    design_effect = n / ess
    
    log(f"📊 Effective sample size: {ess:.1f} ({ess/n:.1%} of original)")
    log(f"📊 Design effect: {design_effect:.2f}")
    
    return df, 'normalized_weight'

def process_data(df, prompt, model, log_callback=None):
    narration = []
    
    # Initialize the narrator
    narrator = Narrator(model)
    
    # Helper to both add to narration and send real-time logs
    def log(message):
        narration.append(message)
        if log_callback:
            log_callback(message)
    
    # Send immediate welcome message
    log("👋 Starting data analysis! I'll clean your data and explain what I find.")
    
    # Store AI enhancement requests for later
    ai_enhancement_requests = []
    
    # Function to enhance logs with AI explanations after core processing is done
    def enhance_logs_with_ai(requests):
        if not model or not model.initialized:
            return
            
        log("🧠 Enhancing explanations with AI insights...")
        for request_id, prompt, log_index, prefix in requests:
            try:
                # Adjust how the response is accessed to handle the list-of-lists format
                response_list = model.pipe(prompt, return_full_text=False)
                if response_list and response_list[0]:
                    response = response_list[0][0]['generated_text']
                    enhanced_message = f"{prefix}: {response.strip()}"
                    # Add the enhanced explanation to the logs
                    if log_callback:
                        log_callback(enhanced_message)
                    narration.append(enhanced_message)
            except Exception as e:
                # Silently fail - we already have fallback explanations
                log(f"⚠️ AI enhancement failed for '{request_id}': {e}")
                pass
    
    # Calculate file size
    memory_usage = df.memory_usage(deep=True).sum()
    file_size_mb = memory_usage / (1024 * 1024)
    log(f"📊 File size in memory: {file_size_mb:.2f} MB")

    # 1. Original Shape and Columns
    original_shape = df.shape
    log(f"📋 Original data shape: {original_shape[0]} rows × {original_shape[1]} columns")
    log(f"🔡 Columns in dataset: {list(df.columns)}")
    
    # Add a human-friendly dataset description (rule-based first)
    dataset_description = narrator._fallback_dataset_description(df)
    log(f"🧠 NARRATOR: {dataset_description}")
    
    # Queue up AI enhancement for later
    if model and model.initialized:
        rows, cols = df.shape
        dtypes = df.dtypes.value_counts().to_dict()
        num_numeric = sum(count for dtype, count in dtypes.items() if pd.api.types.is_numeric_dtype(dtype))
        num_categorical = sum(count for dtype, count in dtypes.items() if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype))
        
        prompt = f"""
        Write a brief, friendly paragraph describing this dataset. Use simple, non-technical language:
        
        - Dataset has {rows} rows and {cols} columns
        - There are {num_numeric} numeric columns and {num_categorical} text/categorical columns
        - Column names: {', '.join(df.columns.tolist())}
        
        Your explanation (1-2 sentences, friendly tone):
        """
        ai_enhancement_requests.append(("dataset_desc", prompt, len(narration)-1, "🧠 AI INSIGHT"))

    # Track changes made
    changes_made = {
        "missing_values_filled": {},
        "outliers_removed": {},
        "column_classifications": {}
    }

    # 2. Handle missing values
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    total_missing = missing_summary.sum()
    log(f"❓ Total missing values in dataset: {total_missing}")
    
    if len(missing_cols) > 0:
        log("Missing values by column:")
        for col, count in missing_cols.items():
            log(f"  - {col}: {count} missing values ({count/len(df):.1%})")
            
    # Separate numeric and categorical columns
    log("🧠 Using AI to classify columns based on names...")
    column_names = list(df.columns)
    ai_classifications = model.classify_columns_with_ai(column_names, log_callback=log)

    # Track column classifications
    for col, classification in ai_classifications.items():
        changes_made["column_classifications"][col] = classification
        log(f"🏷️ AI classified '{col}' as {classification['type']}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # For columns that weren't classified by name, use detailed analysis
    special_numeric_cols = []
    regular_numeric_cols = []

    for col in numeric_cols:
        # If already classified by AI name analysis, use that classification
        if ai_classifications[col]["is_special"]:
            special_numeric_cols.append(col)
        else:
            regular_numeric_cols.append(col)
    log(f"Found {len(special_numeric_cols)} special numeric columns and {len(regular_numeric_cols)} regular numeric columns")

    if total_missing > 0:
        # Check if dataset is large (more than 10,000 rows)
        is_large_dataset = df.shape[0] > 10000
        log(f"Dataset size: {df.shape[0]} rows ({'Large' if is_large_dataset else 'Small'})")
        
        # --- Impute numerical with median (faster than KNN) ---
        if numeric_cols:
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    missing_count = df[col].isnull().sum()
                    median_val = df[col].median()
                    df.loc[:, col] = df[col].fillna(median_val)
                    log(f"Filled {missing_count} missing values in numeric column '{col}' with median: {median_val}")
                    changes_made["missing_values_filled"][col] = {
                        "count": int(missing_count),
                        "method": "median",
                        "value": float(median_val)
                    }
                    
                    # Add immediate explanation for missing values strategy
                    explanation = narrator._fallback_missing_explanation(col, "median", median_val)
                    log(f"🧠 NARRATOR: {explanation}")

        # --- Impute categorical with mode (kept as is) ---
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                missing_count = df[col].isnull().sum()
                mode_value = df[col].mode()[0]
                df.loc[:, col] = df[col].fillna(mode_value)
                log(f"Filled {missing_count} missing values in categorical column '{col}' with mode: {mode_value}")
                changes_made["missing_values_filled"][col] = {
                    "count": int(missing_count),
                    "method": "mode",
                    "value": str(mode_value)
                }
                
                # Add immediate explanation for missing values strategy
                explanation = narrator._fallback_missing_explanation(col, "mode", mode_value)
                log(f"🧠 NARRATOR: {explanation}")
    else:
        log("No missing values found.")

    # 3. Smart Outlier Detection and Handling
    original_row_count = df.shape[0]
    outliers_detected = {}
    total_outliers_removed = 0

    # Create a copy of the original dataframe
    df_clean = df.copy()

    log("🔍 Analyzing potential outliers before taking action...")

    # Choose the best outlier handling strategy for each column
    for col in regular_numeric_cols:
        # Get basic distribution stats
        skew = df[col].skew()
        kurtosis = df[col].kurtosis()
        
        # Determine appropriate method based on distribution
        if abs(skew) > 2:
            # Highly skewed data - use modified z-score
            method = "modified_zscore"
        elif kurtosis > 5:
            # Heavy-tailed data - use percentile method
            method = "percentile"
        else:
            # Normal-ish data - use standard IQR
            method = "iqr"
            
        # Check for domain-specific considerations
        domain_specific = any(keyword in col.lower() for keyword in 
                             ["price", "rate", "percent", "ratio", "amount"])
        
        # Detect outliers with the chosen method
        col_outlier_mask, lower_bound, upper_bound = detect_outliers(
            df, col, method=method, domain_specific=domain_specific)
        outlier_count = col_outlier_mask.sum()
        
        if outlier_count > 0:
            percentage = (outlier_count/len(df))*100
            
            # Decide whether to winsorize or remove
            use_winsorization = False
            
            # Time-series financial data usually prefers winsorization
            if any(keyword in col.lower() for keyword in ["price", "rate", "return"]):
                use_winsorization = True
                log(f"🔧 Using winsorization for financial column '{col}'")
            # High percentage of outliers
            elif percentage > 5:
                use_winsorization = True
                log(f"🔧 Using winsorization due to high outlier percentage ({percentage:.1f}%)")
            # Important ID or category-like numeric column
            elif col in special_numeric_cols:
                use_winsorization = True
                log(f"🔧 Using winsorization for special column '{col}'")
            
            # Record outlier info
            outliers_detected[col] = {
                "count": int(outlier_count),
                "percentage": round(percentage, 2),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "method": method,
                "handling": "winsorized" if use_winsorization else "removed"
            }
            
            if use_winsorization:
                # Winsorize: cap values at the bounds
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                log(f"Winsorized {outlier_count} outliers in column '{col}'")
            else:
                # Remove rows with outliers
                df_clean = df_clean[~col_outlier_mask]
                log(f"Removed {outlier_count} outliers in column '{col}'")
            
            changes_made["outliers_removed"][col] = int(outlier_count)

    # Replace original df with clean version if changes were made
    if total_outliers_removed > 0:
        log(f"Applied outlier handling strategy to the dataset")
        df = df_clean
    else:
        log("No outliers were handled in the dataset.")

    if not outliers_detected:
        log("No outliers detected in numeric columns.")
        
    # Report on special columns that were skipped
    if special_numeric_cols:
        log("\nThe following special numeric columns were not subject to outlier detection:")
        for col in special_numeric_cols:
            col_type = changes_made["column_classifications"][col]["type"]
            log(f"  - {col}: Identified as {col_type}")

    # Apply weights if needed
    weight_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in ["weight", "multiplier", "wgt"]):
            weight_col = col
            break

    if weight_col:
        log(f"🔢 Found potential weight column: '{weight_col}'")
        df, weight_var = apply_weights(df, weight_col, log)
    else:
        weight_var = None

    # Generate descriptive statistics for report
    log("📊 Generating descriptive statistics for report...")

    # Basic statistics for numeric columns (weighted if applicable)
    numeric_stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        # Skip the weight column itself
        if col == weight_col or col == 'normalized_weight':
            continue
            
        if weight_var:
            # Weighted statistics
            weighted_mean = np.average(df[col], weights=df[weight_var])
            weighted_var = np.average((df[col] - weighted_mean)**2, weights=df[weight_var])
            weighted_std = np.sqrt(weighted_var)
            
            numeric_stats[col] = {
                "mean": float(weighted_mean),
                "std": float(weighted_std),
                "min": float(df[col].min()),
                "25%": float(np.percentile(df[col], 25)),  # Simple percentile - could be weighted
                "median": float(np.percentile(df[col], 50)),
                "75%": float(np.percentile(df[col], 75)),
                "max": float(df[col].max()),
                "weighted": True
            }
        else:
            # Unweighted statistics
            numeric_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "25%": float(df[col].quantile(0.25)),
                "median": float(df[col].median()),
                "75%": float(df[col].quantile(0.75)),
                "max": float(df[col].max()),
                "weighted": False
            }

    # Categorical column distributions
    categorical_stats = {}
    for col in df.select_dtypes(exclude=[np.number]).columns:
        value_counts = df[col].value_counts(normalize=True).to_dict()
        top_categories = {str(k): float(v) for k, v in sorted(value_counts.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True)[:10]}
        categorical_stats[col] = {
            "top_categories": top_categories,
            "unique_values": int(df[col].nunique())
        }

    # 4. Dataset After Cleaning
    final_shape = df.shape
    log(f"Final data shape after cleaning: {final_shape[0]} rows × {final_shape[1]} columns")
    
    # Add immediate summary of all changes
    summary_explanation = narrator._fallback_summary(original_shape, final_shape, total_outliers_removed, 
                                             len(changes_made["missing_values_filled"]), 
                                             len(changes_made["outliers_removed"]))
    log(f"🧠 NARRATOR SUMMARY: {summary_explanation}")
    
    # Queue up AI enhancement for summary
    if model and model.initialized:
        num_filled_cols = len(changes_made.get("missing_values_filled", {}))
        num_outlier_cols = len(changes_made.get("outliers_removed", {}))
        prompt = f"""
        Summarize for a non-technical person what cleaning we performed on the dataset:
        
        - Started with {original_shape[0]} rows and {original_shape[1]} columns
        - Ended with {final_shape[0]} rows and {final_shape[1]} columns
        - Filled missing values in {num_filled_cols} columns
        - Removed outliers from {num_outlier_cols} columns
        - Total rows removed: {total_outliers_removed}
        
        Your explanation (2-3 friendly sentences, avoid technical terms):
        """
        ai_enhancement_requests.append(("summary", prompt, len(narration)-1, "🧠 AI FINAL SUMMARY"))
    
    # Final data size
    final_memory_usage = df.memory_usage(deep=True).sum()
    final_file_size_mb = final_memory_usage / (1024 * 1024)
    
    # Calculate size difference
    size_diff = file_size_mb - final_file_size_mb
    if abs(size_diff) > 0.01:  # If difference is more than 0.01 MB
        log(f"Final file size in memory: {final_file_size_mb:.2f} MB ({'+' if size_diff < 0 else '-'}{abs(size_diff):.2f} MB)")
    else:
        log(f"Final file size in memory: {final_file_size_mb:.2f} MB (unchanged)")

    # Log that core processing is complete
    log("✅ Data processing complete! Your data is ready.")
    
    # Define function to safely generate a single sentence or small text chunk
    def generate_micro_chunk(prompt, chunk_name, max_tokens=75):
        """Generate a very small chunk of text safely"""
        # Add a check for empty prompt before calling the model
        if not prompt or not prompt.strip():
            log(f"⚠️ Skipping {chunk_name} generation due to empty prompt.")
            return f"[Could not generate {chunk_name} - prompt was empty]"
            
        try:
            # Use very limited token count to avoid memory issues
            result = model.pipe(
                prompt, 
                return_full_text=False, 
                max_new_tokens=max_tokens
            )[0]['generated_text']
            log(f"✓ Generated {chunk_name} successfully")
            return result.strip()
        except Exception as e:
            log(f"⚠️ Error on {chunk_name}: {str(e)}...")
            return f"[Could not generate {chunk_name}]"

    # Start a background thread to enhance logs with AI if model is available
    if model and model.initialized and ai_enhancement_requests:
        log("🧠 Now enhancing explanations with AI insights... (this happens in the background)")
        ai_thread = threading.Thread(
            target=enhance_logs_with_ai,
            args=(ai_enhancement_requests,)
        )
        ai_thread.daemon = True
        ai_thread.start()

    # 5. Package and return summary
    summary = {
        "prompt_used": prompt,
        "original_shape": {
            "rows": original_shape[0],
            "columns": original_shape[1]
        },
        "final_shape": {
            "rows": final_shape[0], 
            "columns": final_shape[1]
        },
        "file_size": {
            "original_mb": round(file_size_mb, 2),
            "final_mb": round(final_file_size_mb, 2)
        },
        "columns": list(df.columns),
        "sample_data": df.head(10).to_dict(orient="records"),
        "narration": narration,
        "changes_made": changes_made,
        "outliers_detected": outliers_detected,
        "rows_removed": total_outliers_removed,
        "special_columns": special_numeric_cols,
        "narrator_summary": summary_explanation,
        "descriptive_statistics": {
            "numeric": numeric_stats,
            "categorical": categorical_stats,
            "weighted_analysis": weight_var is not None
        },
        # Add the cleaned data for download functionality
        "processed_data": df_clean.to_dict(orient="records")
    }

    # 6. Generate AI-powered report if model is available
    if model and model.initialized:
        log("🧠 Generating AI-powered report...")
        
        # --- Create more detailed prompts that include data statistics ---

        # Helper to format stats for the prompt
        def format_stats_for_prompt(stats_dict, max_items=5):
            formatted_lines = []
            for col, stats in list(stats_dict.items())[:max_items]:
                if 'mean' in stats: # Numeric stat
                    line = f"- {col}: (avg: {stats.get('mean', 0):.2f}, min: {stats.get('min', 0):.2f}, max: {stats.get('max', 0):.2f})"
                else: # Categorical stat
                    top_cat = next(iter(stats.get('top_categories', {})), "N/A")
                    line = f"- {col}: (top category: {top_cat}, unique values: {stats.get('unique_values', 0)})"
                formatted_lines.append(line)
            if len(stats_dict) > max_items:
                formatted_lines.append(f"- ... and {len(stats_dict) - max_items} more variables.")
            return "\n".join(formatted_lines)

        numeric_summary_for_prompt = format_stats_for_prompt(numeric_stats)
        categorical_summary_for_prompt = format_stats_for_prompt(categorical_stats)

        # Prompts designed to elicit insights from the data, not just the process
        prompts_to_run = {
            "welcome": "Write ONE welcoming sentence for a data analysis report.",
            "dataset_desc": f"Write ONE sentence describing a dataset with {original_shape[0]} rows and {original_shape[1]} columns.",
            "purpose": f"Write ONE sentence stating that this analysis aims to address: '{prompt}'",
            "missing_values": f"Write ONE sentence stating that we handled missing values in {len(changes_made.get('missing_values_filled', {}))} columns.",
            "outliers": f"Write ONE sentence explaining that we processed outliers in {len(changes_made.get('outliers_removed', {}))} columns, resulting in {df_clean.shape[0]} rows of clean data.",
            "findings_intro": f"Write ONE introductory sentence about findings from analyzing {len(numeric_stats)} numeric and {len(categorical_stats)} categorical variables.",
            
            # --- The Key Change: Prompts with Data Context ---
            "finding_1": f"""
            Based on this user request: '{prompt}'
            And these key numeric variables:
            {numeric_summary_for_prompt}
            And these key categorical variables:
            {categorical_summary_for_prompt}
            Write ONE specific, data-driven insight. Be concise and mention a specific variable or number.
            """,
            "finding_2": f"""
            Based on this user request: '{prompt}'
            And these key numeric variables:
            {numeric_summary_for_prompt}
            And these key categorical variables:
            {categorical_summary_for_prompt}
            Write a SECOND specific, data-driven insight that is different from the first one.
            """,
            "finding_3": f"""
            Based on this user request: '{prompt}'
            And these key numeric variables:
            {numeric_summary_for_prompt}
            And these key categorical variables:
            {categorical_summary_for_prompt}
            Write a THIRD specific, data-driven insight that is different from the others.
            """,
            
            "conclusion_summary": "Write ONE concluding sentence summarizing this data analysis.",
            "next_steps": f"Write ONE sentence suggesting a next step related to this request: '{prompt}'"
        }

        prompt_keys = list(prompts_to_run.keys())
        prompt_texts = list(prompts_to_run.values())

        # Run all prompts in a single batch
        try:
            generated_results = model.pipe(
                prompt_texts,
                return_full_text=False,
                max_new_tokens=100  # Increased token limit for more detailed insights
            )

            # Map results back to keys
            report_sections = {}
            for i, key in enumerate(prompt_keys):
                if generated_results[i] and generated_results[i][0]:
                    report_sections[key] = generated_results[i][0]['generated_text'].strip()
                else:
                    report_sections[key] = f"[AI generation failed for {key}]"

            # Assemble the final report string
            report = f"""
### Introduction
{report_sections.get('welcome', '')} This report details the analysis of a dataset with {original_shape[0]} rows and {original_shape[1]} columns.
The primary goal is to {report_sections.get('purpose', '').lower().replace('this analysis aims to address:', '').strip()}.

### Data Preparation
{report_sections.get('missing_values', '')} {report_sections.get('outliers', '')}

### Key Findings
{report_sections.get('findings_intro', '')}
- {report_sections.get('finding_1', '')}
- {report_sections.get('finding_2', '')}
- {report_sections.get('finding_3', '')}

### Conclusion
{report_sections.get('conclusion_summary', '')}

### Next Steps
{report_sections.get('next_steps', '')}
"""
            summary["report"] = report
            log("✅ AI report generated successfully.")

        except Exception as e:
            log(f"❌ Failed to generate AI report: {e}")
            summary["report"] = "An error occurred during AI report generation."

    return df_clean, summary, narration
