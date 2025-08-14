import io
import datetime
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.platypus import XPreformatted
from reportlab.lib.enums import TA_CENTER
from reportlab.graphics import renderPDF
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics.renderPDF import GraphicsFlowable
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import base64
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# Add imports for Plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import kaleido  # Required for plotly static image export

# Configure matplotlib to use non-GUI backend
plt.switch_backend('agg')

# Set a consistent, attractive color palette and style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
CMAP = "viridis"
PLOTLY_TEMPLATE = "plotly_white"  # Modern, clean template for plotly

# Add new function to create plotly figures and convert them to images for PDF
def create_plotly_figure(df, chart_type, **kwargs):
    """Create a plotly figure and convert to image for PDF inclusion"""
    
    if chart_type == "histogram":
        col = kwargs.get('col')
        fig = px.histogram(
            df, x=col, 
            marginal="box", 
            histnorm="probability density",
            color_discrete_sequence=[COLORS[0]],
            template=PLOTLY_TEMPLATE
        )
        
        # Add mean and median lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS[2],
                     annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top")
        fig.add_vline(x=median_val, line_dash="dot", line_color=COLORS[1],
                    annotation_text=f"Median: {median_val:.2f}", annotation_position="bottom")
        
        # Enhanced styling
        fig.update_layout(
            title=f"Distribution of {col}",
            title_font=dict(size=20, family="Arial", color="#333333"),
            xaxis_title=dict(text=col, font=dict(size=14, family="Arial")),
            yaxis_title=dict(text="Density", font=dict(size=14, family="Arial")),
            height=500,
            width=800,
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor='#EEEEEE', zeroline=True, zerolinecolor='#CCCCCC')
        fig.update_yaxes(showgrid=True, gridcolor='#EEEEEE')
    
    elif chart_type == "boxplot":
        cols = kwargs.get('cols', [])
        melted_df = pd.melt(df[cols].reset_index(), 
                          id_vars=['index'], 
                          value_vars=cols,
                          var_name='Variable', value_name='Value')
                          
        fig = px.box(melted_df, x="Variable", y="Value", 
                    points="outliers",
                    color="Variable", 
                    color_discrete_sequence=COLORS,
                    template=PLOTLY_TEMPLATE,
                    notched=True)  # Add notches for improved visual appearance
        
        # Enhanced styling
        fig.update_layout(
            title="Box Plots of Numeric Variables",
            title_font=dict(size=20, family="Arial", color="#333333"),
            xaxis_title=dict(text="", font=dict(size=14, family="Arial")),
            yaxis_title=dict(text="Value", font=dict(size=14, family="Arial")),
            height=500,
            width=800,
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#EEEEEE')
    
    elif chart_type == "scatter":
        x_col = kwargs.get('x_col')
        y_col = kwargs.get('y_col')
        
        # Calculate correlation
        corr = df[[x_col, y_col]].corr().iloc[0,1]
        
        fig = px.scatter(df, x=x_col, y=y_col, 
                       opacity=0.7,  # Slightly increased opacity
                       trendline="ols", 
                       color_discrete_sequence=[COLORS[0]],
                       template=PLOTLY_TEMPLATE)
        
        # Add correlation annotation with better styling
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            text=f"Correlation: {corr:.2f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="#333333",
            borderwidth=1,
            borderpad=6,
            font=dict(size=12, color="#333333", family="Arial")
        )
        
        # Enhanced styling
        fig.update_layout(
            title=f"Relationship: {x_col} vs {y_col}",
            title_font=dict(size=20, family="Arial", color="#333333"),
            xaxis_title=dict(text=x_col, font=dict(size=14, family="Arial")),
            yaxis_title=dict(text=y_col, font=dict(size=14, family="Arial")),
            height=500,
            width=800,
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor='#EEEEEE', zeroline=True, zerolinecolor='#CCCCCC')
        fig.update_yaxes(showgrid=True, gridcolor='#EEEEEE', zeroline=True, zerolinecolor='#CCCCCC')
    
    elif chart_type == "bar":
        col = kwargs.get('col')
        value_counts = df[col].value_counts().nlargest(8)
        
        fig = px.bar(
            x=value_counts.index, 
            y=value_counts.values,
            color_discrete_sequence=COLORS,
            template=PLOTLY_TEMPLATE,
            labels={"x": col, "y": "Count"}
        )
        
        # Add percentage labels with better styling
        total = value_counts.sum()
        for i, value in enumerate(value_counts):
            percentage = 100 * value / total
            fig.add_annotation(
                x=value_counts.index[i],
                y=value/2,
                text=f"{percentage:.1f}%",
                showarrow=False,
                font=dict(color="white", size=12, family="Arial Bold")
            )
        
        # Enhanced styling
        fig.update_layout(
            title=f"Distribution of {col}",
            title_font=dict(size=20, family="Arial", color="#333333"),
            xaxis_title=dict(text=col, font=dict(size=14, family="Arial")),
            yaxis_title=dict(text="Count", font=dict(size=14, family="Arial")),
            height=500,
            width=800,
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#EEEEEE')
    
    elif chart_type == "timeseries":
        date_col = kwargs.get('date_col')
        value_col = kwargs.get('value_col')
        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col])
        temp_df = temp_df.set_index(date_col)
        
        # Resample data to smooth it
        resampled = temp_df[value_col].resample('D').mean()
        
        fig = px.line(
            resampled, 
            color_discrete_sequence=[COLORS[0]],
            template=PLOTLY_TEMPLATE
        )
        
        # Add area under the line for better visualization
        fig.update_traces(
            fill='tozeroy',  # Fill to zero on y-axis
            fillcolor=f"rgba({int(COLORS[0][1:3], 16)}, {int(COLORS[0][3:5], 16)}, {int(COLORS[0][5:7], 16)}, 0.2)"  # Semi-transparent fill
        )
        
        # Enhanced styling
        fig.update_layout(
            title=f"Time Series: {value_col} over Time",
            title_font=dict(size=20, family="Arial", color="#333333"),
            xaxis_title=dict(text="Date", font=dict(size=14, family="Arial")),
            yaxis_title=dict(text=value_col, font=dict(size=14, family="Arial")),
            height=500,
            width=800,
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor='#EEEEEE', zeroline=True, zerolinecolor='#CCCCCC')
        fig.update_yaxes(showgrid=True, gridcolor='#EEEEEE', zeroline=True, zerolinecolor='#CCCCCC')
    
    try:
        # Convert to static image for PDF with increased DPI for better quality
        img_bytes = fig.to_image(format="png", width=800, height=500, scale=3)
        return io.BytesIO(img_bytes)
    except Exception as e:
        # Fallback to a simpler rendering if the first attempt fails
        print(f"Error rendering chart: {e}, trying fallback method")
        try:
            img_bytes = fig.to_image(format="png", width=800, height=500, scale=2, engine="kaleido")
            return io.BytesIO(img_bytes)
        except Exception as e2:
            print(f"Fallback rendering also failed: {e2}")
            # Create a simple error image
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, f"Chart rendering failed: {str(e)[:50]}...",
                   horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return buf

def run_statistical_tests(df):
    """Run statistical tests on the dataset and return results"""
    results = {
        "normality_tests": {},
        "correlation_tests": {},
        "categorical_tests": {},
    }
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Normality tests for numeric columns
    for col in numeric_cols[:10]:  # Limit to 10 columns
        # Skip columns with too few values
        if len(df[col].dropna()) < 3:
            continue
            
        # Shapiro-Wilk test (works best for n<5000)
        sample = df[col].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
            
        try:
            stat, p_value = stats.shapiro(sample)
            is_normal = p_value > 0.05
            results["normality_tests"][col] = {
                "test": "Shapiro-Wilk",
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": is_normal,
                "interpretation": "Normal distribution" if is_normal else "Not normally distributed"
            }
        except Exception as e:
            # Some distributions might cause the test to fail
            results["normality_tests"][col] = {
                "test": "Shapiro-Wilk",
                "error": str(e)
            }
    
    # Correlation tests between numeric columns
    if len(numeric_cols) >= 2:
        # Pearson correlation (parametric)
        try:
            pearson_corr = df[numeric_cols].corr(method='pearson')
            results["correlation_tests"]["pearson"] = {
                "matrix": pearson_corr.to_dict(),
                "interpretation": "Measures linear relationship between variables"
            }
            
            # Spearman correlation (non-parametric)
            spearman_corr = df[numeric_cols].corr(method='spearman')
            results["correlation_tests"]["spearman"] = {
                "matrix": spearman_corr.to_dict(),
                "interpretation": "Measures monotonic relationship, robust to outliers"
            }
        except Exception as e:
            results["correlation_tests"]["error"] = str(e)
    
    # Chi-square tests for categorical columns
    if len(categorical_cols) >= 2:
        for i, col1 in enumerate(categorical_cols[:5]):  # Limit to first 5 columns
            for col2 in categorical_cols[i+1:6]:  # And their pairs
                try:
                    # Create contingency table
                    contingency = pd.crosstab(df[col1], df[col2])
                    
                    # Run chi-square test if we have enough data
                    if contingency.size >= 4:  # At least a 2x2 table
                        chi2, p, dof, expected = stats.chi2_contingency(contingency)
                        is_independent = p > 0.05
                        
                        test_name = f"{col1} vs {col2}"
                        results["categorical_tests"][test_name] = {
                            "test": "Chi-square",
                            "statistic": float(chi2),
                            "p_value": float(p),
                            "dof": int(dof),
                            "is_independent": is_independent,
                            "interpretation": "Variables are independent" if is_independent else "Variables are not independent"
                        }
                except Exception as e:
                    # Skip if test fails
                    continue
    
    return results

def create_statistical_report(task_info):
    """
    Generate a PDF report with statistical analysis of the processed data.
    
    Args:
        task_info: Dictionary containing task information and processed data
    
    Returns:
        BytesIO object containing the PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=72, 
        leftMargin=72,
        topMargin=72, 
        bottomMargin=18
    )
    
    # Container for elements to be added to the document
    elements = []
    styles = getSampleStyleSheet()
    
    # Modify existing styles instead of adding new ones with the same name
    styles['Heading1'].fontSize = 18
    styles['Heading1'].spaceAfter = 12
    styles['Heading1'].textColor = colors.darkblue
    
    styles['Heading2'].fontSize = 14
    styles['Heading2'].spaceBefore = 12
    styles['Heading2'].spaceAfter = 6
    styles['Heading2'].textColor = colors.darkblue
    
    # Check if 'Heading3' exists in the stylesheet and modify it,
    # or add it if it doesn't exist
    if 'Heading3' in styles:
        styles['Heading3'].fontSize = 12
        styles['Heading3'].spaceBefore = 10
        styles['Heading3'].spaceAfter = 4
        styles['Heading3'].textColor = colors.darkblue
    else:
        styles.add(ParagraphStyle(
            name='Heading3',
            parent=styles['Heading2'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=4,
            textColor=colors.darkblue
        ))
    
    styles['BodyText'].fontSize = 11
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceBefore = 6
    
    # Title and timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph("Data Analysis Report", styles['Heading1']))
    elements.append(Paragraph(f"Generated on {current_time}", styles['BodyText']))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add summary information
    summary = task_info["summary"]
    
    elements.append(Paragraph("Dataset Overview", styles['Heading2']))
    
    # Dataset shape info
    original_rows = summary["original_shape"]["rows"]
    original_cols = summary["original_shape"]["columns"]
    final_rows = summary["final_shape"]["rows"]
    final_cols = summary["final_shape"]["columns"]
    
    shape_data = [
        ["Metric", "Original", "Final", "Difference"],
        ["Rows", original_rows, final_rows, original_rows - final_rows],
        ["Columns", original_cols, final_cols, original_cols - final_cols],
    ]
    
    shape_table = Table(shape_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1*inch])
    shape_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(shape_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Create dataframe from processed data
    if "processed_data" in task_info and "column_names" in task_info:
        df = pd.DataFrame(task_info["processed_data"], columns=task_info["column_names"])
        
        # Generate visualizations for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # Detect datetime columns for time series analysis
        datetime_cols = []
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    # Try to convert to datetime
                    pd.to_datetime(df[col], errors='raise')
                    datetime_cols.append(col)
            except:
                continue
        
        # Add Statistical Tests section
        elements.append(Paragraph("Statistical Tests", styles['Heading2']))
        
        # Run statistical tests
        test_results = run_statistical_tests(df)
        
        # Add normality test results
        if test_results["normality_tests"]:
            elements.append(Paragraph("Normality Tests (Shapiro-Wilk)", styles['Heading3']))
            elements.append(Paragraph("Tests whether data follows a normal distribution. P-value > 0.05 suggests normal distribution.", styles['BodyText']))
            
            # Create table for normality tests
            norm_data = [["Column", "Test Statistic", "P-Value", "Result"]]
            
            for col, results in test_results["normality_tests"].items():
                if "error" not in results:
                    norm_data.append([
                        col, 
                        f"{results['statistic']:.4f}", 
                        f"{results['p_value']:.4f}",
                        results['interpretation']
                    ])
            
            if len(norm_data) > 1:  # If we have actual test results
                norm_table = Table(norm_data, colWidths=[1.5*inch, 1.2*inch, 1*inch, 2*inch])
                norm_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(norm_table)
                elements.append(Spacer(1, 0.2*inch))
        
        # Add correlation results
        if "pearson" in test_results["correlation_tests"]:
            elements.append(Paragraph("Correlation Analysis", styles['Heading3']))
            elements.append(Paragraph("Pearson correlation measures linear relationships. Values range from -1 (perfect negative) to 1 (perfect positive).", styles['BodyText']))
            
            # Generate correlation heatmap
            plt.figure(figsize=(8, 6))
            corr_matrix = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap=CMAP,
                        vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title("Correlation Matrix (Pearson)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save the figure to a BytesIO object
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # Add the image to the PDF
            img = Image(img_buffer, width=6*inch, height=4.5*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        # Add chi-square test results
        if test_results["categorical_tests"]:
            elements.append(Paragraph("Chi-Square Tests for Independence", styles['Heading3']))
            elements.append(Paragraph("Tests whether categorical variables are independent. P-value < 0.05 suggests variables are related.", styles['BodyText']))
            
            # Create table for chi-square tests
            chi_data = [["Variables", "Chi² Statistic", "P-Value", "Result"]]
            
            for test_name, results in test_results["categorical_tests"].items():
                chi_data.append([
                    test_name, 
                    f"{results['statistic']:.4f}", 
                    f"{results['p_value']:.4f}",
                    results['interpretation']
                ])
            
            if len(chi_data) > 1:  # If we have actual test results
                chi_table = Table(chi_data, colWidths=[2*inch, 1.2*inch, 1*inch, 1.5*inch])
                chi_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(chi_table)
                elements.append(Spacer(1, 0.2*inch))
    
        # Add AI-generated report if available
        if "report" in summary:
            elements.append(Paragraph("AI Analysis", styles['Heading2']))
            
            # Process AI report sections
            ai_report = summary["report"]
            sections = ai_report.split("###")
            
            for section in sections[1:]:  # Skip first empty item
                lines = section.strip().split("\n")
                if lines:
                    section_title = lines[0].strip()
                    section_content = "\n".join(lines[1:]).strip()
                    
                    elements.append(Paragraph(section_title, styles['Heading3']))
                    
                    # Process bullet points separately
                    paragraphs = section_content.split("\n")
                    for para in paragraphs:
                        if para.strip():
                            elements.append(Paragraph(para, styles['BodyText']))
                            
                    elements.append(Spacer(1, 0.1*inch))
        
        # Create data visualizations
        elements.append(Paragraph("Enhanced Data Visualizations", styles['Heading2']))
        
        # Generate histograms and density plots for numeric columns
        if numeric_cols:
            elements.append(Paragraph("Distributions of Numeric Variables", styles['Heading3']))
            
            for i, col in enumerate(numeric_cols[:4]):  # Limit to first 4 numeric columns
                # Create histogram with Plotly instead of matplotlib
                img_buffer = create_plotly_figure(df, "histogram", col=col)
                
                # Add the image to the PDF
                img = Image(img_buffer, width=6.5*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
        
        # Generate boxplots for numeric columns
        if numeric_cols:
            elements.append(Paragraph("Box Plots (Distribution and Outliers)", styles['Heading3']))
            
            # Create boxplots with Plotly
            cols_to_plot = numeric_cols[:6]  # Limit to 6 columns
            img_buffer = create_plotly_figure(df, "boxplot", cols=cols_to_plot)
            
            # Add the image to the PDF
            img = Image(img_buffer, width=6.5*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        # Generate scatter plots for pairs of numeric columns
        if len(numeric_cols) >= 2:
            elements.append(Paragraph("Relationships Between Key Variables", styles['Heading3']))
            
            # Select top 2 numeric columns for demonstration
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            
            # Create scatter plot with Plotly
            img_buffer = create_plotly_figure(df, "scatter", x_col=x_col, y_col=y_col)
            
            # Add the image to the PDF
            img = Image(img_buffer, width=6.5*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        # Generate enhanced bar charts for categorical columns using Plotly
        if categorical_cols:
            elements.append(Paragraph("Categorical Variable Analysis", styles['Heading3']))
            
            for i, col in enumerate(categorical_cols[:3]):  # Limit to first 3 categorical columns
                # Create bar chart with Plotly instead of matplotlib
                img_buffer = create_plotly_figure(df, "bar", col=col)
                
                # Add the image to the PDF
                img = Image(img_buffer, width=6.5*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
    
        # Generate time series plots with Plotly if date columns found
        if datetime_cols:
            elements.append(Paragraph("Time Series Analysis", styles['Heading3']))
            
            for date_col in datetime_cols[:1]:  # Just use first date column
                try:
                    # Find a numeric column to plot
                    if numeric_cols:
                        value_col = numeric_cols[0]
                        
                        # Create time series plot with Plotly
                        img_buffer = create_plotly_figure(df, "timeseries", date_col=date_col, value_col=value_col)
                        
                        # Add the image to the PDF
                        img = Image(img_buffer, width=6.5*inch, height=4*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    # Skip if time series plotting fails
                    continue
    
        # Include statistical summary tables
        elements.append(Paragraph("Statistical Summary", styles['Heading2']))
        
        if "descriptive_statistics" in summary:
            stats = summary["descriptive_statistics"]
            
            # Numeric statistics - enhanced version
            if "numeric" in stats and stats["numeric"]:
                elements.append(Paragraph("Numeric Columns Summary", styles['Heading3']))
                
                # Create table data with more statistics
                numeric_data = [["Column", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max", "Skewness", "Kurtosis"]]
                
                for col, col_stats in list(stats["numeric"].items())[:10]:  # Limit to 10 columns
                    # Get additional statistics from DataFrame if possible
                    try:
                        skew = df[col].skew()
                        kurt = df[col].kurtosis()
                        q25 = df[col].quantile(0.25)
                        q75 = df[col].quantile(0.75)
                    except:
                        skew = "N/A"
                        kurt = "N/A"
                        q25 = "N/A"
                        q75 = "N/A"
                    
                    numeric_data.append([
                        col, 
                        f"{col_stats.get('mean', 'N/A'):.2f}", 
                        f"{col_stats.get('std', 'N/A'):.2f}",
                        f"{col_stats.get('min', 'N/A'):.2f}",
                        f"{q25:.2f}" if isinstance(q25, (int, float)) else "N/A",
                        f"{col_stats.get('median', 'N/A'):.2f}",
                        f"{q75:.2f}" if isinstance(q75, (int, float)) else "N/A",
                        f"{col_stats.get('max', 'N/A'):.2f}",
                        f"{skew:.2f}" if isinstance(skew, (int, float)) else "N/A",
                        f"{kurt:.2f}" if isinstance(kurt, (int, float)) else "N/A"
                    ])
                
                # Use smaller font for this table since it's wider
                numeric_table = Table(numeric_data, colWidths=[0.9*inch, 0.6*inch, 0.6*inch, 0.5*inch, 
                                                             0.5*inch, 0.5*inch, 0.5*inch, 0.5*inch,
                                                             0.6*inch, 0.6*inch])
                numeric_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 0), (-1, -1), 8)  # Smaller font for this wide table
                ]))
                
                elements.append(numeric_table)
                elements.append(Spacer(1, 0.2*inch))
                
                # Add explanation of statistics
                elements.append(Paragraph("Statistical Measures Explained:", styles['BodyText']))
                elements.append(Paragraph("• <b>Skewness</b>: Measures asymmetry of the distribution. Values > 0 indicate right skew, < 0 indicate left skew.", styles['BodyText']))
                elements.append(Paragraph("• <b>Kurtosis</b>: Measures 'tailedness' of the distribution. Higher values indicate heavier tails than normal distribution.", styles['BodyText']))
                elements.append(Spacer(1, 0.1*inch))
            
            # Categorical statistics
            if "categorical" in stats and stats["categorical"]:
                elements.append(Paragraph("Categorical Columns Summary", styles['Heading3']))
                
                # Create table data
                cat_data = [["Column", "Unique Values", "Top Category", "Top Count", "Top %", "Missing"]]
                
                for col, col_stats in list(stats["categorical"].items())[:10]:  # Limit to 10 columns
                    # Get top category and its count
                    top_category = next(iter(col_stats.get("top_categories", {}).keys()), "N/A")
                    top_count = col_stats.get("top_categories", {}).get(top_category, 0) if top_category != "N/A" else 0
                    
                    # Calculate percentage
                    total = sum(col_stats.get("top_categories", {}).values()) if col_stats.get("top_categories") else 0
                    top_pct = (top_count / total * 100) if total > 0 else 0
                    
                    # Calculate missing values
                    missing = df[col].isna().sum()
                    missing_pct = (missing / len(df) * 100)
                    
                    cat_data.append([
                        col,
                        str(col_stats.get("unique_values", "N/A")),
                        top_category[:20],  # Truncate long category names
                        str(top_count),
                        f"{top_pct:.1f}%",
                        f"{missing} ({missing_pct:.1f}%)"
                    ])
                
                cat_table = Table(cat_data, colWidths=[1.5*inch, 0.8*inch, 1.5*inch, 0.8*inch, 0.7*inch, 1*inch])
                cat_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(cat_table)
                elements.append(Spacer(1, 0.2*inch))
        
        # Add information about data cleaning steps
        elements.append(Paragraph("Data Cleaning Summary", styles['Heading2']))
        
        # Missing values filled
        if "changes_made" in summary and "missing_values_filled" in summary["changes_made"]:
            missing_filled = summary["changes_made"]["missing_values_filled"]
            if missing_filled:
                elements.append(Paragraph(f"Missing Values Filled: {len(missing_filled)} columns", styles['Heading3']))
                
                # Create table for missing values
                missing_data = [["Column", "Count", "% of Data", "Method", "Value"]]
                
                for col, details in list(missing_filled.items())[:10]:  # Limit to 10 columns
                    # Calculate percentage
                    count = details.get("count", 0)
                    pct = (count / len(df) * 100) if len(df) > 0 else 0
                    
                    missing_data.append([
                        col,
                        str(count),
                        f"{pct:.2f}%",
                        details.get("method", "N/A"),
                        str(details.get("value", "N/A"))[:20]  # Truncate long values
                    ])
                
                missing_table = Table(missing_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 1*inch, 1.5*inch])
                missing_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(missing_table)
                elements.append(Spacer(1, 0.2*inch))
                
                # Add missing values visualization
                if len(missing_filled) > 1:
                    plt.figure(figsize=(8, 4))
                    
                    # Extract counts and columns
                    cols = list(missing_filled.keys())[:10]  # Limit to 10
                    counts = [missing_filled[col].get("count", 0) for col in cols]
                    
                    # Create bar chart
                    sns.barplot(x=cols, y=counts, palette=COLORS)
                    plt.title("Missing Values by Column", fontsize=14, fontweight='bold')
                    plt.xlabel("Columns", fontsize=12)
                    plt.ylabel("Missing Count", fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
                    sns.despine(left=False, bottom=False)
                    plt.tight_layout()
                    
                    # Save the figure to a BytesIO object
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    plt.close()
                    
                    # Add the image to the PDF
                    img = Image(img_buffer, width=6*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
        
        # Outliers removed
        if "outliers_detected" in summary:
            outliers = summary["outliers_detected"]
            if outliers:
                elements.append(Paragraph(f"Outliers Handled: {len(outliers)} columns", styles['Heading3']))
                
                # Create table for outliers
                outlier_data = [["Column", "Count", "Percentage", "Method", "Handling"]]
                
                for col, details in list(outliers.items())[:10]:  # Limit to 10 columns
                    outlier_data.append([
                        col,
                        str(details.get("count", "N/A")),
                        f"{details.get('percentage', 0):.2f}%",
                        details.get("method", "N/A"),
                        details.get("handling", "N/A")
                    ])
                
                outlier_table = Table(outlier_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 1*inch, 1.5*inch])
                outlier_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(outlier_table)
                elements.append(Spacer(1, 0.2*inch))
                
                # Add outliers visualization if there are enough columns with outliers
                if len(outliers) > 1:
                    plt.figure(figsize=(8, 4))
                    
                    # Extract counts and columns
                    cols = list(outliers.keys())[:8]  # Limit to 8
                    counts = [outliers[col].get("count", 0) for col in cols]
                    percentages = [outliers[col].get("percentage", 0) for col in cols]
                    
                    # Create dual-axis plot
                    fig, ax1 = plt.subplots(figsize=(8, 4))
                    
                    # Bar chart for counts
                    ax1.bar(cols, counts, color=COLORS[0], alpha=0.6, label='Count')
                    ax1.set_xlabel("Columns", fontsize=12)
                    ax1.set_ylabel("Outlier Count", fontsize=12, color=COLORS[0])
                    ax1.tick_params(axis='y', labelcolor=COLORS[0])
                    
                    # Line plot for percentages
                    ax2 = ax1.twinx()
                    ax2.plot(cols, percentages, 'o-', color=COLORS[2], linewidth=2, label='Percentage')
                    ax2.set_ylabel("Percentage (%)", fontsize=12, color=COLORS[2])
                    ax2.tick_params(axis='y', labelcolor=COLORS[2])
                    
                    # Add title and legend
                    plt.title("Outliers by Column", fontsize=14, fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add a legend
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                    
                    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
                    plt.tight_layout()
                    
                    # Save the figure to a BytesIO object
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    plt.close()
                    
                    # Add the image to the PDF
                    img = Image(img_buffer, width=6*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(f"Report generated in {task_info['execution_time']} • Powered by Statathon", styles['BodyText']))
    
    # Build the PDF document
    doc.build(elements)
    buffer.seek(0)
    return buffer

def create_basic_report(task_id, error_message):
    """Generate a basic error report when data is not available"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    elements = []
    elements.append(Paragraph("Error Report", styles['Heading1']))
    elements.append(Paragraph(f"Task ID: {task_id}", styles['BodyText']))
    elements.append(Paragraph(f"Error: {error_message}", styles['BodyText']))
    elements.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['BodyText']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer
