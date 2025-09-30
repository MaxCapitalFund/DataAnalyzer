from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import base64
import uvicorn
import math

app = FastAPI()

def clean_nan_values(obj):
    """Recursively clean NaN values from dictionaries and lists"""
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    else:
        return obj

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/analyze")
async def analyze_csv(
    file: UploadFile = File(...),
    timeframe: str = Form(default="180d:15m"),
    capital: float = Form(default=2500.0),
    commission: float = Form(default=4.04)
):
    """Analyze uploaded CSV file using the Python backtester"""
    
    try:
        # Read uploaded file content
        csv_content = await file.read()
        csv_text = csv_content.decode('utf-8')
        
        # Create temporary file
        temp_dir = Path(tempfile.gettempdir())
        csv_file = temp_dir / f"temp_{os.getpid()}.csv"
        
        with open(csv_file, 'w') as f:
            f.write(csv_text)
        
        # Determine script path and working directory based on environment
        if os.environ.get('VERCEL'):
            script_path = Path('/var/task') / 'Backtester_vercel.py'
            # Use /var/task as working directory for Vercel
            run_cwd = Path('/var/task')
            # Set matplotlib cache directory to /tmp for Vercel
            os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
        else:
            script_path = Path(__file__).parent / 'Backtester_vercel.py'
            run_cwd = Path(__file__).parent
            # Ensure Backtests directory exists
            backtests_root = run_cwd / 'Backtests'
            backtests_root.mkdir(parents=True, exist_ok=True)
        
        # Run the Python backtester
        cmd = [
            sys.executable,
            str(script_path),
            "--csv", str(csv_file),
            "--timeframe", timeframe,
            "--capital", str(capital),
            "--commission", str(commission)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=run_cwd)
        
        if result.returncode != 0:
            print(f"Python script stdout: {result.stdout}")
            print(f"Python script stderr: {result.stderr}")
            
            # Check for specific common errors and provide helpful messages
            if "KeyError: 'EntryTime'" in result.stderr:
                raise Exception("CSV file format error: Missing required columns. Please ensure your CSV has 'Date/Time' column and proper trade data structure.")
            elif "Could not find 'Date/Time'" in result.stderr:
                raise Exception("CSV file format error: Missing 'Date/Time' column. Please check your CSV file format.")
            else:
                raise Exception(f"Analysis failed: {result.stderr}")
        
        # Find the most recent output directory
        if os.environ.get('VERCEL'):
            # In Vercel, look in /tmp for the output
            output_base_dir = Path('/tmp')
            # Find directories that start with "Backtests_"
            backtest_dirs = [d for d in output_base_dir.iterdir() if d.is_dir() and d.name.startswith('Backtests_')]
        else:
            # Local development - look in Backtests directory
            output_base_dir = run_cwd / 'Backtests'
            if not output_base_dir.exists():
                raise Exception(f'Backtests directory not found at {output_base_dir}')
            backtest_dirs = [d for d in output_base_dir.iterdir() if d.is_dir()]
        
        if not backtest_dirs:
            raise Exception('No backtest output directories found')
        
        latest_dir = max(backtest_dirs, key=lambda p: p.stat().st_mtime)
        
        # Read results
        results = {}
        
        # Read metrics
        metrics_file = latest_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics_data = json.load(f)
                # Clean NaN values from metrics
                results['metrics'] = clean_nan_values(metrics_data)
        
        # Read analytics markdown
        analytics_file = latest_dir / 'analytics.md'
        if analytics_file.exists():
            with open(analytics_file) as f:
                results['analytics_md'] = f.read()
        
        # Helper function to read CSV as JSON
        def read_csv_as_json(file_path):
            if not file_path.exists():
                return []
            df = pd.read_csv(file_path)
            # Clean NaN values from DataFrame
            df = df.fillna('')
            return df.to_dict('records')
        
        # Read all CSV data
        results['data'] = {
            'monthly_performance': read_csv_as_json(latest_dir / 'monthly_performance.csv'),
            'dow_kpis': read_csv_as_json(latest_dir / 'dow_kpis.csv'),
            'session_kpis': read_csv_as_json(latest_dir / 'session_kpis.csv'),
            'hold_kpis': read_csv_as_json(latest_dir / 'hold_kpis.csv'),
            'top_best_trades': read_csv_as_json(latest_dir / 'top_best_trades.csv'),
            'top_worst_trades': read_csv_as_json(latest_dir / 'top_worst_trades.csv'),
            'max_win_streak': read_csv_as_json(latest_dir / 'max_win_streak_trades.csv'),
            'max_loss_streak': read_csv_as_json(latest_dir / 'max_loss_streak_trades.csv')
        }
        
        # Helper function to read images as base64
        def read_image_as_base64(file_path):
            if not file_path.exists():
                return None
            with open(file_path, 'rb') as f:
                img_data = f.read()
                return f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
        
        # Read charts
        results['charts'] = {
            'equity_curve': read_image_as_base64(latest_dir / 'equity_curve_180d.png'),
            'drawdown_curve': read_image_as_base64(latest_dir / 'drawdown_curve.png'),
            'pl_histogram': read_image_as_base64(latest_dir / 'pl_histogram.png'),
            'heatmap': read_image_as_base64(latest_dir / 'heatmap_dow_hour_count.png')
        }
        
        # Count trades
        trades_file = latest_dir / 'trades_enriched.csv'
        if trades_file.exists():
            df = pd.read_csv(trades_file)
            results['trades_count'] = len(df)
            results['trades_rth_count'] = results['metrics'].get('num_trades', 0)
        else:
            results['trades_count'] = 0
            results['trades_rth_count'] = 0
        
        results['strategy_name'] = results['metrics'].get('strategy_name', 'Unknown')
        
        # Final cleanup of any remaining NaN values
        results = clean_nan_values(results)
        
        return JSONResponse(content=results)
        
    except Exception as e:
        print(f"Error in analyze_csv: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(csv_file):
            os.unlink(csv_file)

# For local development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
