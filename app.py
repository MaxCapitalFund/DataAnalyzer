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

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {"status": "ok", "message": "API is working", "timestamp": "2025-10-20T01:00:00Z"}

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
        
        debug_info = {
            "command": ' '.join(cmd),
            "working_directory": str(run_cwd),
            "script_exists": script_path.exists(),
            "script_path": str(script_path)
        }
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=run_cwd)
        
        debug_info.update({
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        })
        
        if result.returncode != 0:
            # Check for specific common errors and provide helpful messages
            if "KeyError: 'EntryTime'" in result.stderr:
                raise Exception(f"CSV file format error: Missing required columns. Debug: {debug_info}")
            elif "Could not find 'Date/Time'" in result.stderr:
                raise Exception(f"CSV file format error: Missing 'Date/Time' column. Debug: {debug_info}")
            else:
                raise Exception(f"Analysis failed: {result.stderr}. Debug: {debug_info}")
        
        # Find the most recent output directory
        if os.environ.get('VERCEL'):
            # In Vercel, look in /tmp for the output
            output_base_dir = Path('/tmp')
            debug_info["output_search"] = {
                "search_path": str(output_base_dir),
                "directory_exists": output_base_dir.exists()
            }
            if output_base_dir.exists():
                all_dirs = list(output_base_dir.iterdir())
                debug_info["output_search"]["all_directories"] = [d.name for d in all_dirs if d.is_dir()]
                # Find directories that start with "Backtests_"
                backtest_dirs = [d for d in all_dirs if d.is_dir() and d.name.startswith('Backtests_')]
                debug_info["output_search"]["backtest_directories"] = [d.name for d in backtest_dirs]
            else:
                backtest_dirs = []
        else:
            # Local development - look in /tmp for backtester v1.4.6
            output_base_dir = Path('/tmp')
            if not output_base_dir.exists():
                raise Exception(f'/tmp directory not found')
            backtest_dirs = [d for d in output_base_dir.iterdir() if d.is_dir() and d.name.startswith('Backtests_')]
        
        if not backtest_dirs:
            raise Exception(f'No backtest output directories found. Debug info: {debug_info}')
        
        latest_dir = max(backtest_dirs, key=lambda p: p.stat().st_mtime)
        
        # Read results
        results = {}
        
        # Read metrics - backtester v1.4.6 doesn't create metrics.json, extract from analytics.md
        metrics_file = latest_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics_data = json.load(f)
                # Clean NaN values from metrics
                results['metrics'] = clean_nan_values(metrics_data)
        else:
            # Extract metrics from analytics.md for backtester v1.4.6
            analytics_file = latest_dir / 'analytics.md'
            if analytics_file.exists():
                with open(analytics_file) as f:
                    analytics_content = f.read()
                    # Extract metrics from analytics.md
                    metrics = {}
                    lines = analytics_content.split('\n')
                    for line in lines:
                        if '**Total Trades:**' in line:
                            metrics['num_trades'] = int(line.split('**Total Trades:**')[1].strip())
                        elif '**Net Profit:**' in line:
                            profit_str = line.split('**Net Profit:**')[1].strip().replace('$', '').replace(',', '')
                            metrics['net_profit'] = float(profit_str)
                        elif '**Total Return:**' in line:
                            return_str = line.split('**Total Return:**')[1].strip().replace('%', '')
                            metrics['total_return_pct'] = float(return_str)
                        elif '**Win Rate:**' in line:
                            winrate_str = line.split('**Win Rate:**')[1].strip().split('%')[0]
                            metrics['win_rate_pct'] = float(winrate_str)
                        elif '**Profit Factor:**' in line:
                            pf_str = line.split('**Profit Factor:**')[1].strip()
                            if pf_str == 'inf':
                                metrics['profit_factor'] = float('inf')
                            else:
                                metrics['profit_factor'] = float(pf_str)
                        elif '**Max Drawdown:**' in line:
                            dd_str = line.split('**Max Drawdown:**')[1].strip().split('(')[0].replace('$', '').replace(',', '')
                            metrics['max_drawdown_dollars'] = float(dd_str)
                    results['metrics'] = clean_nan_values(metrics)
        
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
        
        # Read all CSV data - backtester v1.4.6 only creates trades_enriched.csv
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
        
        # For backtester v1.4.6, also add trades data
        trades_file = latest_dir / 'trades_enriched.csv'
        if trades_file.exists():
            trades_data = read_csv_as_json(trades_file)
            results['trades_rth_count'] = len(trades_data)
            if trades_data and len(trades_data) > 0:
                results['strategy_name'] = trades_data[0]['BaseStrategy']
        
        # Read detailed metrics if available
        detailed_metrics_file = latest_dir / 'detailed_metrics.json'
        if detailed_metrics_file.exists():
            with open(detailed_metrics_file) as f:
                detailed_metrics = json.load(f)
                results['detailed_metrics'] = detailed_metrics
        
        # Helper function to read images as base64
        def read_image_as_base64(file_path):
            if not file_path.exists():
                return None
            with open(file_path, 'rb') as f:
                img_data = f.read()
                return f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
        
        # Read charts - try new chart generation first, fallback to old files
        charts = {}
        
        # Try to read from new chart generation (base64 encoded in JSON)
        charts_json_file = latest_dir / 'charts.json'
        if charts_json_file.exists():
            with open(charts_json_file) as f:
                charts = json.load(f)
        else:
            # Fallback to old image files
            charts = {
                'equity_curve': read_image_as_base64(latest_dir / 'equity_curve_180d.png'),
                'drawdown_curve': read_image_as_base64(latest_dir / 'drawdown_curve.png'),
                'pl_histogram': read_image_as_base64(latest_dir / 'pl_histogram.png'),
                'heatmap': read_image_as_base64(latest_dir / 'heatmap_dow_hour_count.png')
            }
        
        results['charts'] = charts
        
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
