import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import base64

def handler(request):
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'Method not allowed'})
        }

    try:
        # Parse request body
        body = json.loads(request.body)
        timeframe = body.get('timeframe', '180d:15m')
        capital = body.get('capital', 2500.0)
        commission = body.get('commission', 4.04)
        csv_data = body.get('csvData', '')

        # Create a temporary CSV file
        temp_dir = Path(tempfile.gettempdir())
        csv_file = temp_dir / f"temp_{os.getpid()}.csv"
        with open(csv_file, 'w') as f:
            f.write(csv_data)

        # Define the path to Backtester_1.py
        # Use /var/task for Vercel, current directory for local
        if os.environ.get('VERCEL'):
            script_path = Path('/var/task') / 'Backtester_1.py'
            run_cwd = Path('/var/task')
        else:
            script_path = Path(__file__).parent.parent / 'Backtester_1.py'
            run_cwd = Path(__file__).parent.parent
        
        # Ensure Backtests directory exists for output
        backtests_root = run_cwd / 'Backtests'
        backtests_root.mkdir(parents=True, exist_ok=True)

        # Construct the command to run the Python script
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
            raise Exception(f"Python script failed: {result.stderr}")

        # Find the most recent output directory
        output_base_dir = run_cwd / 'Backtests'
        if not output_base_dir.exists():
            raise Exception(f'Backtests directory not found at {output_base_dir}')
        
        latest_dir = max(output_base_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        
        # Read results
        results = {}
        
        # Read metrics
        metrics_file = latest_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                results['metrics'] = json.load(f)
        
        # Read analytics
        analytics_file = latest_dir / 'analytics.md'
        if analytics_file.exists():
            with open(analytics_file) as f:
                results['analytics_md'] = f.read()
        
        # Read CSV files
        def read_csv_as_json(file_path):
            if not file_path.exists():
                return []
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        
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
        
        # Read images as base64
        def read_image_as_base64(file_path):
            if not file_path.exists():
                return None
            with open(file_path, 'rb') as f:
                img_data = f.read()
                return f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
        
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
        
        # Return results
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(results)
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }
        
    finally:
        # Clean up temporary file
        if 'csv_file' in locals() and csv_file.exists():
            csv_file.unlink()
