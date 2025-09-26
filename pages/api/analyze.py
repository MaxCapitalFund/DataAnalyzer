import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import base64

def handler(req, res):
    if req.method != 'POST':
        res.status(405).json({'error': 'Method not allowed'})
        return

    try:
        # Parse request body
        body = json.loads(req.body)
        timeframe = body.get('timeframe', '180d:15m')
        capital = body.get('capital', 2500.0)
        commission = body.get('commission', 4.04)
        csv_data = body.get('csvData', '')

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            csv_file = f.name

        try:
            # Run the Python backtester
            script_path = Path('/var/task') / 'Backtester_1.py'
            cmd = [
                sys.executable,
                str(script_path),
                '--csv', csv_file,
                '--timeframe', timeframe,
                '--capital', str(capital),
                '--commission', str(commission)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='/var/task')
            
            if result.returncode != 0:
                raise Exception(f"Python script failed: {result.stderr}")

            # Find the most recent output directory
            backtests_dir = Path('/var/task') / 'Backtests'
            if not backtests_dir.exists():
                raise Exception('Backtests directory not found')
            
            latest_dir = max(backtests_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            
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
            
            # Return results directly for Next.js
            res.status(200).json(results)
            
        finally:
            # Clean up temporary file
            if os.path.exists(csv_file):
                os.unlink(csv_file)
                
    except Exception as e:
        # Return error directly for Next.js
        res.status(500).json({
            'error': str(e),
            'type': type(e).__name__
        })