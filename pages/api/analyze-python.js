// This will call your existing Python backtester
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';

const execAsync = promisify(exec);

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { timeframe, capital, commission, csvData } = req.body;
    
    if (!csvData) {
      return res.status(400).json({ error: 'No CSV data provided' });
    }
    
    // Create temporary files
    const tempDir = '/tmp';
    const csvFile = path.join(tempDir, `temp_${Date.now()}.csv`);
    const outputDir = path.join(tempDir, `output_${Date.now()}`);
    
    // Write CSV data to temporary file
    fs.writeFileSync(csvFile, csvData);
    
    // Create output directory
    fs.mkdirSync(outputDir, { recursive: true });
    
    // Run your Python backtester
    const pythonScript = path.join(process.cwd(), 'Backtester_1.py');
    const command = `cd "${process.cwd()}" && python3 "${pythonScript}" --csv "${csvFile}" --timeframe "${timeframe}" --capital ${capital} --commission ${commission}`;
    
    console.log('Running command:', command);
    
    const { stdout, stderr } = await execAsync(command);
    
    console.log('Python output:', stdout);
    if (stderr) console.log('Python errors:', stderr);
    
    // The Python script creates output in Backtests/ directory
    // We need to find the most recent output directory
    const backtestsDir = path.join(process.cwd(), 'Backtests');
    let latestDir = null;
    let latestTime = 0;
    
    if (fs.existsSync(backtestsDir)) {
      const dirs = fs.readdirSync(backtestsDir);
      for (const dir of dirs) {
        const dirPath = path.join(backtestsDir, dir);
        const stats = fs.statSync(dirPath);
        if (stats.isDirectory() && stats.mtime.getTime() > latestTime) {
          latestTime = stats.mtime.getTime();
          latestDir = dirPath;
        }
      }
    }
    
    if (!latestDir) {
      throw new Error('Could not find output directory');
    }
    
    console.log('Using output directory:', latestDir);
    
    // Read all the results files
    const metricsFile = path.join(latestDir, 'metrics.json');
    const tradesFile = path.join(latestDir, 'trades_enriched.csv');
    const analyticsFile = path.join(latestDir, 'analytics.md');
    const monthlyFile = path.join(latestDir, 'monthly_performance.csv');
    const dowKpisFile = path.join(latestDir, 'dow_kpis.csv');
    const sessionKpisFile = path.join(latestDir, 'session_kpis.csv');
    const holdKpisFile = path.join(latestDir, 'hold_kpis.csv');
    const topBestFile = path.join(latestDir, 'top_best_trades.csv');
    const topWorstFile = path.join(latestDir, 'top_worst_trades.csv');
    const maxWinStreakFile = path.join(latestDir, 'max_win_streak_trades.csv');
    const maxLossStreakFile = path.join(latestDir, 'max_loss_streak_trades.csv');
    
    // Read metrics
    let metrics = {};
    if (fs.existsSync(metricsFile)) {
      metrics = JSON.parse(fs.readFileSync(metricsFile, 'utf8'));
    }
    
    // Read analytics markdown
    let analytics = '';
    if (fs.existsSync(analyticsFile)) {
      analytics = fs.readFileSync(analyticsFile, 'utf8');
    }
    
    // Read CSV files as JSON
    const readCsvAsJson = (filePath) => {
      if (!fs.existsSync(filePath)) return [];
      const csvData = fs.readFileSync(filePath, 'utf8');
      const lines = csvData.split('\n');
      const headers = lines[0].split(',');
      return lines.slice(1).filter(line => line.trim()).map(line => {
        const values = line.split(',');
        const obj = {};
        headers.forEach((header, index) => {
          obj[header.trim()] = values[index] ? values[index].trim() : '';
        });
        return obj;
      });
    };
    
    // Read all CSV data
    const monthlyData = readCsvAsJson(monthlyFile);
    const dowKpis = readCsvAsJson(dowKpisFile);
    const sessionKpis = readCsvAsJson(sessionKpisFile);
    const holdKpis = readCsvAsJson(holdKpisFile);
    const topBest = readCsvAsJson(topBestFile);
    const topWorst = readCsvAsJson(topWorstFile);
    const maxWinStreak = readCsvAsJson(maxWinStreakFile);
    const maxLossStreak = readCsvAsJson(maxLossStreakFile);
    
    // Count trades
    let trades = 0;
    if (fs.existsSync(tradesFile)) {
      const tradesData = fs.readFileSync(tradesFile, 'utf8');
      trades = tradesData.split('\n').length - 1; // Count lines minus header
    }
    
    // Read chart images as base64
    const readImageAsBase64 = (filePath) => {
      if (!fs.existsSync(filePath)) {
        console.log(`Image file not found: ${filePath}`);
        return null;
      }
      const imageBuffer = fs.readFileSync(filePath);
      const base64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
      console.log(`Successfully read image: ${filePath} (${imageBuffer.length} bytes)`);
      return base64;
    };
    
    const charts = {
      equity_curve: readImageAsBase64(path.join(latestDir, 'equity_curve_180d.png')),
      drawdown_curve: readImageAsBase64(path.join(latestDir, 'drawdown_curve.png')),
      pl_histogram: readImageAsBase64(path.join(latestDir, 'pl_histogram.png')),
      heatmap: readImageAsBase64(path.join(latestDir, 'heatmap_dow_hour_count.png'))
    };
    
    console.log('Charts found:', Object.keys(charts).filter(key => charts[key] !== null));
    
    // Clean up temporary files
    try {
      fs.unlinkSync(csvFile);
    } catch (cleanupError) {
      console.log('Cleanup error (non-critical):', cleanupError.message);
    }
    
    const results = {
      metrics,
      trades_count: trades,
      trades_rth_count: metrics.num_trades || 0,
      strategy_name: metrics.strategy_name || "Unknown",
      analytics_md: analytics,
      charts,
      data: {
        monthly_performance: monthlyData,
        dow_kpis: dowKpis,
        session_kpis: sessionKpis,
        hold_kpis: holdKpis,
        top_best_trades: topBest,
        top_worst_trades: topWorst,
        max_win_streak: maxWinStreak,
        max_loss_streak: maxLossStreak
      }
    };

    res.status(200).json(results);

  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ 
      error: error.message,
      details: error.stack 
    });
  }
}
