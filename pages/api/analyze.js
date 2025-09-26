// Working version with real CSV analysis
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
    
    console.log('Request received:', {
      timeframe,
      capital,
      commission,
      csvDataLength: csvData ? csvData.length : 0
    });
    
    if (!csvData) {
      return res.status(400).json({ error: 'No CSV data provided' });
    }
    
    // Parse CSV data manually (simpler approach)
    const lines = csvData.split('\n');
    let dataStartIndex = -1;
    
    // Find the actual data table
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].trim().startsWith('Id;Strategy;')) {
        dataStartIndex = i;
        break;
      }
    }
    
    if (dataStartIndex === -1) {
      return res.status(400).json({ 
        error: 'Could not find data table in CSV file',
        preview: lines.slice(0, 10).join('\n')
      });
    }
    
    // Extract headers and data
    const headerLine = lines[dataStartIndex];
    const headers = headerLine.split(';').map(h => h.trim());
    const dataLines = lines.slice(dataStartIndex + 1);
    
    console.log('Found data starting at line:', dataStartIndex + 1);
    console.log('Headers:', headers);
    console.log('Data lines:', dataLines.length);
    
    // Parse data rows
    const trades = [];
    let tradeId = 1;
    
    for (const line of dataLines) {
      if (!line.trim()) continue;
      
      const values = line.split(';');
      if (values.length < headers.length) continue;
      
      const row = {};
      headers.forEach((header, index) => {
        row[header] = values[index] ? values[index].trim() : '';
      });
      
      // Process this row as a trade
      const trade = processTradeRow(row, commission);
      if (trade) {
        trade.Id = tradeId++;
        trades.push(trade);
      }
    }
    
    console.log('Processed trades:', trades.length);
    
    // Apply stop-loss corrections
    const correctedTrades = trades.map(trade => {
      const slBreached = trade.NetPL < -100.0;
      const slCorrection = slBreached ? (-trade.NetPL) - 100.0 : 0.0;
      const adjustedNetPL = slBreached ? -100.0 : trade.NetPL;
      
      return {
        ...trade,
        SLBreached: slBreached,
        SLCorrection: slCorrection,
        AdjustedNetPL: adjustedNetPL
      };
    });
    
    // Filter for RTH trades
    const rthTrades = correctedTrades.filter(trade => {
      const entryTime = parseDateTime(trade.EntryTime);
      const exitTime = parseDateTime(trade.ExitTime);
      return inRTH(entryTime) || inRTH(exitTime);
    });
    
    console.log('RTH trades:', rthTrades.length);
    
    // Compute metrics
    const metrics = computeMetrics(rthTrades, { initial_capital: capital });
    
    const results = {
      metrics,
      trades_count: trades.length,
      trades_rth_count: rthTrades.length,
      strategy_name: metrics.strategy_name,
      charts: {
        equity_curve: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        drawdown_curve: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        pl_histogram: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
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

// Helper functions
function parseDateTime(dateStr) {
  if (!dateStr) return null;
  const date = new Date(dateStr);
  return isNaN(date.getTime()) ? null : date;
}

function toFloat(value) {
  if (typeof value === 'string') {
    value = value.replace(/[\$,]/g, '').replace(/\(([^)]*)\)/, '-$1');
  }
  const num = parseFloat(value);
  return isNaN(num) ? 0 : num;
}

function inRTH(dateTime) {
  if (!dateTime) return false;
  
  const time = new Date(dateTime);
  const hour = time.getHours();
  const minute = time.getMinutes();
  const timeInMinutes = hour * 60 + minute;
  
  const OPEN_START = 9 * 60 + 30; // 9:30 AM
  const CLOSE_END = 16 * 60;      // 4:00 PM
  
  return timeInMinutes >= OPEN_START && timeInMinutes <= CLOSE_END;
}

function processTradeRow(row, commission) {
  // Extract trade information
  const tradePL = toFloat(row['Trade P/L'] || row.TradePL || 0);
  const quantity = toFloat(row.Quantity || row.Qty || 1);
  const qtyAbs = Math.abs(quantity);
  
  // Calculate commission and net P/L
  const tradeCommission = commission * qtyAbs;
  const netPL = tradePL - tradeCommission;
  
  // Determine direction
  let direction = 'Unknown';
  const side = String(row.Side || row.Action || '').toUpperCase();
  if (/(BTO|BUY TO OPEN|BUY_TO_OPEN|BOT TO OPEN)/.test(side)) {
    direction = 'Long';
  } else if (/(STO|SELL TO OPEN|SELL_TO_OPEN|SELL SHORT)/.test(side)) {
    direction = 'Short';
  } else if (!isNaN(quantity)) {
    direction = quantity > 0 ? 'Long' : 'Short';
  }
  
  // Parse dates
  const entryTime = parseDateTime(row['Date/Time'] || row.Date);
  const exitTime = entryTime; // For now, use same time
  
  // Calculate holding time
  const holdMins = 0; // Simplified for now
  
  return {
    EntryTime: entryTime,
    ExitTime: exitTime,
    EntryPrice: toFloat(row.Price || 0),
    ExitPrice: toFloat(row.Price || 0),
    EntryQty: quantity,
    QtyAbs: qtyAbs,
    TradePL: tradePL,
    GrossPL: tradePL,
    Commission: tradeCommission,
    NetPL: netPL,
    BaseStrategy: (row.Strategy || '').split('(')[0].trim() || 'Unknown',
    Symbol: row.Symbol || '',
    EntrySide: row.Side || row.Action || '',
    ExitSide: row.Side || row.Action || '',
    Direction: direction,
    HoldMins: holdMins
  };
}

function computeMetrics(trades, config) {
  if (trades.length === 0) {
    return {
      strategy_name: "Unknown",
      num_trades: 0,
      net_profit: 0,
      total_return_pct: 0,
      win_rate_pct: 0,
      profit_factor: 0,
      max_drawdown_dollars: 0,
      sharpe_annualized: 0,
      avg_win_dollars: 0,
      avg_loss_dollars: 0,
      largest_winning_trade: 0,
      largest_losing_trade: 0,
      expectancy_per_trade_dollars: 0,
      recovery_factor: 0
    };
  }
  
  const plNet = trades.map(t => t.AdjustedNetPL || t.NetPL);
  const totalNet = plNet.reduce((sum, pl) => sum + pl, 0);
  const totalReturnPct = (totalNet / config.initial_capital) * 100.0;
  
  const wins = plNet.filter(pl => pl > 0);
  const losses = plNet.filter(pl => pl < 0);
  const winRate = (wins.length / plNet.length) * 100.0;
  
  const grossProfit = wins.reduce((sum, pl) => sum + pl, 0);
  const grossLoss = Math.abs(losses.reduce((sum, pl) => sum + pl, 0));
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);
  
  // Calculate equity curve and drawdown
  let equity = config.initial_capital;
  const equityCurve = [equity];
  let peak = equity;
  let maxDrawdown = 0;
  
  plNet.forEach(pl => {
    equity += pl;
    equityCurve.push(equity);
    if (equity > peak) peak = equity;
    const drawdown = (peak - equity) / peak;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  });
  
  const maxDrawdownDollars = maxDrawdown * peak;
  
  // Simple Sharpe calculation
  const avgReturn = plNet.reduce((sum, pl) => sum + pl, 0) / plNet.length;
  const variance = plNet.reduce((sum, pl) => sum + Math.pow(pl - avgReturn, 2), 0) / plNet.length;
  const stdDev = Math.sqrt(variance);
  const sharpe = stdDev > 0 ? avgReturn / stdDev : 0;
  
  const avgWin = wins.length > 0 ? wins.reduce((sum, pl) => sum + pl, 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((sum, pl) => sum + pl, 0) / losses.length : 0;
  const largestWin = wins.length > 0 ? Math.max(...wins) : 0;
  const largestLoss = losses.length > 0 ? Math.min(...losses) : 0;
  const expectancy = avgReturn;
  const recoveryFactor = maxDrawdownDollars > 0 ? totalNet / maxDrawdownDollars : 0;
  
  return {
    strategy_name: trades[0]?.BaseStrategy || "Unknown",
    num_trades: trades.length,
    net_profit: totalNet,
    total_return_pct: totalReturnPct,
    win_rate_pct: winRate,
    profit_factor: profitFactor,
    max_drawdown_dollars: maxDrawdownDollars,
    max_drawdown_pct: maxDrawdown * 100,
    sharpe_annualized: sharpe,
    avg_win_dollars: avgWin,
    avg_loss_dollars: avgLoss,
    largest_winning_trade: largestWin,
    largest_losing_trade: largestLoss,
    expectancy_per_trade_dollars: expectancy,
    recovery_factor: recoveryFactor
  };
}