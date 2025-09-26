export default function handler(req, res) {
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
    
    // Simple test response
    const results = {
      metrics: {
        strategy_name: "Test Strategy",
        net_profit: 1500.50,
        total_return_pct: 60.02,
        win_rate_pct: 55.5,
        profit_factor: 1.8,
        max_drawdown_dollars: 250.75,
        sharpe_annualized: 1.2,
        avg_win_dollars: 125.30,
        avg_loss_dollars: -85.20,
        largest_winning_trade: 450.00,
        largest_losing_trade: -150.00,
        expectancy_per_trade_dollars: 15.50,
        recovery_factor: 6.0
      },
      trades_count: 100,
      trades_rth_count: 85,
      strategy_name: "Test Strategy",
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
