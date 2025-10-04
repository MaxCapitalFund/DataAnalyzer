def compute_metrics(trades_rth: pd.DataFrame, cfg: BacktestConfig, scope_label: str = "RTH") -> dict:
    """Compute performance metrics for RTH trades (clean, error-free)."""
    df = trades_rth.copy()

    if 'AdjustedNetPL' not in df.columns:
        raise RuntimeError("AdjustedNetPL missing; call apply_stoploss_corrections() before compute_metrics().")

    # --- Basic P/L series
    pl_net = df['AdjustedNetPL'].fillna(0.0)
    pl_gross = df['GrossPL'].fillna(0.0) if 'GrossPL' in df.columns else pl_net.copy()
    equity = cfg.initial_capital + pl_net.cumsum()

    # --- Totals & returns
    total_net = float(pl_net.sum())
    total_gross = float(pl_gross.sum())
    total_return_pct = (total_net / cfg.initial_capital) * 100.0 if cfg.initial_capital else np.nan

    # --- Win/loss stats
    win_mask = pl_net > 0
    loss_mask = pl_net < 0
    avg_win = float(pl_net[win_mask].mean()) if win_mask.any() else np.nan
    avg_loss = float(pl_net[loss_mask].mean()) if loss_mask.any() else np.nan

    # --- Drawdowns
    max_dd_pct = abs(_max_drawdown(equity)) * 100.0
    max_dd_dollars = float((equity.cummax() - equity).max())
    recovery_factor = float(total_net / max_dd_dollars) if max_dd_dollars else np.nan

    # --- Expectancy & risk-adjusted stats
    expectancy_dollars = float(pl_net.mean()) if len(pl_net) else np.nan
    trade_rets = pl_net / cfg.initial_capital if cfg.initial_capital else pd.Series(np.nan, index=pl_net.index)
    per_trade_sharpe = float(trade_rets.mean() / trade_rets.std(ddof=1)) if trade_rets.std(ddof=1) > 0 else np.nan

    first_dt = pd.to_datetime(df['ExitTime']).min()
    last_dt  = pd.to_datetime(df['ExitTime']).max()
    days = max((last_dt - first_dt).days, 1) if pd.notna(first_dt) and pd.notna(last_dt) else 1
    trades_per_year = (len(df) / days * 252.0) if days and days > 0 else np.nan
    sharpe_annualized = float(np.sqrt(trades_per_year) * per_trade_sharpe) if trades_per_year and per_trade_sharpe == per_trade_sharpe else np.nan

    # --- Extremes
    largest_win = float(pl_net.max()) if len(pl_net) else np.nan
    largest_loss = float(pl_net.min()) if len(pl_net) else np.nan

    # --- Compile metrics dict (fully closed and valid)
    metrics = {
        "scope": scope_label,
        "strategy_name": cfg.strategy_name,
        "version": cfg.version,
        "timeframe": cfg.timeframe,
        "initial_capital": cfg.initial_capital,
        "point_value": cfg.point_value,
        "num_trades": int(len(df)),
        "net_profit": total_net,
        "gross_profit": total_gross,
        "total_return_pct": total_return_pct,
        "profit_factor": _profit_factor(pl_net),
        "win_rate_pct": float((pl_net > 0).mean() * 100.0),
        "avg_win_dollars": avg_win,
        "avg_loss_dollars": avg_loss,
        "expectancy_per_trade_dollars": expectancy_dollars,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_dollars": max_dd_dollars,
        "recovery_factor": recovery_factor,
        "sharpe_annualized": sharpe_annualized,
        "largest_winning_trade": largest_win,
        "largest_losing_trade": largest_loss
    }

    return metrics
