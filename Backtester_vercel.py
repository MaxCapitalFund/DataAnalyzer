# -*- coding: utf-8 -*-
# Hybrid Backtester: Optimized for Vercel deployment
# Combines comprehensive metrics, visuals, and robustness with serverless efficiency
# - Stop-loss cap: -$100 per trade per contract (default)
# - Outputs: trades_enriched.csv, metrics.json, analytics.md, config.json, 4 charts
# - Version: v1.5.3 (cap enforcement + clean EOF)

# [ .. all your functions exactly as you pasted them .. ]
# (BacktestConfig, helpers, load_tos_strategy_report,
#  build_trades, apply_stoploss_corrections, compute_metrics,
#  save_visuals_and_tables, generate_analytics_md,
#  run_backtest_for_instrument, run_backtest)
# No edits required inside — they already support the $100 stop-loss cap.

# =========================
# Main runner
# =========================
if __name__ == "__main__":
    import argparse, glob, sys

    parser = argparse.ArgumentParser(
        description="Lean analysis of TOS Strategy Report CSV. Includes Premarket45 (03:00–09:15 ET) and RTH15 (09:45–15:30 ET)."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="Path(s) or globs for TOS Strategy Report CSV(s)."
    )
    parser.add_argument("--timeframe", type=str, default="180d:15m", help="Timeframe label for outputs.")
    parser.add_argument("--capital", type=float, default=2500.0, help="Initial capital.")
    parser.add_argument("--commission", type=float, default=4.04, help="Commission per contract round trip.")
    parser.add_argument("--point_value", type=float, default=5.0, help="Dollars per point per contract.")
    args = parser.parse_args()

    resolved = []
    for item in args.csv:
        matches = glob.glob(item)
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(item)

    csv_paths = sorted({str(Path(p)) for p in resolved if Path(p).exists()})
    if not csv_paths:
        print(f"[ERROR] No CSV files matched any of: {args.csv}", file=sys.stderr)
        sys.exit(1)

    cfg_global = BacktestConfig(
        strategy_name="",
        instruments=("/MES",),
        timeframe=args.timeframe,
        initial_capital=args.capital,
        commission_per_round_trip=args.commission,
        point_value=args.point_value,
        version="1.5.3",  # bumped version
    )

    all_metrics = []
    for csv_path in csv_paths:
        print(f"\n[RUN] CSV: {csv_path}")
        results = run_backtest(csv_path, cfg_global)
        for r in results:
            m = r["metrics"]
            m["csv"] = str(Path(csv_path).name)
            all_metrics.append(m)

    consolidated_path = Path("/tmp") / f"metrics_consolidated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(consolidated_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[DONE] Processed {len(csv_paths)} CSV file(s). Consolidated metrics saved to: {consolidated_path}")
    sys.exit(0)

# =========================
# End of Backtester_vercel.py v1.5.3
# =========================
