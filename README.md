# SpiderSensei-Performance-Tracker

***IMPORTANT NOTES:

   - ALPHA Version: This is an early release with limited functionality and experimental features. Use with caution!

   - For now the code needs TWEEKING if you want to run it yourself (contact me)

   - Channel Support: Currently, the tracker only monitors the CURATED-QUICKLY Telegram channel.

   - Token Compatibility: At this stage, the tracker exclusively supports Solana (SOL) tokens.

   - A coin is a **Winner** if price increases by more than 50%

   - A coin is a **Loss** if price drops after the alert even if it first rose by less than 50%

   - For the performance table, the Max X's column gives us the MAX returns we could have had if we exited at the top.

   - Similarly if the coin dropped in price the Max X's is less than 1.

   - For now the median gain is just the median of ALL the Max X's but we are still trying to quantify this better as we sometimes ride coins to 0 other times we exit for a loss...

SpiderSensei-Bot-Analyzer evaluates the performance of the Spider Sensei bot, providing metrics, hit rates, and insights into crypto trading alerts, with professional PDF reports and analytics.

Description of How the Script Works

SpiderSensei-Bot-Analyzer is a Python-based tool that evaluates the performance of the Spider Sensei bot by monitoring the CURATED-QUICKLY (for now) Telegram channel for alerts and assessing their effectiveness. The bot tracks real-time alerts for Solana tokens (currently migrated from Pump.fun) that include Contract Addresses (CAs). It pulls market data from GeckoTerminal, focusing on the liquidity pool with the highest trading volume for each token. The tool analyzes the price performance of each alert and generates detailed reports with metrics and visualizations.

## How the Script Works

1. **Tracking Telegram Alerts**:  
   The bot monitors a specific Telegram channel for curated alerts. Alerts with Contract Addresses (CAs) are processed, while others are ignored.

2. **Fetching Market Data**:  
   Market data is pulled from GeckoTerminal using the highest liquidity pool available for the Solana token mentioned in the alert. Both 1-minute and 5-minute OHLCV (Open, High, Low, Close, Volume) data are used for analysis.

3. **Analysis and Metrics**:  
   - The **alert price** is taken as the **open price of the next 1-minute candle** after the alert timestamp, giving traders a realistic entry point.
   - The **maximum price** is derived from the **close prices** of 5-minute candles to exclude artificial spikes caused by MEV bots.
   - A token’s performance is classified into:
     - **Winners**: Price increased by more than 50% from the alert price.
     - **Losses**: Price never exceeded the alert price and declined instead.

4. **Charting and Visualization**:  
   Candlestick charts for different timeframes (1m, 5m, 1h) are generated for each token. A 1-hour chart is included to assess older coins for historical pumps or sustained price trends.

5. **PDF Reporting**:  
   The script generates PDF reports with:
   - A performance summary on the first page (hit rate, median gains, total alerts).
   - Detailed insights and charts for each token.
   - Naming convention: spidersensei_report_YYYYMMDD_HHMMSS.pdf
