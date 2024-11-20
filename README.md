# SpiderSensei-Performance-Tracker
SpiderSensei-Bot-Analyzer evaluates the performance of the Spider Sensei bot, providing metrics, hit rates, and insights into crypto trading alerts, with professional PDF reports and analytics.

Description of How the Bot Works
SpiderSensei-Bot-Analyzer is a Python-based tool that evaluates the performance of the Spider Sensei bot by monitoring a Telegram channel for curated crypto alerts and assessing their effectiveness. The bot tracks real-time alerts for Solana tokens (currently migrated from Pump.fun) that include Contract Addresses (CAs). It pulls market data from GeckoTerminal, focusing on the liquidity pool with the highest trading volume for each token. The tool analyzes the price performance of each alert and generates detailed reports with metrics and visualizations.

How the Bot Works:
Tracking Telegram Alerts:

The bot monitors a specific Telegram channel for curated alerts.
Alerts with Contract Addresses (CAs) are processed, while others are ignored.
Fetching Market Data:

Market data is pulled from GeckoTerminal using the highest liquidity pool available for the Solana token mentioned in the alert.
Both 1-minute and 5-minute OHLCV (Open, High, Low, Close, Volume) data are used for analysis.
Analysis and Metrics:

The alert price is taken as the open price of the next 1-minute candle after the alert timestamp, giving traders a realistic entry point.
The maximum price is derived from the close prices of 5-minute candles to exclude artificial spikes caused by MEV bots.
A token’s performance is classified into:
Winners: Price increased by more than 50% from the alert price.
Losses: Price never exceeded the alert price and declined instead.
Charting and Visualization:

Candlestick charts for different timeframes (1m, 5m, 1h) are generated for each token.
A 1-hour chart is included to assess older coins for historical pumps or sustained price trends.
PDF Reporting:

The bot generates professional PDF reports with:
A performance summary on the first page (hit rate, median gains, total alerts).
Detailed insights and charts for each token.
