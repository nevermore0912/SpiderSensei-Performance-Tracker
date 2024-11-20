"""
SpiderSensei-Performance-Tracker (ALPHA version)
------------------------------------------------
IMPORTANT: This script is currently under active development.
The code is experimental, not fully optimized, and may lack proper cleaning or readability.
Please use with caution as features may change, and bugs or inefficiencies may exist.

NOTE: Contributions or suggestions for improvement are welcome!
"""

from telethon import TelegramClient
import csv
import datetime
import re
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import requests
from io import BytesIO
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, String
import time
import mplfinance as mpf
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np

# Telegram API credentials
api_id = '24770422'  # obtained from my.telegram.org
api_hash = 'ca5c7a4029912c4516b0a8a86d6770bd'  # obtained from my.telegram.org

# Create the client and connect
client = TelegramClient('session_name', api_id, api_hash)

# Function to parse relative dates
def parse_relative_date(text):
    now = datetime.datetime.now()
    units = {
        'years': 'years', 'year': 'years', 'year,':'years',
        'months': 'months', 'month': 'months',
        'weeks': 'weeks', 'week': 'weeks',
        'days': 'days', 'day': 'days',
        'hours': 'hours', 'hour': 'hours',
        'minutes': 'minutes', 'minute': 'minutes',
        'seconds': 'seconds', 'second': 'seconds'
    }
    parts = text.split()
    number = 1 if parts[0] in ['a', 'an'] else int(parts[0])
    unit = units[parts[1]]
    past_date = now - relativedelta(**{unit: number})
    difference_in_hours = int((now - past_date).total_seconds() / 3600)
    return difference_in_hours

# Function to extract data from messages
def extract_info(filename):
    entries = []
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    current_data = {
        "timestamp": '', "name": 'n/a', "fdv": 'n/a',
        "followers_now": 'n/a', "followers_discovered": 'n/a',
        "following": 'n/a', "created_in_hours": 'n/a',
        "tweets": 'n/a', "5k_followers": 'n/a', "vc_followers": 'n/a',
        "chad_followers_list": [], "chad_followers_count": 'n/a',
        "time_to_curated": 'n/a', "contract_created": 'n/a',
        "contract": 'n/a', "website": 0,
        "x_link": 'n/a',
        "pic_filename": 'n/a'
    }

    for i, line in enumerate(lines):
        line = line.strip()
        
        # Detect a new entry by finding a Twitter link
        if re.search(r'\((https?://pbs\.twimg\.com/[^)]+)\)', line):
            # Save the current entry if it exists
            if current_data["pic_filename"] !='n/a':
                entries.append(list(current_data.values()))
            current_data.update({k: 'n/a' for k in current_data})  # Reset
            current_data["timestamp"] = line[:25].strip()
            #print(f"Timestamp is {current_data["timestamp"]}")
            pic_link = re.search(r'\((https?://pbs\.twimg\.com/[^)]+)\)', line)
            ## Extract picture from link
            if pic_link:
                extracted_pic_link = pic_link.group(1)
                #print(extracted_pic_link)  # Outputs the picture link
                
                # Define the folder and filename
                folder = 'saved_pics'
                os.makedirs(folder, exist_ok=True)
                file_name = os.path.join(folder, extracted_pic_link.split('/')[-1])
                current_data["pic_filename"] = extracted_pic_link.split('/')[-1]
                #print(current_data["pic_filename"])
                # Download and save the picture
                response = requests.get(extracted_pic_link)
                if response.status_code == 200:
                    with open(file_name, 'wb') as file:
                        file.write(response.content)
                    #print(f"Image saved to {file_name}")
                else:
                    print(f"Failed to download image: {response.status_code}")
        elif '**[**' in line:
            ## Extract twitter link from the line
            x_link = re.search(r'\((https?://(?:x\.com|twitter\.com)/[^\)]+)\)', line)
            if x_link:
                extracted_link = x_link.group(1)
                current_data["x_link"] = extracted_link
                #print(current_data["x_link"])
        elif '**Name**' in line:
            current_data["name"] = line[10:].strip()
        elif '**FDV**' in line:
            current_data["fdv"] = line[8:].strip()
        elif '**Followers**' in line:
            current_data["followers_now"] = line[15:].strip()
        elif '**First discovered followers**' in line:
            current_data["followers_discovered"] = line[32:].strip()
        elif '**Following**' in line:
            current_data["following"] = line[15:].strip()
        elif '**Created**' in line:
            created = line[13:].strip()
            current_data["created_in_hours"] = parse_relative_date(created)
        elif '**Tweets**' in line:
            current_data["tweets"] = line[12:].strip()
        elif '**5K+ Followers**' in line:
            current_data["5k_followers"] = line[19:].strip()
        elif '**VC Followers**' in line:
            current_data["vc_followers"] = line[18:].strip()
        elif '**Chad Followers**' in line:
            matches = re.findall(r'\[([^\]]+)\]\(https?://twitter\.com/[^)]+\)', line)
            current_data["chad_followers_list"] = [match.strip() for match in matches if match.strip()]
        elif '**Time to Curated**' in line:
            time_curated = re.findall(r'\d+', line[21:].strip())
            current_data["time_to_curated"] = int(time_curated[0]) if time_curated else 0
        elif '**Contract Created' in line:
            date_str = line.split(': ')[1].split('**')[0].strip()
            date_obj = datetime.datetime.strptime(date_str, '%d/%m/%Y')
            current_data["contract_created"] = date_obj.strftime('%Y-%m-%d')
        elif '**Contract**' in line:
            current_data["contract"] = line[15:].strip()
        elif '**Website**' in line and 'pump.fun' not in line:
            current_data["website"] = line[13:].strip()
            
        current_data["chad_followers_count"] = len(current_data["chad_followers_list"])
        
    if current_data["timestamp"] and current_data["name"] != 'n/a':
        entries.append(list(current_data.values()))
    
    return entries

# Save extracted data to CSV
def save_to_csv(filename, entries):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Timestamp', 'Name', 'FDV', 'Followers_now', 'Followers_discovered',
            'Following', 'Created_x_hours_ago', 'Tweets', '5k+_Followers',
            'VC_follower', 'Chad_followers', 'Chad_followers_count',
            'Time_to_Curated', 'Contract_Created', 'Contract', 'Website', 'x_link', 'pic_filename'
        ])
        writer.writerows(entries)

# Function to fetch DEX price data
request_count=0 # Use a counter to track requests
def read_dex(token, chain, timeframe, timestamp_marker, limit):
    global request_count
    # Convert timestamp_marker to integer seconds since epoch
    timestamp_marker_epoch = int(datetime.datetime.strptime(timestamp_marker, '%Y-%m-%d %H:%M:%S%z').timestamp())

    # Increment the request counter by 3 since we send 3 request for every token
    # one for the liquidity pool, one for data_before the timestamp_marker 
    # and one for data_after the timestamp_marker
    request_count += 3
    print(f"Request is {request_count}")
    # Check if the request count has reached 30
    if request_count % 21 == 0:
        print("Pausing for 1 minute to comply with request limits...")
        time.sleep(60)  # Pause for 60 seconds
        
    # Determine aggregation interval in seconds
    if timeframe == '/minute?aggregate=1':
        dt = 60  # 1 minute
        time_label = '1m'
    elif timeframe == '/minute?aggregate=5':
        dt = 300  # 5 minutes
        time_label = '5m'
    elif timeframe == '/hour?aggregate=1':
        dt = 3600  # 1 hour
        time_label = '1h'
    else:
        raise ValueError("Unsupported timeframe!")
    yes_token = False
    
    # Get the highest liquidity pool
    url = f"https://api.geckoterminal.com/api/v2/networks/{chain}/tokens/{token}/pools?page=1"
    response = requests.get(url)
    if response.status_code == 200:
        yes_token= True
        pool_data = response.json()
        first_id = pool_data['data'][0]['id']
        pool = first_id.split('_', 1)[1] if '_' in first_id else first_id
        #print(f"Highest liquidity pool for token {token}: {pool}")
    else:
        print(f"Failed to retrieve pool data for token {token}: {response.status_code}, probably pump.fun did not migrate")
        return
        
    # Initialize storage for data
    data_before = []
    data_after = []

    # Fetch data BEFORE the timestamp_marker
    url_before = (f"https://api.geckoterminal.com/api/v2/networks/{chain}/pools/{pool}/ohlcv{timeframe}"
                  f"&limit={limit}&before_timestamp={timestamp_marker_epoch}")
    #print(f"URL BEFORE is: {url_before}")
    response_before = requests.get(url_before)
    if response_before.status_code == 200:
        data_before = response_before.json().get('data', {}).get('attributes', {}).get('ohlcv_list', [])
    else:
        print(f"Error fetching before data: {response_before.status_code}")

    # Fetch data AFTER the timestamp_marker
    # Offset by {limit} intervals ({limit} * dt)
    offset_timestamp = timestamp_marker_epoch + (limit * dt)
    url_after = (f"https://api.geckoterminal.com/api/v2/networks/{chain}/pools/{pool}/ohlcv{timeframe}"
                 f"&limit={limit}&before_timestamp={offset_timestamp}")
    #print(f"URL AFTER is: {url_after}")
    response_after = requests.get(url_after)
    if response_after.status_code == 200:
        data_after = response_after.json().get('data', {}).get('attributes', {}).get('ohlcv_list', [])
    else:
        print(f"Error fetching after data: {response_after.status_code}")

    # Reverse before data to chronological order and combine with after data
    combined_data = data_before[::-1] + data_after

    # Remove any duplicate entries based on timestamp to ensure no overlap
    seen_timestamps = set()
    continuous_data = []
    for entry in combined_data:
        if entry[0] not in seen_timestamps:
            continuous_data.append(entry)
            seen_timestamps.add(entry[0])
    # Sort continuous_data by timestamp
    continuous_data.sort(key=lambda x: x[0])

    # Save combined data to a file
    folder = "saved_data"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f'dex_data_{token}_{time_label}.txt')
    with open(filename, 'w') as file:
        for entry in continuous_data:
            timestamp = datetime.datetime.fromtimestamp(entry[0], tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            open_price, high_price, low_price, close_price, volume = entry[1:6]
            file.write(f"{timestamp} {open_price} {high_price} {low_price} {close_price} {volume}\n")

    #print(f"Combined data saved to {filename}")
    return filename
        
def plot_dex_data(token, timeframe, timestamp_marker):
    if timeframe == '/minute?aggregate=1':
        time_label = '1m'
    elif timeframe == '/minute?aggregate=5':
        time_label = '5m'
    elif timeframe == '/hour?aggregate=1':
        time_label = '1h'

    data = []
    folder = 'saved_data'
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f'dex_data_{token}_{time_label}.txt')
    
    # Read data from ohlcv file
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 7:  # Ensure line contains timestamp, open, high, low, close, volume
                timestamp = parts[0] + " " + parts[1]
                open_price = float(parts[2])
                high_price = float(parts[3])
                low_price = float(parts[4])
                close_price = float(parts[5])
                volume = float(parts[6])
                data.append([timestamp, open_price, high_price, low_price, close_price, volume])

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    # Make the DataFrame index timezone-aware
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')  # Assume data timestamps are in UTC
    df.set_index('timestamp', inplace=True)

    # Sort the DataFrame by timestamp in ascending order
    df = df.sort_index()

    # Prepare the vertical line timestamp
    alines = []
    dont_plot=False
    if timestamp_marker:
        timestamp_marker_dt = pd.to_datetime(timestamp_marker, format='%Y-%m-%d %H:%M:%S%z')  # Parse with timezone
        # Check if the marker is out of range
        if timestamp_marker_dt < df.index.min() or timestamp_marker_dt > df.index.max():
            print(f"Warning: The timestamp_marker {timestamp_marker_dt} is outside the available data range. Skipping marker.")
            dont_plot=True
            alines = []
        else:
            alines = [[(timestamp_marker_dt, df['low'].min()), (timestamp_marker_dt, df['high'].max())]]


    # Create candlestick chart with the vertical line
    folder = 'saved_plots'
    os.makedirs(folder, exist_ok=True)
    filename = f"{token}_candlestick_{time_label}.png"
    file_path = os.path.join(folder, filename)
    
    if not dont_plot:
        # Define a custom style with updated font sizes
        custom_style = mpf.make_mpf_style(
            base_mpf_style='binance',  # Use 'binance' as the base style
            rc={
                'font.size': 18,          # General font size
                'axes.titlesize': 24,     # Font size for the title
                'axes.labelsize': 20,     # Font size for axis labels
                'ytick.labelsize': 16,    # Font size for y-axis tick labels
                'axes.titlepad': 2      # Adjust padding between title and plot (lower value brings it closer)
            }
        )
        # Preprocess the data to handle outliers
        threshold_multiplier = 5
        df['body_size'] = abs(df['open'] - df['close'])
        df['max_reasonable_high'] = df[['open', 'close']].max(axis=1) + (df['body_size'] * threshold_multiplier)
        df['min_reasonable_low'] = df[['open', 'close']].min(axis=1) - (df['body_size'] * threshold_multiplier)
        df['high'] = df[['high', 'max_reasonable_high']].min(axis=1)
        df['low'] = df[['low', 'min_reasonable_low']].max(axis=1)
        
        # Calculate the new y-axis limits
        y_max = df['max_reasonable_high'].max()
        ylim = (0, y_max)

        # Drop temporary columns
        df.drop(['body_size', 'max_reasonable_high', 'min_reasonable_low'], axis=1, inplace=True)

        # Ensure datetime index and timezone awareness
        df.index = pd.to_datetime(df.index)

        # Create and save the plot
        fig, axlist = mpf.plot(
            df,
            type='candle',                # Specify candlestick chart
            style=custom_style,           # Apply the custom style
            alines=dict(alines=alines, colors=['black'], linestyle='--'),
            #title=f"Price Action ({time_label})",  # Set the title
            ylabel="Price (USD)",         # Set y-axis label
            volume=False,                 # Exclude volume subplot
            ylim=ylim,  # Apply the calculated y-axis limits
            xrotation=0,
            returnfig=True  # Return the figure and axes
        )

        axlist[0].set_title(f"Price Action ({time_label})", pad=10, fontweight='bold')  # Lower the title by reducing the pad value
        
        # Save the modified plot
        fig.savefig(file_path)
        plt.close(fig)  # Close the figure to free up memory
        #print(f"Candlestick chart saved to {file_path}")

def df_to_reportlab_table(df, title=None, float_formats=None):
    """
    Convert a Pandas DataFrame to a ReportLab Table with optional title and formatting.

    Parameters:
        df (pd.DataFrame): The DataFrame to convert.
        title (str): Optional title for the table.
        float_formats (dict): A dictionary specifying column-wise float formatting, e.g., {"Alert Price": "{:.7f}", "Max X's": "{:.2f}"}.

    Returns:
        List[Flowable]: A list containing the table and an optional title.
    """
    flowables = []

    if title:
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()
        title_paragraph = Paragraph(title, styles["Heading2"])
        flowables.append(title_paragraph)

    # Format the DataFrame
    if float_formats:
        df = df.copy()
        for column, fmt in float_formats.items():
            if column in df.columns:
                df[column] = df[column].map(lambda x: fmt.format(x) if isinstance(x, (int, float)) else x)

    # Convert DataFrame to a list of lists
    data = [df.columns.tolist()] + df.values.tolist()

    # Create a ReportLab Table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Header row background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header row text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center-align all cells
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Padding for header
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),  # Body rows background
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Add grid lines
    ]))

    flowables.append(table)
    return flowables

def generate_pdf_report(output_pdf, tokens, alert_times):
    """
    Create a PDF report for multiple coins.
    """
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    BodyText = styles["BodyText"]  # Retrieve the "BodyText" style
    flowables = []

    # Title page
    title = Paragraph("Spider Sensei Bot Report", styles['Title'])
    flowables.append(title)
    flowables.append(Spacer(1, 20))
    
    thresholds = {"winner": 50, "loss": 0}  # Winners need a 50% gain, losses are no gain
    
    # Define the images
    gauge_image_path = "saved_plots/hit_rate_gauge.png"  # Replace with your image path
    gauge_image = Image(gauge_image_path, width=150, height=150)  # Adjust width and height

    greed_fear_url = "https://alternative.me/crypto/fear-and-greed-index.png"  # Replace with your image URL
    response = requests.get(greed_fear_url)
    web_image = Image(BytesIO(response.content), width=150, height=150)  # Adjust width and height

    # Create a table to align the images side by side
    gauges_table = Table([[gauge_image, web_image]])
    gauges_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'CENTER'),        # Align content to the center
    ]))
    # Add the table to flowables
    flowables.append(gauges_table)
    flowables.append(Spacer(1, 20))  # Add spacer for separation if needed

    results_summary, token_performance_table = analyze_alerts_performance(tokens, thresholds, alert_times, )
    # Define float formats
    float_formats_performance = {"Alert Price": "{:.7f}", "Max X's": "{:.2f}", "Max Price": "{:.7f}"}
    float_formats_summary = {"Median Gain (X)": "{:.2f}"}
    # Convert and add performance summary table
    flowables.extend(df_to_reportlab_table(results_summary, title="Performance Summary", float_formats=float_formats_summary))

    # Add a spacer after performance summary
    flowables.append(Spacer(1, 20))

    # Convert and add token performance table
    flowables.extend(df_to_reportlab_table(token_performance_table, title="Token Performance Table", float_formats=float_formats_performance))

    # Add a spacer and page break after token performance table
    flowables.append(Spacer(1, 20))    
    flowables.append(PageBreak())
    
    with open('extracted_info.csv', 'r', encoding='utf-8', errors='replace') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            token = row.get('Contract')  # Assumes "Contract" column holds token addresses
            if token != 'n/a':
                name = row.get('Name')
                time_of_ping = row.get('Timestamp')
                curated_at_utc = pd.to_datetime(time_of_ping).strftime('%Y-%m-%d %H:%M:%S UTC')
                curated_in_dt = row.get('Time_to_Curated')
                pic_filename = row.get('pic_filename')
                fdv = row.get('FDV')
                x_link = row.get('x_link')

                # Process image and details
                folder = 'saved_pics'
                file_name = os.path.join(folder, pic_filename)
                img = Image(file_name, width=100, height=100)

                details = [
                    [Paragraph(f"<b>X link:</b> <a href='{x_link}' color='blue'>{x_link}</a>", BodyText)],
                    [Paragraph(f"<b>Contract:</b> {token}", BodyText)],
                    [Paragraph(f"<b>FDV at alert:</b> {fdv}", BodyText)],
                    [Paragraph(f"<b>Curated at:</b> {curated_at_utc}", BodyText)],
                    [Paragraph(f"<b>Time to curated:</b> {curated_in_dt} minutes", BodyText)]
                ]

                # Process plots
                def generate_plot(filename):
                    if os.path.exists(filename):
                        return Image(filename, width=150, height=150)
                    else:
                        placeholder = Drawing(150, 150)
                        placeholder.add(String(50, 75, "n/a token too recent", textAnchor="middle", fontSize=10))
                        return placeholder

                filename_plot1 = os.path.join('saved_plots', f'{token}_candlestick_1m.png')
                filename_plot2 = os.path.join('saved_plots', f'{token}_candlestick_5m.png')
                filename_plot3 = os.path.join('saved_plots', f'{token}_candlestick_1h.png')

                plot1 = generate_plot(filename_plot1)
                plot2 = generate_plot(filename_plot2)
                plot3 = generate_plot(filename_plot3)

                # Combine image and details
                name_row = [Paragraph(f"<b><font size=14>Name = {name}</font></b>", BodyText), ""]  # Adjust font size and bold
                image_details_table = Table(
                    [name_row, [img, details]],  # Add the name row above the existing content
                    colWidths=[110, 360]  # Set column widths
                )
                image_details_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),        # Align content to the top
                    ('LEFTPADDING', (1, 0), (1, 0), 5),         # Adjust left padding for details
                ]))

                # Combine plots
                plots_table = Table(
                    [[plot1, plot2, plot3]],
                    colWidths=[150, 150, 150]  # Equal widths for plots
                )
                plots_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),     # Center-align plots
                ]))

                # Outer table to combine image/details and plots
                combined_table = Table(
                    [[image_details_table], [plots_table]],
                    colWidths=[480],  # Full width for outer table
                )
                combined_table.setStyle(TableStyle([
                    ('BOX', (0, 0), (-1, -1), 1, colors.grey), # Add outer border
                    ('VALIGN', (0, 0), (-1, -1), 'CENTER'),       # Align content to the top
                ]))

                # Add the combined table to the flowables
                flowables.append(combined_table)
                flowables.append(Spacer(1, 20))  # Add spacer for separation between sections

    # Build the PDF
    doc.build(flowables)
    print(f"Report saved to {output_pdf}")

def analyze_alerts_performance(tokens, thresholds, alert_times):
    """
    Analyze the performance of alerts by determining winners and losses.
    Includes alert price from 1m data and max price from 5m data.

    Parameters:
        tokens (list): List of token addresses.
        thresholds (dict): Thresholds for categorization (e.g., {"winner": 50, "loss": 0}).
        alert_times (dict): Dictionary of alert times for each token (e.g., {"TOKEN1": "2024-11-20 14:00:00"}).
        chain (str): Blockchain network (default is 'solana').

    Returns:
        results_summary (DataFrame): A table summarizing total alerts, hit rate, and median gain.
        token_performance_table (DataFrame): A table detailing token performance metrics.
    """
    results = {"winners": 0, "losses": 0}
    max_xs = []  # To store max x's for median calculation
    token_data = []  # To store token-specific data for table

    for token in tokens:
        alert_time = alert_times.get(token)
        if not alert_time:
            continue

        # Read 1m data for alert price
        file_path_1m = os.path.join('saved_data', f'dex_data_{token}_1m.txt')
        if not os.path.exists(file_path_1m):
            continue

        # Load 1m data
        data_1m = []
        with open(file_path_1m, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 7:
                    timestamp, open_price = parts[0] + " " + parts[1], float(parts[2])
                    data_1m.append([timestamp, open_price])

        df_1m = pd.DataFrame(data_1m, columns=["timestamp", "open"])
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp']).dt.tz_localize('UTC')
        df_1m.set_index('timestamp', inplace=True)

        # Convert `alert_time` to timezone-aware datetime in UTC
        alert_time_dt = pd.to_datetime(alert_time).tz_convert('UTC')

        # Get the open price of the next 1m candle closest to the alert time
        alert_price = df_1m[df_1m.index >= alert_time_dt]['open'].iloc[0]

        # Read 5m data for max price
        file_path_5m = os.path.join('saved_data', f'dex_data_{token}_5m.txt')
        if not os.path.exists(file_path_5m):
            continue

        # Load 5m data
        data_5m = []
        with open(file_path_5m, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 7:
                    timestamp, close_price = parts[0] + " " + parts[1], float(parts[5])
                    data_5m.append([timestamp, close_price])

        df_5m = pd.DataFrame(data_5m, columns=["timestamp", "close"])
        df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
        df_5m.set_index('timestamp', inplace=True)

        # Calculate max close price from 5m data
        max_price = df_5m['close'].max()
        max_x = max_price / alert_price

        # Classify based on thresholds
        max_gain = (max_price - alert_price) / alert_price * 100
        if max_gain >= thresholds["winner"]:
            results["winners"] += 1
        else:
            results["losses"] += 1

        # Store token-specific data for the table
        max_xs.append(max_x)
        token_data.append({
            "Token": token,
            "Alert Price": alert_price,
            "Max Price": max_price,
            "Max X's": max_x
        })

    # Calculate hit rate
    total_alerts = sum(results.values())
    hit_rate = (results["winners"] / total_alerts) * 100 if total_alerts > 0 else 0

    # Median of max X's
    median_gain = np.median(max_xs) if max_xs else 0

    # Save hit rate gauge (optional visualization)
    save_hit_rate_gauge(hit_rate, "saved_plots/hit_rate_gauge.png")

    # Create DataFrames for results summary and token performance
    results_summary = pd.DataFrame([{
        "Total Alerts": total_alerts,
        "Hit Rate (%)": hit_rate,
        "Median Gain (X)": median_gain
    }])

    token_performance_table = pd.DataFrame(token_data)

    return results_summary, token_performance_table
    
def save_hit_rate_gauge(hit_rate, file_path):
    """
    Save an accelerometer-style gauge for hit rate to a file.

    Parameters:
        hit_rate (float): The hit rate percentage (0 to 100).
        file_path (str): The file path where the gauge will be saved.
    """
    # Define the gauge parameters
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    theta = np.linspace(0, np.pi, 100)  # Semi-circle angles
    radii = [1] * len(theta)

    # Create a custom colormap (blue to orange to red)
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("red_orange_blue", ["red", "orange", "blue"])

    # Create the gauge background with the custom colormap
    for i, angle in enumerate(theta):
        ax.fill_between([angle, angle + 0.02], 0, radii[i], color=custom_cmap(i / len(theta)))

    # Add the needle (hit rate indicator)
    needle_angle = np.deg2rad(180 - hit_rate / 100 * 180)  # Correct orientation (0% = left, 100% = right)
    ax.plot([needle_angle, needle_angle], [0, 1.2], color="black", linewidth=3)

    # Add text for the gauge (Hot and Cold)
    ax.text(np.pi/2, 1.5, f"{hit_rate:.1f}%", ha='center', fontsize=16, fontweight='bold')
    ax.text(np.pi, 1.3, "Cold", ha='center', va='bottom', fontsize=14, fontweight='bold', color="blue")
    ax.text(0, 1.3, "Hot", ha='center', va='bottom', fontsize=14, fontweight='bold', color="red")

    # Add title inside the image
    ax.text(np.pi/2, 2.0, "Bot Success Rate", ha='center', fontsize=18, fontweight='bold')

    # Hide polar frame and ticks
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.set_axis_off()

    # Save the figure to the specified file path
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)

# Main function to run Telegram scraping and extraction
async def main():
    group_id = -1002131165158  # Replace with your Telegram group ID
    messages = []
    async for message in client.iter_messages(group_id, limit=40):
        messages.append(f"{message.date}: {message.sender_id}: {message.text}")
    
    with open('messages.txt', 'w', encoding='utf-8', errors='replace') as file:
        file.write("\n".join(messages))
    
    entries = extract_info('messages.txt')
    save_to_csv('extracted_info.csv', entries)
    print("Data saved to extracted_info.csv.")
    tokens = []
    alert_times= {}
    with open('extracted_info.csv', 'r', encoding='utf-8', errors='replace') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        j=0
        for row in csv_reader:
            j+=1
            token = row.get('Contract')  # Assumes "Contract" column holds token addresses
            time_of_ping = row.get('Timestamp')
            chain = 'solana'  # Adjust if 'chain' info is in a separate column or hardcoded
            contract_created = row.get('Contract_Created')
            if token!='n/a':  # Ensure token value exists
                tokens.append(token)
                alert_times[token]=time_of_ping
                print(f"ALERT {j} Processing DEX data for token: {token}")
                timeframe_1m='/minute?aggregate=1'
                timeframe_5m='/minute?aggregate=5'
                timeframe_1h='/hour?aggregate=1'
                yes_token = read_dex(token, chain, timeframe_1m, time_of_ping, 50)
                if yes_token: 
                    #print(time_of_ping)
                    plot_dex_data(token,timeframe_1m,time_of_ping)
                
                yes_token = read_dex(token, chain, timeframe_5m, time_of_ping, 50)
                if yes_token: 
                    #print(time_of_ping)
                    plot_dex_data(token,timeframe_5m,time_of_ping)
                
                yes_token = read_dex(token, chain, timeframe_1h, time_of_ping, 50)
                if yes_token: 
                    #print(time_of_ping)
                    plot_dex_data(token,timeframe_1h,time_of_ping)
            else:
                print(f"ALERT {j} No valid token this alert. Skipping.")
    
    
    print("Completed processing all tokens in extracted_info.csv.")
    generate_pdf_report("spidersensei_performance_report.pdf", tokens, alert_times)

with client:
    client.loop.run_until_complete(main())
