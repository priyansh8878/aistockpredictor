# predictor/views.py
from django.shortcuts import render
import yfinance as yf
import pandas as pd
import openai
import os
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def index(request):
    # Get ticker symbol from request (default AAPL)
    ticker = request.GET.get('ticker', 'AAPL')

    # Download historical stock data
    data = yf.download(ticker, period='6mo', interval='1d')

    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Ensure 'Close' is numeric and drop missing values
    if 'Close' in data.columns:
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    else:
        data['Close'] = pd.to_numeric(data.iloc[:, 0], errors='coerce')

    data = data.dropna(subset=['Close'])
    data['Date'] = pd.to_datetime(data['Date'])

    # Prepare last 5 rows for OpenAI prompt
    recent_data = data.tail(5)[['Date', 'Close']].copy()
    recent_data['Date'] = recent_data['Date'].dt.strftime('%Y-%m-%d')

    # Format recent data for prompt
    stock_summary = "\n".join([
        f"{row['Date']}: ${round(row['Close'], 2)}"
        for _, row in recent_data.iterrows()
    ])

    # Prompt GPT to predict next 5 days
    prompt = f"""
You are a financial analyst AI. Here are the last 5 closing prices of {ticker}:
{stock_summary}

Predict the next 5 days of closing prices and describe the stock's near-term trend and sentiment.

Return ONLY valid JSON (no text before or after) in this exact format:
{{
    "predicted_prices": [list of 5 float prices],
    "analysis": "your explanation here"
}}
"""

    # Call OpenAI API and safely parse JSON
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON from AI response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            result = json.loads(json_text)
        else:
            raise ValueError("No JSON found in AI response.")

        predicted_prices = result.get("predicted_prices", [])
        analysis = result.get("analysis", "No analysis available.")

    except Exception as e:
        predicted_prices = []
        analysis = f"Error during AI prediction: {e}\nAI raw response: {content if 'content' in locals() else 'None'}"

    # Prepare data for charting
    actual_dates = data['Date'].dt.strftime('%Y-%m-%d').tolist()
    actual_prices = data['Close'].astype(float).tolist()

    context = {
        'ticker': ticker,
        'actual_dates': actual_dates,
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'analysis': analysis,
    }

    return render(request, 'predictor/index.html', context)
