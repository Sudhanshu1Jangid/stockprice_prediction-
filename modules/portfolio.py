import pandas as pd
from modules.stock_data import get_current_price, fetch_stock_data

class Portfolio:
    def __init__(self):
        self.positions = {} 
        
    def add_position(self, ticker, shares):
        """Add or update a position in the portfolio"""
        if ticker in self.positions:
            self.positions[ticker] += shares
        else:
            self.positions[ticker] = shares
            
    def get_portfolio_status(self):
        """Get current portfolio status"""
        if not self.positions:
            return pd.DataFrame()
            
        data = []
        for ticker, shares in self.positions.items():
            try:
                current_price = get_current_price(ticker)
                position_value = current_price * shares

                data.append({
                    'Ticker': ticker,
                    'Shares': shares,
                    'Current Price': current_price,
                    'Position Value': position_value
                })
            except Exception:
                continue
                
        return pd.DataFrame(data)
        
    def get_portfolio_history(self):
        """Get portfolio value history"""
        if not self.positions:
            return pd.DataFrame()
            
        portfolio_value = pd.DataFrame()
        
        for ticker, shares in self.positions.items():
            try:
                df = fetch_stock_data(ticker)
                portfolio_value[ticker] = df['Close'] * shares
            except Exception:
                continue
                
        portfolio_value['Total Value'] = portfolio_value.sum(axis=1)
        return portfolio_value
