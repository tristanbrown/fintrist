#!/usr/bin/env python3
"""
Script to track a stock's daily chart.
"""
import argparse

from fintrist import settings
from fintrist.tracker import TrackingEngine

parser = argparse.ArgumentParser(description='Start Fintrist.')
parser.add_argument('symbol', metavar='SYM', type=str, help='Stock symbol')
parser.add_argument('-o', '--opt', type=str, help='Optional arguments', required=False)
args = parser.parse_args()

def main():
    # Make one of these specific for each scraper. 
    # Determine inputs.
    name = "{}_daily".format(args.symbol)
    inputs = {'symbol': args.symbol}
    stock = TrackingEngine(name, 'stock', 'exists', inputs)

if __name__ == "__main__":
    main()
