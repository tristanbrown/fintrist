#!/usr/bin/env python3
"""
Main routine of the package. 
"""
import argparse
import sys
import os
import time

from fintrist import settings
from fintrist.examples.example_av import Example
from fintrist.scrapers.equity import Equity
from fintrist.history import History
from fintrist.sims.sim import Ticker

parser = argparse.ArgumentParser(description='Start Fintrist.')
parser.add_argument('symbol', metavar='SYM', type=str, help='Stock symbol')
parser.add_argument('-o', '--opt', type=str, help='Optional arguments', required=False)
args = parser.parse_args()

def main():
    
    # Determine inputs.
    
    pass

if __name__ == "__main__":
    main()
