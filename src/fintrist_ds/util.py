"""Utility functions"""
import re

def split_alphanum(astr):
    match = re.match(r"([0-9]+)([a-z]+)", astr, re.I)
    if match:
        num, alpha = match.groups()
        return int(num), alpha
