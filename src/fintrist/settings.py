"""
This is the config file for fintrist, containing various parameters the user may
wish to modify.
"""
import os
import platform    # For getting the operating system name
import subprocess  # For executing a shell command
from dotenv import load_dotenv

load_dotenv()

def ping(host):
    """
    Returns True if host (str) responds to a ping request.
    Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.
    """

    # Option for the number of packets as a function of
    param = '-n' if platform.system().lower()=='windows' else '-c'

    # Building the command. Ex: "ping -c 1 google.com"
    command = ['ping', param, '1', host]

    return subprocess.run(command).returncode == 0

def choose_host(options):
    """Use the first available host."""
    for host in options:
        if ping(host):
            return host
    raise ConnectionError("No available host connection.")

class ConfigObj():
    APP_HOST = os.getenv('COMPUTERNAME')
    DATABASE_NAME = os.getenv('DB_NAME')
    local = int(os.getenv('DB_LOCAL') or 0)
    if local:
        USERNAME = None
        PASSWORD = None
        DB_HOST = 'localhost'
    else:
        USERNAME = os.getenv('DB_USERNAME')
        PASSWORD = os.getenv('DB_PASSWORD')
        remote_host = os.getenv('DB_HOST')
        lan_host = os.getenv('DB_HOST_LAN')
        if remote_host and lan_host:
            DB_HOST = choose_host([remote_host, lan_host])
        elif remote_host:
            DB_HOST = remote_host
        elif lan_host:
            DB_HOST = lan_host
        else:
            DB_HOST = 'localhost'
    DB_PORT = int(os.getenv('DB_PORT') or 27017)
    TZ = os.getenv('TIMEZONE') or 'UTC'

Config = ConfigObj()
