"""Run the corrected, streaming 64-channel analysis. See --help for options."""

from src.pipeline import run_cli

if __name__ == '__main__':
    run_cli(default_channels=64)
