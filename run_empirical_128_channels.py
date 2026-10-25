"""Run the same corrected workflow with all 128 candidate channels."""

from src.pipeline import run_cli

if __name__ == '__main__':
    run_cli(default_channels=128)
