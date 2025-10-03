"""
Workflow CLI - Simple Command Line Interface for Ichimoku Trading System

This module provides a user-friendly command-line interface to interact with
streamlined data collection utilities in this folder.

Features:
- Interactive menu system
- Input validation
- Progress indicators
- Result display
"""

import sys
import time
from pathlib import Path
from typing import List, Any

# Ensure backend is on the path so "streamline_workflow" package imports work when running directly
sys.path.append(str(Path(__file__).parent.parent))

from data_fetching.collect_historical_data import collect_all_historical_data_for_all_pairs

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class WorkflowCLI:
    """
    Command-line interface for the streamlined workflow (data collection).
    """
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.session_active = False
        
    def print_header(self):
        """Print the application header."""
        print("\n" + "="*70)
        print(f"{Colors.BOLD}{Colors.HEADER}🌊 ICHIMOKU CLOUD TRADING SYSTEM 🌊{Colors.ENDC}")
        print("="*70)
        print(f"{Colors.CYAN}Streamlined Workflow: Data Collection{Colors.ENDC}")
        print("="*70 + "\n")
    
    def print_menu(self):
        """Display the main menu."""
        print(f"\n{Colors.BOLD}=== MAIN MENU ==={Colors.ENDC}")
        print(f"{Colors.GREEN}1.{Colors.ENDC} Collect historical data")
        print(f"{Colors.GREEN}0.{Colors.ENDC} Exit")
        print("-"*30)
    
    def get_user_choice(self, prompt: str, valid_choices: List[str]) -> str:
        """Get validated user input."""
        while True:
            choice = input(f"{Colors.CYAN}{prompt}{Colors.ENDC} ").strip()
            if choice in valid_choices:
                return choice
            print(f"{Colors.WARNING}Invalid choice. Please try again.{Colors.ENDC}")
    
    def get_user_input(self, prompt: str, default: Any = None, 
                      input_type: type = str) -> Any:
        """Get user input with type conversion and default values."""
        if default is not None:
            prompt = f"{prompt} [{default}]"
        
        while True:
            user_input = input(f"{Colors.CYAN}{prompt}: {Colors.ENDC}").strip()
            
            # Use default if no input
            if not user_input and default is not None:
                return default
            
            # Convert to appropriate type
            try:
                if input_type == bool:
                    return user_input.lower() in ['yes', 'y', 'true', '1']
                else:
                    return input_type(user_input)
            except ValueError:
                print(f"{Colors.WARNING}Invalid input. Expected {input_type.__name__}.{Colors.ENDC}")
    
    def display_progress(self, message: str):
        """Display a progress indicator."""
        print(f"\n{Colors.BLUE}⏳ {message}...{Colors.ENDC}", end='', flush=True)
        
        # Simple animated dots
        for _ in range(3):
            time.sleep(0.5)
            print(".", end='', flush=True)
        print()
    
    def run(self):
        """Run the main CLI loop."""
        self.print_header()
        
        while True:
            self.print_menu()
            
            choice = self.get_user_choice("Select option: ", ['0', '1'])
            
            if choice == '0':
                print(f"\n{Colors.GREEN}Thank you for using Ichimoku Cloud Trading System!{Colors.ENDC}")
                print(f"{Colors.CYAN}Good luck with your trading! 📈{Colors.ENDC}\n")
                break
            elif choice == '1':
                self.display_progress("Collecting historical data")
                collect_all_historical_data_for_all_pairs()
            
            # Pause before returning to menu
            if choice != '0':
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")


def main():
    """Main entry point for the CLI."""
    try:
        cli = WorkflowCLI()
        cli.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted by user.{Colors.ENDC}")
        print(f"{Colors.CYAN}Goodbye! 👋{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n{Colors.FAIL}An unexpected error occurred: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}Please check the logs for details.{Colors.ENDC}")


if __name__ == "__main__":
    main()
