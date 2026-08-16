from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.table import Table
from rich import box


class Logger:

    def __init__(self):

        self.console = Console()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = Path("outputs") / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def banner(self):

        banner = r"""
 █████╗ ███████╗████████╗██╗  ██╗███████╗██████╗
██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗
███████║█████╗     ██║   ███████║█████╗  ██████╔╝
██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗
██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
"""

        self.console.print()

        self.console.print(
            Panel(
                Align.center(
                    f"[bold cyan]{banner}[/bold cyan]\n"
                    "[bold white]Modular AI Training Framework[/bold white]"
                ),
                border_style="cyan",
                title="🚀 AETHER",
            )
        )

    def configuration(self, config):

        table = Table(
            title="Experiment Configuration",
            box=box.ROUNDED,
        )

        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model", config.model["name"])
        table.add_row("Dataset", config.dataset["name"])
        table.add_row("Trainer", config.trainer["name"])
        table.add_row("Epochs", str(config.training["epochs"]))
        table.add_row(
            "Learning Rate",
            str(config.training["learning_rate"]),
        )

        self.console.print(table)

    def info(self, msg):
        self.console.print(f"[cyan][INFO][/cyan] {msg}")

    def success(self, msg):
        self.console.print(f"[green][SUCCESS][/green] {msg}")

    def warning(self, msg):
        self.console.print(f"[yellow][WARNING][/yellow] {msg}")

    def error(self, msg):
        self.console.print(f"[red][ERROR][/red] {msg}")