from pathlib import Path

from rich.text import Text
from textual.app import App
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Static

HERE = Path(__file__).parent

VERSION = "0.1.0"
AUTHOR = "Shreyan Banerjee"
GITHUB = "https://github.com/shreyanbruh/afterburner"


def load_art(name, fallback):
    path = HERE / name
    if path.exists():
        return Text.from_ansi(path.read_text(encoding="utf-8"))
    return Text(fallback)


def build_details():
    text = Text()
    rows = [("Version", VERSION, None), ("Author", AUTHOR, None), ("GitHub", GITHUB, GITHUB)]
    for i, (label, value, link) in enumerate(rows):
        text.append("• ", style="#ff7a1a")
        text.append(f"{label}: ", style="bold")
        text.append(value, style=f"underline link {link}" if link else "")
        if i < len(rows) - 1:
            text.append("\n")
    return text


LOGO = load_art("logo.ans", "salmonberry")
TITLE = load_art("title.ans", "afterburner")


class TUI(App):
    CSS_PATH = "afterburner.tcss"

    def compose(self):
        with Vertical(id="main-container"):
            with Horizontal(id="header"):
                yield Static(LOGO, id="logo")
                with Vertical(id="info"):
                    yield Static(TITLE, id="title")
                    yield Static(build_details(), id="details")
            yield Static("Output Window", id="output")
            yield Input(placeholder="type a command, or 'help'", id="command")


if __name__ == "__main__":
    TUI().run()