"""Custom Pygments styles for Rich syntax highlighting."""

from pygments.styles.onedark import OneDarkStyle


class DarkerOneDarkStyle(OneDarkStyle):
    """One Dark with a slightly darker, purple-tinged background."""

    background_color = "#1b1424"
