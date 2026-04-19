from pathlib import Path

import matplotlib.pyplot as plt


def use_paper_style() -> None:
    plt.style.use(str(Path(__file__).with_name("paper.mplstyle")))
