# /// script
# dependencies = [
#     "marimo",
# ]
# ///
import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Hello from Modal 👋

    This notebook runs on Modal.
    Move the slider to change the sine wave.
    """)
    return


@app.cell
def _(mo):
    frequency = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=2,
        label="Frequency",
        show_value=True,
    )
    frequency
    return (frequency,)


@app.cell
def _(frequency, np, plt):
    _x = np.linspace(0, 2 * np.pi, 500)
    _y = np.sin(frequency.value * _x)

    _fig, _ax = plt.subplots()
    _ax.plot(_x, _y)
    _ax.set(
        title=f"Sine wave — frequency {frequency.value}",
        xlabel="x",
        ylabel="sin(frequency × x)",
    )
    _ax.grid(alpha=0.3)
    _fig
    return


if __name__ == "__main__":
    app.run()
