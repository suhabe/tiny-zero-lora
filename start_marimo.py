from pathlib import Path
import secrets

import modal


def main():
    local_notebook = Path(__file__).resolve().parent / "notebook.py"
    if not local_notebook.is_file():
        raise FileNotFoundError(f"Notebook not found: {local_notebook}")

    app = modal.App.lookup("marimo-notebook", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("marimo", "numpy", "pandas", "matplotlib")
        .add_local_file(
            local_notebook,
            "/notebooks/notebook.py",
            copy=True,
        )
    )

    token = secrets.token_urlsafe(32)

    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "marimo", "-q", "edit", "/notebooks/notebook.py",
            "--host", "0.0.0.0",
            "--port", "2718",
            "--headless",
            "--token-password", token,
            app=app,
            image=image,
            workdir="/notebooks",
            encrypted_ports=[2718],
            timeout=60 * 60,
            # gpu="A10G",
        )

    try:
        url = sandbox.tunnels()[2718].url
        print(f"Open: {url}/?access_token={token}", flush=True)
        print("Press Ctrl+C to stop the notebook.", flush=True)
        sandbox.wait()
    finally:
        print("\nStopping notebook...", flush=True)
        sandbox.terminate(wait=True)
        print("Sandbox stopped.", flush=True)


if __name__ == "__main__":
    main()