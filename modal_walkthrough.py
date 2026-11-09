"""Run with: modal run modal_walkthrough.py"""

import modal

from modal_train import SOURCE_DIR, image, volume


app = modal.App("tiny-zero-lora-walkthrough")
image = image.add_local_file(SOURCE_DIR / "walkthrough.py", "/app/walkthrough.py")


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/data": volume})
def walkthrough() -> None:
    import runpy

    runpy.run_path("/app/walkthrough.py", run_name="__main__")
    volume.commit()
