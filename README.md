# Reproduction

To reproduce results, do the following:

1. Create a new repository and create a custom Python3.10+ virtual environment in the repository (by doing ``python -m venv .`` while being in that repository) and activate it using ``source bin/activate``
2. Clone this repository to yours
3. Install the requirements using ``pip install -r requirements.txt``
4. Run `download_dataset.py` to download the dataset automatically.
5. Then you can run the distributed training loop with ``torchrun --nproc_per_node=<HOWEVER_MANY_GPUS_YOU_HAVE> train_distributed.py`` with ``--fp32`` as an optional argument that is parsed (default is ``bf16``).
6. To run unit tests, run ``test_project.py``.

Read the report for some more analysis. You can also check out the analyses in the text files.

You can also inspect ``silurian.ipynb`` and run each module yourself if you want to see my testing notebook.
