from pathlib import Path
import shutil
from huggingface_hub import HfApi
from tqdm import tqdm

def upload_with_progress(path):
    file_size = Path(path).stat().st_size
    api = HfApi()
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path,
        repo_id="pennyb/prh",
        repo_type="dataset",
    )

# shutil.make_archive('results', 'zip', 'results')
upload_with_progress("results.zip")