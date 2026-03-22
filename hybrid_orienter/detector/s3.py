import json
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from .settings import settings

log = logging.getLogger(__name__)

LOCK_EXPIRATION = 600


def join_urls(url1: str, url2: str):
    return f"{url1.rstrip('/')}/{url2.lstrip('/')}"


def get_model_name(pretrained_model_name_or_path: str):
    return pretrained_model_name_or_path.split("/")[0]


def download_file(remote_path: str, local_path: str, chunk_size: int = 1024 * 1024):
    local_path = Path(local_path)
    try:
        response = requests.get(remote_path, stream=True, allow_redirects=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        filename = local_path.name
        pbar = tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"Downloading {filename}", miniters=1,
        )
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        pbar.close()
        return local_path
    except Exception as e:
        if local_path.exists():
            local_path.unlink()
        log.error(f"Download error for file {remote_path}: {e}")
        raise


def check_manifest(local_dir: str):
    local_dir = Path(local_dir)
    manifest_path = local_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        for file in manifest["files"]:
            if not (local_dir / file).exists():
                return False
    except Exception:
        return False
    return True


def download_directory(remote_path: str, local_dir: str):
    model_name = get_model_name(remote_path)
    s3_url = join_urls(settings.S3_BASE_URL, remote_path)
    if check_manifest(local_dir):
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_file = join_urls(s3_url, "manifest.json")
        manifest_path = os.path.join(temp_dir, "manifest.json")
        download_file(manifest_file, manifest_path)

        with open(manifest_path) as f:
            manifest = json.load(f)

        pbar = tqdm(
            desc=f"Downloading {model_name} model to {local_dir}",
            total=len(manifest["files"]),
        )
        with ThreadPoolExecutor(max_workers=settings.PARALLEL_DOWNLOAD_WORKERS) as executor:
            futures = []
            for file in manifest["files"]:
                remote_file = join_urls(s3_url, file)
                local_file = os.path.join(temp_dir, file)
                futures.append(executor.submit(download_file, remote_file, local_file))
            for future in futures:
                future.result()
                pbar.update(1)
        pbar.close()

        for file in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, file), local_dir)


class S3DownloaderMixin:
    s3_prefix = "s3://"

    @classmethod
    def get_local_path(cls, pretrained_model_name_or_path) -> str:
        if pretrained_model_name_or_path.startswith(cls.s3_prefix):
            pretrained_model_name_or_path = pretrained_model_name_or_path.replace(cls.s3_prefix, "")
            cache_dir = settings.MODEL_CACHE_DIR
            local_path = os.path.join(cache_dir, pretrained_model_name_or_path)
            os.makedirs(local_path, exist_ok=True)
        else:
            local_path = ""
        return local_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if not pretrained_model_name_or_path.startswith(cls.s3_prefix):
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        local_path = cls.get_local_path(pretrained_model_name_or_path)
        remote_path = pretrained_model_name_or_path.replace(cls.s3_prefix, "")

        retries = 3
        delay = 5
        for attempt in range(retries):
            try:
                download_directory(remote_path, local_path)
                break
            except Exception as e:
                log.error(f"Download error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise

        return super().from_pretrained(local_path, *args, **kwargs)
