"""
Run this file to download data from Dryad and unzip the zip files. Downloaded files end
up in this repostitory's data/ directory.

First create the b2txt25 conda environment. Then in a Terminal, at this repository's
top-level directory (nejm-brain-to-text/), run:

conda activate b2txt25
python download_data.py
"""

import sys
import os
import urllib.request
import urllib.error
import json
import zipfile


########################################################################################
#
# Helpers.
#
########################################################################################


def display_progress_bar(block_num, block_size, total_size, message=""):
    """"""
    bytes_downloaded_so_far = block_num * block_size
    MB_downloaded_so_far = bytes_downloaded_so_far / 1e6
    MB_total = total_size / 1e6
    sys.stdout.write(
        f"\r{message}\t\t{MB_downloaded_so_far:.1f} MB / {MB_total:.1f} MB"
    )
    sys.stdout.flush()


# Where each Dryad file ends up once downloaded/extracted. Files may also sit in a
# data/doi_10_5061_dryad_*/ folder if the dataset was downloaded as one archive.
EXTRACTED_TARGETS = {
    "t15_copyTask_neuralData.zip": "hdf5_data_final",
    "t15_pretrained_rnn_baseline.zip": "t15_pretrained_rnn_baseline",
}


def already_present(data_dirpath, filename):
    search_dirs = [data_dirpath] + [
        os.path.join(data_dirpath, d) for d in os.listdir(data_dirpath) if d.startswith("doi_")
    ]
    names = [filename, EXTRACTED_TARGETS.get(filename)]
    return any(n and os.path.exists(os.path.join(d, n)) for d in search_dirs for n in names)


########################################################################################
#
# Main function.
#
########################################################################################


def main():
    """"""
    DRYAD_DOI = "10.5061/dryad.dncjsxm85"

    ## Make sure the command is being run from the right place and we can see the data/
    ## directory.

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    data_dirpath = os.path.abspath(DATA_DIR)
    assert os.path.exists(
        data_dirpath
    ), "Cannot find the data directory to download into."

    ## Get the list of files from the latest version on Dryad.

    DRYAD_ROOT = "https://datadryad.org"
    urlified_doi = DRYAD_DOI.replace("/", "%2F")

    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"
    with urllib.request.urlopen(versions_url) as response:
        versions_info = json.loads(response.read().decode())

    files_url_path = versions_info["_embedded"]["stash:versions"][-1]["_links"][
        "stash:files"
    ]["href"]
    files_url = f"{DRYAD_ROOT}{files_url_path}"
    with urllib.request.urlopen(files_url) as response:
        files_info = json.loads(response.read().decode())

    file_infos = files_info["_embedded"]["stash:files"]

    ## Download each file into the data directory (and unzip for certain files).

    for file_info in file_infos:
        filename = file_info["path"]

        if filename == "README.md":
            continue

        download_path = file_info["_links"]["stash:download"]["href"]
        download_url = f"{DRYAD_ROOT}{download_path}"

        download_to_filepath = os.path.join(data_dirpath, filename)

        if already_present(data_dirpath, filename):
            print(f"Skipping {filename}: already in {data_dirpath}")
            continue

        try:
            urllib.request.urlretrieve(
            download_url,
                download_to_filepath,
                reporthook=lambda *args: display_progress_bar(
                    *args, message=f"Downloading {filename}"
                ),
            )
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                sys.exit(
                    f"\nDryad refused the anonymous download of {filename} (HTTP {e.code}).\n"
                    f"Download it in a browser from {DRYAD_ROOT}/dataset/doi:{DRYAD_DOI} "
                    f"and put it in {data_dirpath} (unzip .zip files there)."
                )
            raise
        sys.stdout.write("\n")

        # If this file is a zip file, unzip it into the data directory.

        if file_info["mimeType"] == "application/zip":
            print(f"Extracting files from {filename} ...")
            with zipfile.ZipFile(download_to_filepath, "r") as zf:
                zf.extractall(data_dirpath)

    print(f"\nDownload complete. See data files in {data_dirpath}\n")


if __name__ == "__main__":
    main()
