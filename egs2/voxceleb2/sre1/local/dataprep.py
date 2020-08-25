#!/usr/bin/env python3

"""
Copied from https://github.com/clovaai/voxceleb_trainer/blob/master/dataprep.py
The script downloads the VoxCeleb datasets and converts all files to WAV.
Requirement: ffmpeg and wget running on a Linux system.
"""

import argparse
import os
import subprocess
import pdb
import hashlib
from pathlib import Path
import time
import glob
from zipfile import ZipFile

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="VoxCeleb2 downloader")

    parser.add_argument("--save_path", type=str, default="data", help="Target directory")
    parser.add_argument("--user", type=str, default="user", help="Username")
    parser.add_argument("--password", type=str, default="pass", help="Password")

    parser.add_argument(
        "--download", action="store_true", help="Enable download"
    )
    parser.add_argument(
        "--extract", action="store_true", help="Enable extract"
    )
    parser.add_argument(
        "--convert", action="store_true", help="Enable convert"
    )
    parser.add_argument(
        "--audio_format", type=str, default="flac", choices=["wav", "flac", "mp3"], help="Enable convert"
    )
    return parser


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(args, lines):
    if Path(args.save_path / ".download").exists():
        print("Already downloaded.")
        return

    for line in lines:
        url = line.split()[0]
        md5gt = line.split()[1]
        outfile = url.split("/")[-1]

        ## Download files
        out = subprocess.call(
            "wget %s --user %s --password %s -O %s/%s"
            % (url, args.user, args.password, args.save_path, outfile),
            shell=True,
        )
        if out != 0:
            raise ValueError(
                "Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website."
                % url
            )

        ## Check MD5
        md5ck = md5("%s/%s" % (args.save_path, outfile))
        if md5ck == md5gt:
            print("Checksum successful %s." % outfile)
        else:
            raise ValueError("Checksum failed %s." % outfile)
    with (Path(args.save_path) / ".download").open("w"):
        pass

def concatenate(args, lines):
    for line in lines:
        infile = line.split()[0]
        outfile = line.split()[1]
        md5gt = line.split()[2]

        ## Concatenate files
        out = subprocess.call(
            "cat %s/%s > %s/%s" % (args.save_path, infile, args.save_path, outfile),
            shell=True,
        )

        ## Check MD5
        md5ck = md5("%s/%s" % (args.save_path, outfile))
        if md5ck == md5gt:
            print("Checksum successful %s." % outfile)
        else:
            raise ValueError("Checksum failed %s." % outfile)

        out = subprocess.call("rm %s/%s" % (args.save_path, infile), shell=True)


def extract(args):
    files = glob.glob("%s/*.zip" % args.save_path)

    for fname in files:
        print("Extracting %s" % fname)
        zf = ZipFile(fname, "r")
        zf.extractall(args.save_path)
        zf.close()


def convert(args):
    if Path(args.save_path / ".convert").exists():
        print("Already converted.")
        return

    folders = glob.glob("%s/voxceleb2/*/*/" % args.save_path)
    files = glob.glob("%s/voxceleb2/*/*/*.m4a" % args.save_path)
    files.sort()

    print("Converting files from AAC to WAV")
    for fname in tqdm(files):
        outfile = fname.replace(".m4a", "." + args.audio_format)
        if args.audio_format == "wav":
            acodec = "pcm_s16le"
        elif args.audio_format == "flac":
            acodec = "flac"
        elif args.audio_format == "mp3":
            acodec = "libmp3lame"
        else:
            raise RuntimeError("Not supported audio_format=" + args.audio_format)
        out = subprocess.call(
            "ffmpeg -y -i %s -ac 1 -vn -acodec %s -ar 16000 -sample_fmt s16 %s >/dev/null 2>/dev/null"
            % (acodec, fname, outfile),
            shell=True,
        )
        if out != 0:
            raise ValueError("Conversion failed %s." % fname)

    with (Path(args.save_path) / ".convert").open("w"):
        pass


def main():
    parser = get_parser()
    args = parser.parse_args()
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    with open("local/lists/fileparts.txt", "r", encoding="utf-8") as f:
        fileparts = list(f)

    with open("local/lists/files.txt", "r", encoding="utf-8") as f:
        files = list(f)

    if args.download:
        download(args, fileparts)

    if args.extract:
        if Path(args.save_path / ".extract").exists():
            print("Already extracted.")
        else:
            concatenate(args, files)
            extract(args)
            out = subprocess.call(
                "mv %s/dev/aac/* %s/aac/ && rm -r %s/dev"
                % (args.save_path, args.save_path, args.save_path),
                shell=True,
            )
            out = subprocess.call(
                "mv %s/wav %s/voxceleb1" % (args.save_path, args.save_path), shell=True
            )
            out = subprocess.call(
                "mv %s/aac %s/voxceleb2" % (args.save_path, args.save_path), shell=True
            )
            with (Path(args.save_path) / ".extract").open("w"):
                pass

    if args.convert:
        convert(args)


if __name__ == "__main__":
    main()
