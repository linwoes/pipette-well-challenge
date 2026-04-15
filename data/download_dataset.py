#!/usr/bin/env python3
"""
Downloads the pipette_well_dataset from the GitHub repo LFS.
Run: python data/download_dataset.py
"""
import urllib.request
import json
import os
import sys

def download_dataset(output_dir='data'):
    """Download the pipette_well_dataset from GitHub LFS."""
    OID = "88b8799cee91f72f9eebae4b81625f1ef0aee2ae8680ffdbce743d29f99e01cc"
    SIZE = 174132820
    REPO = "linwoes/pipette-well-challenge"
    
    print(f"Downloading pipette_well_dataset.tar.gz ({SIZE / 1e6:.1f} MB)...", file=sys.stderr)
    
    # Get LFS download URL via batch API
    batch_url = f"https://github.com/{REPO}.git/info/lfs/objects/batch"
    payload = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": OID, "size": SIZE}]
    }).encode()
    
    req = urllib.request.Request(batch_url, data=payload, headers={
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
        "User-Agent": "git-lfs/3.0.0"
    })
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        
        if "objects" not in result or not result["objects"]:
            print("Error: LFS batch API returned no objects", file=sys.stderr)
            sys.exit(1)
        
        if "actions" not in result["objects"][0]:
            print("Error: No download action in LFS response", file=sys.stderr)
            sys.exit(1)
        
        dl_url = result["objects"][0]["actions"]["download"]["href"]
        dest = os.path.join(output_dir, 'pipette_well_dataset.tar.gz')
        
        print(f"Downloading from LFS...", file=sys.stderr)
        urllib.request.urlretrieve(dl_url, dest)
        print(f"Saved to {dest}", file=sys.stderr)
        print(f"\nNext step: tar -xzf {dest} -C {output_dir}/", file=sys.stderr)
        
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        print(f"Failed to download from {batch_url}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    download_dataset()
