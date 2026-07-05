"""
Fermi LAT extended-data query client (TRANSIENT-class events).

Submits queries to the Fermi LAT Data Server and downloads EV00/PH00 FITS.
Weekly photon files are SOURCE-class only and unsuitable for GRB timing.

  https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

QUERY_URL = "https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi"
RESULT_URL_RE = re.compile(
    r'results of your query may be found at <a href="(https://fermi[^"]+)"', re.I
)
FITS_URL_RE = re.compile(
    r"wget (https://fermi\.gsfc\.nasa\.gov/FTP/fermi/data/lat/queries/[A-Za-z0-9_]*\.fits)"
)


def submit_query(
    ra_deg: float,
    dec_deg: float,
    met_start: float,
    met_stop: float,
    *,
    radius_deg: float = 20.0,
    e_min_mev: float = 100.0,
    e_max_mev: float = 300_000.0,
    lat_datatype: str = "Extended",
    spacecraft: bool = False,
    timeout: int = 120,
) -> str:
    """Return QueryResults.cgi URL."""
    try:
        import requests
    except ImportError as exc:
        raise ImportError("pip install requests") from exc

    payload = {
        "shapefield": str(radius_deg),
        "coordsystem": "J2000",
        "coordfield": f"{ra_deg},{dec_deg}",
        "destination": "query",
        "timefield": f"{met_start:.3f},{met_stop:.3f}",
        "timetype": "MET",
        "energyfield": f"{e_min_mev:g},{e_max_mev:g}",
        "photonOrExtendedOrNone": lat_datatype,
        "spacecraft": "on" if spacecraft else "off",
    }
    resp = requests.post(QUERY_URL, data=payload, timeout=timeout)
    resp.raise_for_status()
    match = RESULT_URL_RE.search(resp.text)
    if not match:
        raise RuntimeError("Fermi LAT query did not return a result URL")
    return match.group(1)


def poll_fits_urls(
    result_url: str,
    *,
    max_wait_min: float = 10.0,
    poll_interval_s: float = 15.0,
    timeout: int = 120,
) -> list[str]:
    """Poll QueryResults page until FITS wget links appear."""
    try:
        import requests
    except ImportError as exc:
        raise ImportError("pip install requests") from exc

    deadline = time.time() + max_wait_min * 60
    while time.time() < deadline:
        resp = requests.post(result_url, timeout=timeout)
        resp.raise_for_status()
        urls = FITS_URL_RE.findall(resp.text)
        if urls:
            return urls
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Fermi query timed out after {max_wait_min} min: {result_url}")


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def pick_event_fits(urls: list[str]) -> str:
    """Prefer EV00 (extended events), fall back to PH00 (photons)."""
    for suffix in ("_EV00.fits", "_PH00.fits"):
        for url in urls:
            if url.endswith(suffix):
                return url
    raise RuntimeError(f"No event/photon FITS in query results: {urls}")


def query_and_download(
    dest_dir: Path,
    ra_deg: float,
    dec_deg: float,
    met_start: float,
    met_stop: float,
    *,
    tag: str = "grb",
    force: bool = False,
    **query_kwargs,
) -> tuple[Path, dict]:
    """
    Submit query, download event FITS, save manifest.

    Returns (fits_path, manifest_dict).
    """
    manifest_path = dest_dir / "query_manifest.json"
    manifest: dict = {}
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        cached = Path(manifest.get("event_fits", ""))
        if cached.exists():
            print(f"Using cached {cached}")
            return cached, manifest

    print("Submitting Fermi LAT extended query (may take 1–3 min)...")
    result_url = submit_query(ra_deg, dec_deg, met_start, met_stop, **query_kwargs)
    print(f"Query submitted: {result_url}")
    urls = poll_fits_urls(result_url)
    event_url = pick_event_fits(urls)
    query_id = event_url.split("/")[-1].split("_")[0]
    dest = dest_dir / f"{tag}_{query_id}_EV00.fits"
    if not dest.exists() or force:
        download_file(event_url, dest)

    manifest = {
        "result_url": result_url,
        "event_url": event_url,
        "all_urls": urls,
        "event_fits": str(dest),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "met_start": met_start,
        "met_stop": met_stop,
        "query_kwargs": query_kwargs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return dest, manifest
