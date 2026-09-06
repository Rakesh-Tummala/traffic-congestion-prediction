"""Browse and fetch snapshots from Caltrans's public CCTV camera network.

Public, no API key required: https://cwwp2.dot.ca.gov/documentation/cctv/cctv.htm
("There is no charge for the use of this data.") Covers CA's 12 Caltrans
districts. Verified against the live feed while building this project —
e.g. district 3 (Sacramento) returns ~100 real, currently-in-service cameras
with direct JPEG snapshot URLs refreshed every ~1 minute.
"""
import argparse

import requests

DISTRICTS = list(range(1, 13))
# Folder segment is NOT zero-padded ("d3") while the filename is ("cctvStatusD03.json") -
# verified against the live feed; a zero-padded folder ("d03") 500s.
JSON_URL_TEMPLATE = "https://cwwp2.dot.ca.gov/data/d{d}/cctv/cctvStatusD{d:02d}.json"


def fetch_cameras(district: int, in_service_only: bool = True, timeout: int = 15) -> list[dict]:
    if district not in DISTRICTS:
        raise ValueError(f"district must be 1-12, got {district}")

    url = JSON_URL_TEMPLATE.format(d=district)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    records = resp.json().get("data", [])

    cameras = []
    for rec in records:
        cctv = rec.get("cctv", {})
        loc = cctv.get("location", {})
        in_service = cctv.get("inService") == "true"
        if in_service_only and not in_service:
            continue
        cameras.append({
            "district": district,
            "name": loc.get("locationName", "").strip(),
            "nearby_place": loc.get("nearbyPlace", "").strip(),
            "route": loc.get("route", ""),
            "county": loc.get("county", ""),
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "in_service": in_service,
            "image_url": cctv.get("imageData", {}).get("static", {}).get("currentImageURL", ""),
            "stream_url": cctv.get("imageData", {}).get("streamingVideoURL", ""),
        })
    return [c for c in cameras if c["image_url"]]


def main():
    parser = argparse.ArgumentParser(description="List or fetch Caltrans public traffic camera snapshots.")
    parser.add_argument("district", type=int, choices=DISTRICTS)
    parser.add_argument("--search", default=None, help="Filter by substring in camera name")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    cameras = fetch_cameras(args.district)
    if args.search:
        needle = args.search.lower()
        cameras = [c for c in cameras if needle in c["name"].lower() or needle in c["nearby_place"].lower()]

    print(f"{len(cameras)} in-service cameras in district {args.district}"
          f"{f' matching {args.search!r}' if args.search else ''}\n")
    for c in cameras[: args.limit]:
        print(f"- {c['name']:<45s} ({c['route']}, {c['nearby_place']})")
        print(f"    {c['image_url']}")


if __name__ == "__main__":
    main()
