from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def build_quality_report(output: Path) -> None:
    rows: list[dict] = []
    intervals: list[dict] = []
    timestamp_table = pq.read_table(
        output / "source" / "frame_timestamps.parquet"
    ).to_pydict()
    timestamps = timestamp_table["timestamp_s"]
    sources = [
        ("objects", output / "objects" / "quality.json"),
        ("hands", output / "hands" / "quality.json"),
        ("depth", output / "depth" / "quality.json"),
    ]
    for stage, path in sources:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for frame in data.get("frames", []):
            rows.append(
                {
                    "stage": stage,
                    "frame_index": int(frame["frame_index"]),
                    "timestamp_s": float(timestamps[int(frame["frame_index"])]),
                    "confidence": float(frame["confidence"]),
                    "status": frame["status"],
                    "reasons": json.dumps(frame.get("reasons", [])),
                }
            )
        for interval in data.get("quarantined_intervals", []):
            intervals.append({"stage": stage, **interval})

    quality = output / "quality"
    quality.mkdir(exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), quality / "frame_scores.parquet")
    (quality / "quarantined_intervals.json").write_text(json.dumps(intervals, indent=2))
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row["stage"], {}).setdefault(row["status"], 0)
        counts[row["stage"]][row["status"]] += 1
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Gradyn quality report</title>
<style>body{{font-family:system-ui;max-width:960px;margin:40px auto}}pre{{background:#f4f4f4;padding:16px}}
.warning{{color:#9a5b00}}</style></head><body>
<h1>Gradyn processing report</h1>
<p class="warning">All results are model-derived predictions. Camera-relative hand motion includes camera movement. Depth is relative/ordinal and is not measured in meters.</p>
<h2>Frame status counts</h2><pre>{json.dumps(counts, indent=2)}</pre>
<h2>Quarantined intervals</h2><pre>{json.dumps(intervals, indent=2)}</pre>
</body></html>"""
    (quality / "report.html").write_text(html)
