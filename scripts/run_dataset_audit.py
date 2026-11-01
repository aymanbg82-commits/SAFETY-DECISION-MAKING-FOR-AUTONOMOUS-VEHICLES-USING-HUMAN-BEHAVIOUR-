"""
Script to execute the Indian Road Dataset audit and generate documentation report.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.data.dataset_audit import IndianRoadDatasetAuditor


def main():
    print("=" * 70)
    print("RUNNING INDIAN ROAD DATASET AUDIT (thirdeyelabs/indian-road-dataset)")
    print("=" * 70)

    auditor = IndianRoadDatasetAuditor()
    results = auditor.run_full_audit()

    meta = results["metadata"]
    feas = results["feasibility_assessment"]

    print(f"Dataset ID:          {meta['dataset_id']}")
    print(f"License:             {meta['license']}")
    print(f"Total Storage:       {meta['total_storage_gb']} GB across {meta['shard_count']} tar shards")
    print(f"Clips / Frames:      {meta['total_clips']} clips / {meta['annotated_frames']} annotated frames")
    print(f"Object Detections:   {meta['object_detections']} (with ByteTrack track IDs)")
    print(f"Sampling Frequency:  {meta['keyframe_sampling_rate_hz']} Hz")
    print(f"Number of Classes:   {len(meta['classes'])}")
    print(f"Verified Classes:    {', '.join(c['name'] for c in meta['classes'])}")
    print("-" * 70)
    print("PHYSICAL TRAJECTORY FEASIBILITY ASSESSMENT:")
    print(f"[Coordinates]        {feas['coordinate_frame_gap']}")
    print(f"[Sampling Rate]      {feas['sampling_frequency_limitation']}")
    print(f"[Physical TTC]       {feas['ttc_calculation_invalidity']}")
    print(f"[Conclusion]         {feas['conclusion_and_role_in_project']}")
    print("=" * 70)
    print("Audit artifact written to: outputs/metrics/indian_road_dataset_audit.json")


if __name__ == "__main__":
    main()
