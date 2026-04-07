"""Multi-part episode detection, grouping, and merge orchestration.

Detects when YouTube videos are parts of the same episode (e.g., Thai shows
uploaded in 4 parts) and orchestrates merging them into a single file after
all parts are downloaded.

Common title patterns supported by the default regex:
  - "ShowName ตอนจบ พากย์ไทย (1/4)"
  - "ShowName EP01 [Part 2/4]"
  - "ShowName Part 3 of 4"
  - "ShowName พาร์ท 1/4"
  - "ShowName (2/4)"
  - "ShowName 1/4"

The detection works by:
1. Matching the title against a regex to extract part_number and total_parts
2. Stripping the part indicator from the title to get a "group key"
3. Videos with the same group key (within the same channel) are considered
   parts of the same episode
"""

import hashlib
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from app.utils.file_utils import sanitize_filename

logger = logging.getLogger(__name__)

# Default regex pattern for detecting multi-part episodes.
# Captures group 1: part_number, group 2: total_parts
# Handles: (1/4), [1/4], Part 1/4, พาร์ท 1/4, Part 1 of 4, bare 1/4
DEFAULT_MULTI_PART_PATTERN = (
    r"[\[\(]?\s*(?:Part\s+|พาร์ท\s+)?"
    r"(\d+)\s*(?:/|of)\s*(\d+)"
    r"[\]\)]?"
)


@dataclass
class MultiPartMatch:
    """Result of matching a video title against the multi-part pattern."""
    part_number: int
    total_parts: int
    group_key: str
    clean_title: str
    # Span of the matched part indicator in the original title (for stripping)
    match_start: int
    match_end: int


def detect_multi_part(
    title: str,
    pattern: str,
    channel_id: str,
) -> Optional[MultiPartMatch]:
    """Try to detect if a video title indicates it's part of a multi-part episode.

    Args:
        title: The video title from YouTube.
        pattern: Regex pattern with two capture groups: (part_number, total_parts).
        channel_id: Channel ID used to namespace group keys across channels.

    Returns:
        MultiPartMatch if the title matches a multi-part pattern, None otherwise.
    """
    if not title or not pattern:
        return None

    try:
        match = re.search(pattern, title, re.IGNORECASE)
    except re.error:
        logger.warning("Invalid multi-part regex pattern: %s", pattern)
        return None

    if not match:
        return None

    part_number = int(match.group(1))
    total_parts = int(match.group(2))

    # Validate: part must be >= 1 and total must be >= 2
    if total_parts < 2 or part_number < 1 or part_number > total_parts:
        return None

    # Compute clean title by removing the part indicator from the original title
    clean_title = title[:match.start()] + title[match.end():]
    # Clean up any leftover separators, brackets, whitespace artifacts
    clean_title = re.sub(r"\s*[([\s]-\s*[)\]]\s*$", "", clean_title.strip())
    clean_title = re.sub(r"\s+", " ", clean_title).strip()
    # Remove trailing dashes, pipes, colons that were separators before the part
    clean_title = re.sub(r"\s*[-–—|:]\s*$", "", clean_title).strip()
    # Remove leading brackets if the match started with one
    clean_title = clean_title.strip()

    if not clean_title:
        clean_title = title  # Fallback to original title

    # Create a stable group key from clean_title + channel_id
    # Normalize whitespace and case for grouping
    normalized_key = re.sub(r"\s+", " ", clean_title.lower().strip())
    group_key = hashlib.md5(f"{channel_id}:{normalized_key}".encode()).hexdigest()[:16]

    return MultiPartMatch(
        part_number=part_number,
        total_parts=total_parts,
        group_key=group_key,
        clean_title=clean_title,
        match_start=match.start(),
        match_end=match.end(),
    )


def strip_part_from_title(title: str, match: MultiPartMatch) -> str:
    """Remove the part indicator (e.g., ' (1/4)') from the video title."""
    return match.clean_title


def merge_video_files(
    part_files: list[str],
    output_path: str,
) -> tuple[str, int]:
    """Concatenate multiple MP4 video files using ffmpeg concat demuxer.

    This is the fastest method — no re-encoding. It requires all input files
    to have the same codec parameters (which they will, since they're all
    downloaded by yt-dlp with the same quality settings).

    Args:
        part_files: List of absolute file paths to the part MP4 files,
                    ordered by part_number.
        output_path: Absolute path for the merged output file.

    Returns:
        Tuple of (output_path, total_duration_seconds).

    Raises:
        RuntimeError if ffmpeg fails.
    """
    if not part_files:
        raise RuntimeError("No part files provided for merge")

    logger.info(
        "Merging %d video parts into: %s",
        len(part_files), output_path,
    )

    # Verify all input files exist
    for f in part_files:
        if not os.path.exists(f):
            raise RuntimeError(f"Part file does not exist: {f}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get total duration from all parts
    total_duration = 0
    for f in part_files:
        dur = _get_video_duration(f)
        if dur:
            total_duration += dur

    # Create a temporary concat list file
    concat_content = ""
    for f in part_files:
        # Escape single quotes in path for ffmpeg concat format
        escaped = f.replace("'", "'\\''")
        concat_content += f"file '{escaped}'\n"

    # Write concat file to same directory as output (avoids cross-device issues)
    concat_file = output_path + ".concat.txt"
    try:
        with open(concat_file, "w", encoding="utf-8") as fh:
            fh.write(concat_content)

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",          # No re-encoding — fast stream copy
            "-movflags", "+faststart",  # Optimize for streaming
            output_path,
        ]

        logger.info("Running ffmpeg concat: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for large files
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed (exit code {result.returncode}):\n"
                f"STDOUT: {result.stdout[-2000:]}\n"
                f"STDERR: {result.stderr[-2000:]}"
            )

        if not os.path.exists(output_path):
            raise RuntimeError(
                f"ffmpeg completed but output file was not created: {output_path}"
            )

        output_size = os.path.getsize(output_path)
        logger.info(
            "Merge complete: %s (%d bytes, ~%d min)",
            output_path, output_size, total_duration // 60,
        )

        return output_path, total_duration

    finally:
        # Clean up concat list file
        if os.path.exists(concat_file):
            try:
                os.remove(concat_file)
            except OSError:
                pass


def _get_video_duration(file_path: str) -> int:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return int(float(result.stdout.strip()))
    except (ValueError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Could not get duration for %s: %s", file_path, e)
    return 0
