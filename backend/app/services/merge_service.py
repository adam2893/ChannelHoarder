"""Orchestrate merging of multi-part video episodes.

This service handles the high-level logic of:
1. Checking if all parts of a multi-part episode group are downloaded
2. Triggering the ffmpeg concat merge
3. Updating database records after merge
4. Cleaning up individual part files
5. Writing metadata for the merged episode
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session
from app.models import Channel, Video, DownloadLog
from app.services.metadata_service import write_episode_nfo, write_tvshow_nfo
from app.services.multi_part_service import merge_video_files, strip_part_from_title
from app.services.naming_service import build_output_path
from app.services.notification_service import NotificationService
from app.utils.file_utils import sanitize_filename

logger = logging.getLogger(__name__)


async def check_and_merge_group(video_pk: int) -> bool:
    """Check if all parts of a multi-part group are downloaded and merge if ready.

    Called after each individual part download completes. If this part belongs
    to a multi-part group AND all other parts are also downloaded, this will
    trigger the merge process.

    Args:
        video_pk: The primary key (id) of the video that just finished downloading.

    Returns:
        True if a merge was triggered and completed, False otherwise.
    """
    async with async_session() as db:
        # Load the video that just completed
        video = await db.get(Video, video_pk)
        if not video or not video.episode_group_key:
            return False  # Not a multi-part video

        # Find ALL videos in the same group
        result = await db.execute(
            select(Video)
            .where(Video.episode_group_key == video.episode_group_key)
            .order_by(Video.part_number.asc())
        )
        group_videos = list(result.scalars().all())

        if len(group_videos) < 2:
            return False  # Single video or group not yet complete

        # Determine expected total from the group
        # Use the maximum total_parts seen across all group members
        expected_total = max(
            (v.total_parts or 0 for v in group_videos), default=0
        )
        if expected_total < 2:
            return False

        # Check if ALL expected parts are downloaded
        downloaded_parts = {
            v.part_number: v
            for v in group_videos
            if v.status == "completed" and v.file_path and v.part_number
        }

        if len(downloaded_parts) < expected_total:
            logger.info(
                "Multi-part group %s: %d/%d parts downloaded, waiting for more",
                video.episode_group_key,
                len(downloaded_parts),
                expected_total,
            )
            return False

        # All parts are downloaded — check we have a contiguous set
        missing = []
        for p in range(1, expected_total + 1):
            if p not in downloaded_parts:
                missing.append(p)

        if missing:
            logger.warning(
                "Multi-part group %s has gaps — missing parts %s. "
                "Some parts may have failed. Skipping merge.",
                video.episode_group_key, missing,
            )
            return False

        # ── All parts ready — perform merge ──────────────────────────────
        logger.info(
            "All %d parts of group %s are downloaded. Starting merge...",
            expected_total, video.episode_group_key,
        )

        # Get channel info for naming and metadata
        channel = await db.get(Channel, video.channel_id)
        if not channel:
            logger.error("Channel not found for video %d", video_pk)
            return False

        # Sort parts by part_number and collect file paths
        sorted_parts = sorted(downloaded_parts.values(), key=lambda v: v.part_number)
        part_files = [v.file_path for v in sorted_parts if v.file_path]

        # Determine the "primary" video (part 1) — this becomes the merged episode
        primary = sorted_parts[0]

        # Build clean title (without part indicator) for the merged file
        # Use the title from the first part, stripped of its part indicator
        from app.services.multi_part_service import DEFAULT_MULTI_PART_PATTERN, detect_multi_part
        pattern = channel.multi_part_pattern or DEFAULT_MULTI_PART_PATTERN
        mp_match = detect_multi_part(primary.title, pattern, channel.channel_id)
        merged_title = mp_match.clean_title if mp_match else primary.title

        # Build output path for merged file
        merged_output_base = build_output_path(
            channel_name=channel.channel_name,
            video_title=merged_title,
            video_id=primary.video_id,
            upload_date=primary.upload_date,
            season=primary.season,
            episode=primary.episode,
            naming_template=channel.naming_template,
            base_dir=channel.download_dir,
        )
        merged_file_path = merged_output_base + ".mp4"

        # Perform the merge (runs ffmpeg — can be slow for large files)
        try:
            loop = asyncio.get_running_loop()
            output_path, total_duration = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    merge_video_files,
                    part_files,
                    merged_file_path,
                ),
                timeout=900,  # 15 minute timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                "Merge timed out for group %s after 15 minutes",
                video.episode_group_key,
            )
            return False
        except Exception as e:
            logger.error(
                "Merge failed for group %s: %s",
                video.episode_group_key, e,
            )
            return False

        # ── Post-merge: update database and clean up ─────────────────────
        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        # Update primary video to point to merged file
        primary.file_path = output_path
        primary.file_size = file_size
        primary.duration = total_duration or primary.duration
        primary.title = merged_title
        # Keep the group key for reference but it's no longer actionable
        # part_number and total_parts remain for informational purposes

        db.add(DownloadLog(
            video_id=primary.id,
            event="merged",
            message=f"Merged {expected_total} parts into single episode ({file_size} bytes, ~{total_duration // 60} min)",
        ))

        # Mark secondary parts as "merged" and clean up their files
        for part_video in sorted_parts[1:]:
            old_file = part_video.file_path
            part_video.status = "merged"
            part_video.file_path = None
            part_video.file_size = None

            # Delete individual part files
            if old_file and os.path.exists(old_file):
                try:
                    os.remove(old_file)
                    logger.info("Deleted part file: %s", old_file)
                    # Also remove the part's NFO file
                    nfo_path = os.path.splitext(old_file)[0] + ".nfo"
                    if os.path.exists(nfo_path):
                        os.remove(nfo_path)
                        logger.info("Deleted part NFO: %s", nfo_path)
                except OSError as e:
                    logger.warning("Failed to delete part file %s: %s", old_file, e)

        # Write metadata for the merged episode
        write_tvshow_nfo(
            channel_name=channel.channel_name,
            channel_id=channel.channel_id,
            channel_url=channel.channel_url,
            description=channel.description,
            thumbnail_url=channel.thumbnail_url,
            base_dir=channel.download_dir,
            platform=channel.platform,
        )
        write_episode_nfo(
            channel_name=channel.channel_name,
            video_title=merged_title,
            video_id=primary.video_id,
            description=primary.description,
            upload_date=primary.upload_date,
            season=primary.season,
            episode=primary.episode,
            duration=total_duration or primary.duration,
            thumbnail_url=primary.thumbnail_url,
            video_file_path=output_path,
            platform=channel.platform,
        )

        # Update channel download count
        channel.downloaded_count = (
            await db.scalar(
                select(func.count(Video.id))
                .where(Video.channel_id == channel.id)
                .where(Video.status == "completed")
            )
        ) or 0

        await db.commit()

        # Broadcast notification
        await NotificationService.broadcast("episode_merged", {
            "video_id": primary.video_id,
            "title": merged_title,
            "parts_merged": expected_total,
            "file_size": file_size,
            "episode_group_key": video.episode_group_key,
        })

        logger.info(
            "Successfully merged %d parts into: %s",
            expected_total, output_path,
        )
        return True
