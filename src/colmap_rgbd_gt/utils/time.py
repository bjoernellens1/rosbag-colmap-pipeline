"""Timestamp handling for ROS1 and ROS2."""

from typing import Union


def ros_time_to_seconds(time_msg) -> float:
    if hasattr(time_msg, "sec"):
        return time_msg.sec + time_msg.nanosec / 1e9
    elif hasattr(time_msg, "secs"):
        return time_msg.secs + time_msg.nsecs / 1e9
    else:
        raise ValueError(f"Unknown time message format: {type(time_msg)}")


def ros_time_to_nanoseconds(time_msg) -> int:
    if hasattr(time_msg, "sec"):
        return time_msg.sec * 1_000_000_000 + time_msg.nanosec
    elif hasattr(time_msg, "secs"):
        return time_msg.secs * 1_000_000_000 + time_msg.nsecs
    else:
        raise ValueError(f"Unknown time message format: {type(time_msg)}")


def nanoseconds_to_seconds(ns: int) -> float:
    return ns / 1e9


def seconds_to_nanoseconds(s: float) -> int:
    return int(s * 1e9)


def format_timestamp_ns(ns: int) -> str:
    return f"{ns / 1e9:.6f}"


def format_timestamp_iso(ns: int) -> str:
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return dt.isoformat()
