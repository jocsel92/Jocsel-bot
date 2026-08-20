"""
News Filter — Avoid trading around high-impact economic news.
==============================================================
Fetches economic calendar data and blocks trading ±15 minutes
around high-impact events.
"""

import json
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

from config import NEWS_BUFFER_MINUTES

_news_cache: list = []
_cache_date: str = ""


def _fetch_news_today(now_utc: datetime) -> list:
    """
    Fetch today's high-impact economic news events.

    Uses the Forex Factory calendar API (noticias.json fallback).
    Returns list of dicts with 'time' (datetime UTC) and 'title'.
    """
    global _news_cache, _cache_date

    today_str = now_utc.strftime("%Y-%m-%d")
    if _cache_date == today_str and _news_cache:
        return _news_cache

    events = []

    # Try fetching from a free calendar API
    try:
        url = (
            f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        )
        req = Request(url, headers={"User-Agent": "JocselBot/1.0"})
        with urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())

        for item in data:
            impact = item.get("impact", "").lower()
            if impact not in ("high", "holiday"):
                continue

            date_str = item.get("date", "")
            if not date_str:
                continue

            try:
                event_time = datetime.fromisoformat(
                    date_str.replace("Z", "+00:00")
                )
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            if event_time.date() == now_utc.date():
                events.append({
                    "time": event_time,
                    "title": item.get("title", "Unknown"),
                    "currency": item.get("country", ""),
                })
    except Exception as e:
        print(f"⚠️ No se pudo obtener calendario de noticias: {e}")

    _news_cache = events
    _cache_date = today_str
    return events


def is_news_window(now_utc: datetime) -> bool:
    """
    Check if current time is within ±NEWS_BUFFER_MINUTES of a
    high-impact news event.

    Returns True if trading should be blocked.
    """
    events = _fetch_news_today(now_utc)
    buffer = timedelta(minutes=NEWS_BUFFER_MINUTES)

    for event in events:
        event_time = event["time"]
        if (event_time - buffer) <= now_utc <= (event_time + buffer):
            print(
                f"⚠️ Noticia fuerte en ventana: {event['title']} "
                f"({event.get('currency', '')}) @ {event_time.strftime('%H:%M')} UTC"
            )
            return True

    return False
