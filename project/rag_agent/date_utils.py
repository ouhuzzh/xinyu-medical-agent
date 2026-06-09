"""Date and time normalization helpers.

Extracted from node_helpers for focused reusability.
"""

import re
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Compiled regex patterns for date/time parsing
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(r"(\d{1,2})[:：点时]")
_YEAR_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")
_MONTH_DAY_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")
_SLASH_DATE_RE = re.compile(r"(\d{4})[/.](\d{1,2})[/.](\d{1,2})")
_WEEKDAY_RE = re.compile(r"(下|这|本)?\s*周([一二三四五六日天])")

_CN_HOUR_MAP = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15, "十六": 16,
    "十七": 17, "十八": 18, "十九": 19, "二十": 20, "二十一": 21, "二十二": 22, "二十三": 23,
}

_CN_HOUR_RE = re.compile(
    r"(二十三|二十二|二十一|二十|十九|十八|十七|十六|十五|十四|十三|十二|十一|十"
    r"|[一二三四五六七八九])"
)


def _normalize_time_slot(raw_value: str) -> str:
    """Normalize various time-slot formats into HH:MM–HH:MM or HH:MM."""
    if not raw_value:
        return ""
    s = str(raw_value).strip()

    # Already in HH:MM format
    if re.match(r"^\d{1,2}:\d{2}", s):
        return s

    # "14点-15点" → "14:00–15:00"
    m = re.match(r"(\d{1,2})\s*[点时]\s*[-–—到至]\s*(\d{1,2})\s*[点时]?", s)
    if m:
        return f"{int(m.group(1)):02d}:00–{int(m.group(2)):02d}:00"

    # "14点" → "14:00"
    m = re.match(r"(\d{1,2})\s*[点时半]", s)
    if m:
        hour = int(m.group(1))
        minute = "30" if "半" in s else "00"
        return f"{hour:02d}:{minute}"

    # Chinese hour: "下午三点" → "15:00"
    m = _CN_HOUR_RE.search(s)
    if m:
        cn_hour = _CN_HOUR_MAP.get(m.group(1))
        if cn_hour is not None:
            if "下午" in s or "晚上" in s or "傍晚" in s:
                cn_hour = cn_hour + 12 if cn_hour < 12 else cn_hour
            elif "中午" in s and cn_hour == 12:
                cn_hour = 12
            elif "凌晨" in s and cn_hour == 12:
                cn_hour = 0
            return f"{cn_hour:02d}:00"

    return s


def _normalize_date(raw_value: str) -> str:
    """Normalize various date formats into YYYY-MM-DD.

    Returns an empty string if the input cannot be parsed.
    """
    if not raw_value:
        return ""
    s = str(raw_value).strip()

    # Already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    today = date.today()

    # "2024年3月15日"
    m = _YEAR_DATE_RE.search(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            pass

    # "3月15日" (assume current year)
    m = _MONTH_DAY_RE.search(s)
    if m:
        try:
            year = today.year
            d = date(year, int(m.group(1)), int(m.group(2)))
            if d < today:
                d = date(year + 1, int(m.group(1)), int(m.group(2)))
            return d.isoformat()
        except ValueError:
            pass

    # "2024/3/15" or "2024.3.15"
    m = _SLASH_DATE_RE.search(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            pass

    # "下周三" / "这周一" etc.
    m = _WEEKDAY_RE.search(s)
    if m:
        prefix = m.group(1) or "这"
        day_name = m.group(2)
        day_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
        target = day_map.get(day_name)
        if target is not None:
            offset = target - today.weekday()
            if prefix == "下":
                offset += 7
            elif offset <= 0:
                offset += 7
            return (today + timedelta(days=offset)).isoformat()

    # Relative dates
    if "今天" in s or "今日" in s:
        return today.isoformat()
    if "明天" in s:
        return (today + timedelta(days=1)).isoformat()
    if "后天" in s:
        return (today + timedelta(days=2)).isoformat()
    if "大后天" in s:
        return (today + timedelta(days=3)).isoformat()

    return ""
