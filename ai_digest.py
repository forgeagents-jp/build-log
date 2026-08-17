#!/usr/bin/env python3
"""
AI news digest -> Discord webhook.

Pulls the last N hours of AI news from several free, key-less sources,
dedupes and ranks them, and posts a single formatted embed to a Discord
channel via webhook.

The webhook URL is read from the DISCORD_WEBHOOK_URL environment variable
(or a .env file next to this script). It is never logged or printed.

Usage:
    python ai_digest.py --dry-run        # print what would be posted
    python ai_digest.py                  # post to Discord
    python ai_digest.py --hours 48 --max-items 12
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable
from urllib.parse import urlparse, urlunparse

import feedparser
import requests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

USER_AGENT = "ai-digest/1.0 (Discord digest bot)"
HTTP_TIMEOUT = 20

# Keywords used to decide whether a general-tech story counts as AI news.
AI_TERMS = [
    "ai", "a.i.", "artificial intelligence", "llm", "llms", "gpt", "claude",
    "gemini", "anthropic", "openai", "deepmind", "mistral", "llama", "qwen",
    "machine learning", "neural", "transformer", "diffusion", "rag",
    "agent", "agentic", "inference", "fine-tune", "fine-tuning", "embedding",
    "prompt", "multimodal", "hugging face", "pytorch", "model weights",
    "reinforcement learning", "chatbot", "copilot", "cursor", "vector database",
]

# RSS/Atom feeds. All free, no keys. Add or remove freely.
RSS_FEEDS = [
    ("Simon Willison", "https://simonwillison.net/atom/everything/"),
    ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml"),
    ("Google Research", "https://research.google/blog/rss/"),
    ("MIT Tech Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
    ("Ars Technica AI", "https://feeds.arstechnica.com/arstechnica/technology-lab"),
]

# Feeds that carry non-AI content too, so entries must pass the keyword filter.
MIXED_FEEDS = {"Ars Technica AI", "Simon Willison"}

# Max entries any single source can contribute to a section, so one prolific
# blog cannot take over the digest.
PER_SOURCE_CAP = 2

# arXiv categories to sample from.
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]
ARXIV_PER_CATEGORY = 12

HN_MIN_POINTS = 40

EMBED_COLOR = 0x5865F2  # Discord blurple
MAX_DESC_CHARS = 4000   # Discord hard limit is 4096; leave headroom


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class Item:
    title: str
    url: str
    source: str
    published: datetime
    score: float = 0.0
    extra: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def domain(self) -> str:
        try:
            return urlparse(self.url).netloc.replace("www.", "")
        except Exception:
            return ""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def clean(text: str) -> str:
    """Strip HTML, collapse whitespace, unescape entities."""
    text = re.sub(r"<[^>]+>", "", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def is_ai_related(text: str) -> bool:
    low = f" {text.lower()} "
    for term in AI_TERMS:
        # word-boundary match so "ai" doesn't fire on "chair" or "said"
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low):
            return True
    return False


def canonical(url: str) -> str:
    """Normalize a URL for dedupe: drop query, fragment, trailing slash."""
    try:
        p = urlparse(url)
        path = p.path.rstrip("/") or "/"
        return urlunparse((p.scheme, p.netloc.replace("www.", ""), path, "", "", "")).lower()
    except Exception:
        return (url or "").lower()


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())[:60]


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def parse_feed_date(entry) -> datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime.fromtimestamp(time.mktime(val), tz=timezone.utc)
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

def fetch_hackernews(since: datetime, verbose: bool = False) -> list[Item]:
    """Front-page-quality HN stories mentioning AI, via the free Algolia API."""
    items: list[Item] = []
    url = "https://hn.algolia.com/api/v1/search_by_date"
    params = {
        "tags": "story",
        "numericFilters": f"created_at_i>{int(since.timestamp())},points>{HN_MIN_POINTS}",
        "hitsPerPage": 100,
    }
    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        hits = r.json().get("hits", [])
    except Exception as e:
        if verbose:
            print(f"  ! Hacker News failed: {e}", file=sys.stderr)
        return items

    for h in hits:
        title = clean(h.get("title") or "")
        link = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        if not title or not is_ai_related(title):
            continue
        try:
            published = datetime.fromtimestamp(h.get("created_at_i", 0), tz=timezone.utc)
        except Exception:
            continue
        points = int(h.get("points") or 0)
        comments = int(h.get("num_comments") or 0)
        items.append(Item(
            title=title,
            url=link,
            source="Hacker News",
            published=published,
            score=points + comments * 0.5,
            extra=f"{points} pts · {comments} comments",
            tags=["discussion"],
        ))
    if verbose:
        print(f"  Hacker News: {len(items)} AI stories")
    return items


def fetch_rss(since: datetime, verbose: bool = False) -> list[Item]:
    items: list[Item] = []
    for name, feed_url in RSS_FEEDS:
        try:
            parsed = feedparser.parse(
                feed_url, request_headers={"User-Agent": USER_AGENT}
            )
        except Exception as e:
            if verbose:
                print(f"  ! {name} failed: {e}", file=sys.stderr)
            continue

        count = 0
        for entry in parsed.entries[:40]:
            title = clean(getattr(entry, "title", ""))
            link = getattr(entry, "link", "")
            if not title or not link:
                continue
            published = parse_feed_date(entry)
            if published is None or published < since:
                continue
            summary = clean(getattr(entry, "summary", ""))[:400]
            # Topic-specific feeds are trusted; mixed feeds get keyword-filtered.
            if name in MIXED_FEEDS and not is_ai_related(f"{title} {summary}"):
                continue
            age_hours = max((now_utc() - published).total_seconds() / 3600, 0.5)
            items.append(Item(
                title=title,
                url=link,
                source=name,
                published=published,
                score=60 / age_hours,
                extra="",
                tags=["article"],
            ))
            count += 1
        if verbose:
            status = getattr(parsed, "status", "?")
            print(f"  {name}: {count} items in window (HTTP {status})")
    return items


def fetch_arxiv(since: datetime, verbose: bool = False) -> list[Item]:
    """Recent arXiv submissions in AI categories, via the public Atom API."""
    items: list[Item] = []
    for cat in ARXIV_CATEGORIES:
        try:
            r = requests.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": f"cat:{cat}",
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": ARXIV_PER_CATEGORY,
                },
                timeout=HTTP_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            r.raise_for_status()
            parsed = feedparser.parse(r.text)
        except Exception as e:
            if verbose:
                print(f"  ! arXiv {cat} failed: {e}", file=sys.stderr)
            continue

        count = 0
        for entry in parsed.entries:
            published = parse_feed_date(entry)
            if published is None or published < since:
                continue
            title = clean(getattr(entry, "title", ""))
            link = getattr(entry, "link", "")
            if not title or not link:
                continue
            items.append(Item(
                title=title,
                url=link,
                source=f"arXiv {cat}",
                published=published,
                score=1.0,
                extra="",
                tags=["paper"],
            ))
            count += 1
        if verbose:
            print(f"  arXiv {cat}: {count} papers in window")
        time.sleep(3)  # arXiv asks for 3s between API calls
    return items


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def dedupe(items: Iterable[Item]) -> list[Item]:
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    out: list[Item] = []
    for item in sorted(items, key=lambda i: i.score, reverse=True):
        u, t = canonical(item.url), title_key(item.title)
        if u in seen_urls or (t and t in seen_titles):
            continue
        seen_urls.add(u)
        if t:
            seen_titles.add(t)
        out.append(item)
    return out


def limit_per_source(items: list[Item], cap: int, total: int) -> list[Item]:
    """Keep the list diverse: no single source may dominate a section."""
    counts: dict[str, int] = {}
    out: list[Item] = []
    for item in items:
        if counts.get(item.source, 0) >= cap:
            continue
        counts[item.source] = counts.get(item.source, 0) + 1
        out.append(item)
        if len(out) >= total:
            break
    return out


def build_embed(items: list[Item], hours: int, max_items: int) -> dict:
    half = max(max_items // 2, 1)
    discussion = limit_per_source(
        [i for i in items if "discussion" in i.tags], cap=half, total=half
    )
    articles = limit_per_source(
        [i for i in items if "article" in i.tags], cap=PER_SOURCE_CAP, total=half
    )
    papers = limit_per_source(
        [i for i in items if "paper" in i.tags], cap=PER_SOURCE_CAP, total=3
    )

    lines: list[str] = []

    def section(header: str, entries: list[Item], show_extra: bool = False) -> None:
        if not entries:
            return
        lines.append(f"**{header}**")
        for it in entries:
            title = truncate(it.title, 110).replace("[", "(").replace("]", ")")
            meta = it.extra if (show_extra and it.extra) else it.domain
            lines.append(f"• [{title}]({it.url})" + (f"  ·  *{meta}*" if meta else ""))
        lines.append("")

    section("Being discussed", discussion, show_extra=True)
    section("Worth reading", articles)
    section("Fresh papers", papers)

    if not lines:
        lines = [f"Nothing notable in the last {hours} hours. Quiet night."]

    description = truncate("\n".join(lines).strip(), MAX_DESC_CHARS)
    stamp = now_utc().strftime("%b %d, %Y")

    return {
        "title": f"AI Digest — {stamp}",
        "description": description,
        "color": EMBED_COLOR,
        "footer": {"text": f"Last {hours}h · {len(items)} items scanned · trust but verify"},
        "timestamp": now_utc().isoformat(),
    }


def post_to_discord(webhook_url: str, embed: dict, username: str) -> None:
    payload = {
        "username": username,
        "embeds": [embed],
        "allowed_mentions": {"parse": []},  # never ping anyone
    }
    r = requests.post(webhook_url, json=payload, timeout=HTTP_TIMEOUT,
                      headers={"User-Agent": USER_AGENT})
    if r.status_code == 429:
        retry = r.json().get("retry_after", 5)
        time.sleep(float(retry) + 1)
        r = requests.post(webhook_url, json=payload, timeout=HTTP_TIMEOUT,
                          headers={"User-Agent": USER_AGENT})
    if r.status_code not in (200, 204):
        # Deliberately does not echo the URL — it is a credential.
        raise RuntimeError(f"Discord rejected the post: HTTP {r.status_code} {r.text[:300]}")


def load_dotenv(path: str) -> None:
    """Minimal .env loader so the URL never has to be typed on a command line."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Post an AI news digest to Discord.")
    ap.add_argument("--hours", type=int, default=24, help="lookback window (default 24)")
    ap.add_argument("--max-items", type=int, default=10, help="max links in the post")
    ap.add_argument("--dry-run", action="store_true", help="print instead of posting")
    ap.add_argument("--username", default="AI Digest", help="webhook display name")
    ap.add_argument("--quiet", action="store_true", help="suppress per-source logging")
    args = ap.parse_args()

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    verbose = not args.quiet

    since = now_utc() - timedelta(hours=args.hours)
    if verbose:
        print(f"Collecting AI news since {since:%Y-%m-%d %H:%M UTC}\n")

    raw: list[Item] = []
    raw += fetch_hackernews(since, verbose)
    raw += fetch_rss(since, verbose)
    raw += fetch_arxiv(since, verbose)

    items = dedupe(raw)
    if verbose:
        print(f"\n{len(raw)} raw -> {len(items)} after dedupe")

    embed = build_embed(items, args.hours, args.max_items)

    if args.dry_run:
        print("\n" + "=" * 68)
        print(embed["title"])
        print("=" * 68)
        print(embed["description"])
        print("-" * 68)
        print(embed["footer"]["text"])
        print("=" * 68)
        print("\n(dry run — nothing posted)")
        return 0

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print(
            "DISCORD_WEBHOOK_URL is not set.\n"
            "Put it in a .env file next to this script:\n"
            '  DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."\n'
            "Do not pass it as a command-line argument — it lands in your shell history.",
            file=sys.stderr,
        )
        return 1

    post_to_discord(webhook_url, embed, args.username)
    print(f"Posted {len(items)} items to Discord.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
