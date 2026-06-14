"""notify — send the morning brief by email (SMTP), streamlit-free.

Used two ways, same code path:

* the Morning Brief page's "Email this brief" form (send-now, credentials are
  session-only and never stored), and
* ``scripts/daily_brief_email.py`` run by a scheduled GitHub Action so the brief
  lands in the inbox every morning automatically.

Dependency-free: builds a multipart text+HTML message with a small, safe
markdown→HTML renderer (headings, bold, bullet lists, GitHub-style tables) so the
emailed brief reads cleanly without pulling in a markdown library.
"""
from __future__ import annotations

import re
import smtplib
import ssl
from email.message import EmailMessage
from html import escape

_BULLET = re.compile(r"^\s*[-*]\s+")


def _inline(text: str) -> str:
    """Escape, then re-enable **bold** and `code` spans (after escaping)."""
    out = escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return out


def markdown_to_html(md: str) -> str:
    """Small, defensive markdown→HTML (headings, bullets, tables, bold, rules).

    Anything it doesn't recognize is rendered as a paragraph, so it never raises
    and never drops content — worst case a line shows as plain text."""
    lines = md.splitlines()
    html: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        # horizontal rule
        if re.fullmatch(r"-{3,}", line.strip()):
            html.append("<hr>")
            i += 1
            continue
        # headings
        m = re.match(r"(#{1,6})\s+(.*)", line)
        if m:
            lvl = len(m.group(1))
            html.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        # GitHub-style table: header row, separator row, then body rows
        if line.lstrip().startswith("|") and i + 1 < n and \
                re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            def cells(row: str) -> list[str]:
                return [c.strip() for c in row.strip().strip("|").split("|")]
            header = cells(line)
            i += 2  # skip header + separator
            body = []
            while i < n and lines[i].lstrip().startswith("|"):
                body.append(cells(lines[i]))
                i += 1
            thead = "".join(f"<th align='left'>{_inline(c)}</th>" for c in header)
            trs = []
            for r in body:
                tds = "".join(f"<td>{_inline(c)}</td>" for c in r)
                trs.append(f"<tr>{tds}</tr>")
            html.append(
                "<table cellpadding='6' cellspacing='0' "
                "style='border-collapse:collapse;border:1px solid #e5e7eb'>"
                f"<thead style='background:#f6f8fa'><tr>{thead}</tr></thead>"
                f"<tbody>{''.join(trs)}</tbody></table>")
            continue
        # bullet list
        if _BULLET.match(line):
            items = []
            while i < n and _BULLET.match(lines[i]):
                items.append(f"<li>{_inline(_BULLET.sub('', lines[i]))}</li>")
                i += 1
            html.append(f"<ul>{''.join(items)}</ul>")
            continue
        # paragraph
        html.append(f"<p>{_inline(line)}</p>")
        i += 1
    return (
        "<div style=\"font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,"
        "sans-serif;font-size:14px;color:#1f2937;line-height:1.5;max-width:760px\">"
        + "".join(html) + "</div>")


def build_message(*, sender: str, recipients: list[str], subject: str,
                  markdown_body: str) -> EmailMessage:
    """A multipart text+HTML email carrying the brief (markdown as the text part)."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(markdown_body)  # text/plain fallback = the raw markdown
    msg.add_alternative(markdown_to_html(markdown_body), subtype="html")
    return msg


def send_email(*, host: str, port: int, username: str, password: str,
               sender: str, recipients: list[str], subject: str,
               markdown_body: str, use_tls: bool = True, timeout: int = 30) -> None:
    """Send the brief via SMTP. Raises on any failure (caller surfaces the error).

    Port 465 → implicit SSL; otherwise STARTTLS when ``use_tls`` (the common 587
    submission path). Credentials are passed in by the caller and never persisted.
    """
    recipients = [r.strip() for r in recipients if r and r.strip()]
    if not recipients:
        raise ValueError("no recipients given")
    msg = build_message(sender=sender, recipients=recipients, subject=subject,
                        markdown_body=markdown_body)
    ctx = ssl.create_default_context()
    if int(port) == 465:
        with smtplib.SMTP_SSL(host, int(port), timeout=timeout, context=ctx) as s:
            if username:
                s.login(username, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, int(port), timeout=timeout) as s:
            s.ehlo()
            if use_tls:
                s.starttls(context=ctx)
                s.ehlo()
            if username:
                s.login(username, password)
            s.send_message(msg)
