#!/usr/bin/env python3
"""Reliably download a Google Drive file with resume and size validation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from gdown.download import (
    _get_filename_from_response,
    _get_modified_time_from_response,
    _get_session,
    get_url_from_gdrive_confirmation,
)
from gdown.parse_url import parse_url


CHUNK_SIZE = 512 * 1024


def resolve_final_response(
    url: str,
    use_cookies: bool = True,
    verify: bool = True,
    user_agent: str | None = None,
) -> tuple[requests.Session, requests.Response, str, str]:
    if user_agent is None:
        user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/39.0.2171.95 Safari/537.36"
        )

    sess, cookies_file = _get_session(
        proxy=None,
        use_cookies=use_cookies,
        user_agent=user_agent,
        return_cookies_file=True,
    )

    file_id, is_gdrive_link = parse_url(url, warning=False)
    url_origin = url

    while True:
        res = sess.get(url, stream=True, verify=verify)
        if not (file_id and is_gdrive_link):
            return sess, res, url_origin, url

        if use_cookies:
            from http.cookiejar import MozillaCookieJar

            cookie_jar = MozillaCookieJar(cookies_file)
            for cookie in sess.cookies:
                cookie_jar.set_cookie(cookie)
            cookie_jar.save()

        if "Content-Disposition" in res.headers:
            return sess, res, url_origin, url

        url = get_url_from_gdrive_confirmation(res.text)


def download_with_resume(
    url: str,
    output: Path,
    max_attempts: int = 20,
    sleep_seconds: int = 3,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    part_path = output.with_name(output.name + ".part")

    for attempt in range(1, max_attempts + 1):
        sess = None
        res = None
        try:
            sess, res, url_origin, final_url = resolve_final_response(url)

            last_modified = _get_modified_time_from_response(res)
            remote_name = _get_filename_from_response(res)
            if remote_name and output.name != remote_name:
                print(
                    f"[attempt {attempt}] remote filename={remote_name}, "
                    f"using output={output.name}",
                    flush=True,
                )

            start_size = part_path.stat().st_size if part_path.exists() else 0
            total = res.headers.get("Content-Length")
            if total is None:
                raise RuntimeError("Missing Content-Length header")
            total_size = int(total)

            if start_size > total_size:
                raise RuntimeError(
                    f"Local partial file is larger than remote size: "
                    f"{start_size} > {total_size}"
                )

            if start_size == total_size and total_size > 0:
                part_path.replace(output)
                if last_modified is not None:
                    mtime = last_modified.timestamp()
                    os.utime(output, (mtime, mtime))
                print(
                    f"[attempt {attempt}] partial file already complete: {output}",
                    flush=True,
                )
                return output

            if start_size:
                res.close()
                headers = {"Range": f"bytes={start_size}-"}
                res = sess.get(final_url, headers=headers, stream=True, verify=True)
                partial_len = res.headers.get("Content-Length")
                if partial_len is None:
                    raise RuntimeError("Resume response missing Content-Length")
                expected_remaining = total_size - start_size
                if int(partial_len) != expected_remaining:
                    raise RuntimeError(
                        "Unexpected resume size: "
                        f"got {partial_len}, expected {expected_remaining}"
                    )

            print(
                f"[attempt {attempt}] downloading from {start_size / (1024**2):.1f}MB "
                f"to {total_size / (1024**2):.1f}MB",
                flush=True,
            )
            if url_origin != final_url:
                print(f"[attempt {attempt}] redirected to {final_url}", flush=True)

            bytes_written = start_size
            with open(part_path, "ab") as f:
                for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bytes_written += len(chunk)

            actual_size = part_path.stat().st_size
            if actual_size != total_size:
                print(
                    f"[attempt {attempt}] incomplete download: "
                    f"{actual_size}/{total_size} bytes",
                    flush=True,
                )
                time.sleep(sleep_seconds)
                continue

            part_path.replace(output)
            if last_modified is not None:
                mtime = last_modified.timestamp()
                os.utime(output, (mtime, mtime))
            print(f"[attempt {attempt}] download complete: {output}", flush=True)
            return output
        finally:
            if res is not None:
                res.close()
            if sess is not None:
                sess.close()

    raise RuntimeError(
        f"Failed to download complete file after {max_attempts} attempts: {output}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Google Drive file URL")
    parser.add_argument("--output", required=True, help="Destination file path")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Maximum number of resume attempts",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=3,
        help="Pause between failed attempts",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    download_with_resume(
        url=args.url,
        output=output,
        max_attempts=args.max_attempts,
        sleep_seconds=args.sleep_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
