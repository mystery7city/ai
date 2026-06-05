#!/usr/bin/env python3
"""Read-only repository scanner for AI/pro그래밍 study folders.

The scanner never deletes, moves, or edits existing project files. It only
creates reports under the selected root's reports/ directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules",
    "staticfiles",
    "_staticfiles",
    "chroma",
    "chroma_upstage",
    "downloads",
}

GIT_EXCLUDE_DIR_NAMES = {
    "venv": "virtual environment",
    ".venv": "virtual environment",
    "__pycache__": "python cache",
    ".ipynb_checkpoints": "jupyter checkpoint",
    "node_modules": "node dependencies",
    "staticfiles": "generated static files",
    "_staticfiles": "generated static files",
    "chroma": "chroma vector database",
    "chroma_upstage": "chroma vector database",
}

GIT_EXCLUDE_FILE_NAMES = {
    ".env": "environment secrets",
}

GIT_EXCLUDE_SUFFIXES = {
    ".pyc": "python bytecode",
    ".sqlite3": "sqlite database",
    ".db": "database",
    ".h5": "model file",
    ".pkl": "model/data artifact",
    ".joblib": "model/data artifact",
    ".pb": "model file",
    ".exe": "executable",
    ".zip": "archive",
    ".tar": "archive",
    ".gz": "archive",
}

GIT_EXCLUDE_NAME_PATTERNS = [
    (re.compile(r"^\.env\..+", re.IGNORECASE), "environment secrets"),
    (re.compile(r"\.tfevents.*", re.IGNORECASE), "training event log"),
]

SENSITIVE_PATH_KEYWORDS = [
    "secret",
    "key",
    "api",
    "token",
    "password",
    "passwd",
    "credential",
    "auth",
    ".env",
    "settings.py",
    "connection.py",
]

SENSITIVE_CONTENT_KEYWORDS = [
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "UPSTAGE_API_KEY",
    "HUGGINGFACE",
    "HF_TOKEN",
    "SECRET_KEY",
    "PASSWORD",
    "TOKEN",
    "API_KEY",
    "DATABASE_URL",
]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".example",
    ".csv",
    ".sql",
    ".html",
    ".css",
    ".js",
    ".ipynb",
}

MODEL_SUFFIXES = {".h5", ".hdf5", ".pkl", ".joblib", ".pth", ".pb", ".npy", ".npz", ".bin", ".traineddata"}
DATA_SUFFIXES = {".csv", ".tsv", ".jsonl", ".json", ".xlsx", ".sqlite3", ".db"}
WEB_SUFFIXES = {".html", ".css", ".js"}
PYTHON_SUFFIXES = {".py", ".ipynb"}


def rel_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def format_size(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.2f} MB"
    if size >= 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size} B"


def is_probably_text(path: Path) -> bool:
    if path.name.lower().startswith(".env"):
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def safe_scan_content(path: Path) -> list[str]:
    """Return matched sensitive keywords without exposing values."""
    if not is_probably_text(path):
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read(1024 * 1024)
    except OSError:
        return []
    lowered = text.lower()
    return [keyword for keyword in SENSITIVE_CONTENT_KEYWORDS if keyword.lower() in lowered]


def git_exclude_reason_for_file(path: Path) -> str | None:
    name = path.name
    suffix = path.suffix.lower()
    if name in GIT_EXCLUDE_FILE_NAMES:
        return GIT_EXCLUDE_FILE_NAMES[name]
    if suffix in GIT_EXCLUDE_SUFFIXES:
        return GIT_EXCLUDE_SUFFIXES[suffix]
    for pattern, reason in GIT_EXCLUDE_NAME_PATTERNS:
        if pattern.search(name):
            return reason
    return None


def classify_folder(name: str, extensions: Counter[str], path_text: str, file_count: int) -> str:
    lowered = f"{name} {path_text}".lower()
    if file_count == 0:
        return "빈 폴더"
    if any(term in lowered for term in ["llm", "rag", "chatbot", "openai", "pinecone", "huggingface", "chroma"]):
        return "AI/LLM/RAG"
    if any(term in lowered for term in ["deep", "machine", "learning", "model", "mnist", "catboost", "xgboost", "lightgbm"]):
        return "머신러닝/딥러닝"
    if any(term in lowered for term in ["economy", "happiness", "data", "analysis", "프로젝트 1"]):
        return "데이터 분석"
    if any(term in lowered for term in ["flask", "django", "web", "html", "css", "javascript", "backend"]):
        return "웹앱/백엔드"
    if "python" in lowered or extensions.get(".py", 0) or extensions.get(".ipynb", 0):
        return "Python 기초/실습"
    if name in {"docs", "reports", "archive", "assets", "scripts", "tools"}:
        return "문서/관리"
    if extensions.get(".pyc", 0) > extensions.get(".py", 0) or "venv" in lowered:
        return "정리 필요"
    return "분류 불명"


def recommendation_for_large(path: Path) -> str:
    suffix = path.suffix.lower()
    parts = {part.lower() for part in path.parts}
    if suffix in MODEL_SUFFIXES:
        return "Git LFS 또는 외부 보관"
    if suffix in {".sqlite3", ".db"} or "chroma" in parts or "chroma_upstage" in parts:
        return "Git 제외 또는 수동 확인"
    if suffix in {".mp4", ".mp3", ".png", ".jpg", ".jpeg", ".gif"}:
        return "Git LFS 또는 외부 보관"
    if suffix in DATA_SUFFIXES:
        return "Git LFS / 외부 보관 / 수동 확인"
    return "수동 확인"


def empty_top_summary(name: str) -> dict:
    return {
        "name": name,
        "file_count": 0,
        "dir_count": 0,
        "size_bytes": 0,
        "extensions": Counter(),
        "git_exclude_count": 0,
        "sensitive_count": 0,
        "large_file_count": 0,
        "sample_paths": [],
    }


def get_top_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if not relative.parts:
        return "."
    if len(relative.parts) == 1 and path.is_file():
        return "."
    return relative.parts[0]


def scan_repository(root: Path, large_threshold_mb: float) -> dict:
    root = root.resolve()
    large_threshold_bytes = int(large_threshold_mb * 1024 * 1024)

    top_summaries: dict[str, dict] = {}
    all_extensions: Counter[str] = Counter()
    git_exclude_candidates = []
    sensitive_candidates = []
    large_files = []
    skipped_dirs = []

    total_files = 0
    total_dirs = 0
    total_size = 0

    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        if ".git" in current.parts:
            continue

        total_dirs += len(dirnames)
        top_name = get_top_name(current, root)
        top = top_summaries.setdefault(top_name, empty_top_summary(top_name))
        if current != root:
            top["dir_count"] += len(dirnames)

        excluded_here = []
        for dirname in list(dirnames):
            child = current / dirname
            if dirname in DEFAULT_EXCLUDED_DIRS:
                reason = GIT_EXCLUDE_DIR_NAMES.get(dirname, "excluded scan directory")
                skipped_dirs.append({"path": rel_path(child, root), "type": reason})
                if dirname != ".git":
                    git_exclude_candidates.append({"path": rel_path(child, root), "type": reason, "kind": "directory"})
                    top_name_for_child = get_top_name(child, root)
                    top_child = top_summaries.setdefault(top_name_for_child, empty_top_summary(top_name_for_child))
                    top_child["git_exclude_count"] += 1
                excluded_here.append(dirname)
        for dirname in excluded_here:
            dirnames.remove(dirname)

        for filename in filenames:
            path = current / filename
            try:
                stat = path.stat()
            except OSError:
                continue

            size = stat.st_size
            suffix = path.suffix.lower() or "[no extension]"
            rel = rel_path(path, root)
            total_files += 1
            total_size += size
            all_extensions[suffix] += 1

            top_name = get_top_name(path, root)
            top = top_summaries.setdefault(top_name, empty_top_summary(top_name))
            top["file_count"] += 1
            top["size_bytes"] += size
            top["extensions"][suffix] += 1
            if len(top["sample_paths"]) < 5:
                top["sample_paths"].append(rel)

            reason = git_exclude_reason_for_file(path)
            if reason:
                git_exclude_candidates.append({"path": rel, "type": reason, "kind": "file"})
                top["git_exclude_count"] += 1

            lowered_rel = rel.lower()
            path_keyword_matches = [keyword for keyword in SENSITIVE_PATH_KEYWORDS if keyword in lowered_rel]
            content_keyword_matches = safe_scan_content(path)
            if path_keyword_matches or content_keyword_matches:
                reasons = []
                if path_keyword_matches:
                    reasons.append("path keywords: " + ", ".join(sorted(set(path_keyword_matches))))
                if content_keyword_matches:
                    reasons.append("content keywords: " + ", ".join(sorted(set(content_keyword_matches))))
                sensitive_candidates.append(
                    {
                        "path": rel,
                        "type": "; ".join(reasons),
                        "note": "value not printed",
                    }
                )
                top["sensitive_count"] += 1

            if size >= large_threshold_bytes:
                large_files.append(
                    {
                        "path": rel,
                        "size_bytes": size,
                        "extension": path.suffix.lower() or "[no extension]",
                        "recommendation": recommendation_for_large(path),
                    }
                )
                top["large_file_count"] += 1

    for top_name, summary in top_summaries.items():
        summary["classification"] = classify_folder(
            top_name,
            summary["extensions"],
            " ".join(summary["sample_paths"]),
            summary["file_count"],
        )
        summary["top_extensions"] = summary["extensions"].most_common(10)
        del summary["extensions"]

    cleanup_priorities = build_priorities(git_exclude_candidates, sensitive_candidates, large_files, top_summaries)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "large_threshold_mb": large_threshold_mb,
        "total_files": total_files,
        "total_dirs": total_dirs,
        "total_size_bytes": total_size,
        "top_extensions": all_extensions.most_common(20),
        "top_folders": sorted(top_summaries.values(), key=lambda item: item["name"].lower()),
        "git_exclude_candidates": git_exclude_candidates,
        "sensitive_candidates": sensitive_candidates,
        "large_files": sorted(large_files, key=lambda item: item["size_bytes"], reverse=True),
        "skipped_dirs": skipped_dirs,
        "cleanup_priorities": cleanup_priorities,
        "readme_summary": build_readme_summary(top_summaries),
    }


def build_priorities(git_candidates: list[dict], sensitive: list[dict], large_files: list[dict], top_summaries: dict[str, dict]) -> list[str]:
    priorities = []
    if sensitive:
        priorities.append("민감정보 후보(.env, Secret/API/Token/Password 키워드 파일)를 먼저 수동 검토")
    if any("virtual environment" in item["type"] for item in git_candidates):
        priorities.append("venv/.venv 폴더는 재현 가능한 requirements로 대체 가능한지 검토")
    if any("python cache" in item["type"] or item["path"].endswith(".pyc") for item in git_candidates):
        priorities.append("__pycache__ 및 .pyc 추적 제외/정리 검토")
    if any("chroma" in item["type"] for item in git_candidates):
        priorities.append("Chroma DB는 재생성 가능 여부 확인 후 Git 제외 또는 외부 보관 검토")
    if large_files:
        priorities.append("10MB 이상 데이터/모델/미디어 파일은 Git LFS 또는 외부 보관 기준 결정")
    largest = sorted(top_summaries.values(), key=lambda item: item["size_bytes"], reverse=True)[:3]
    if largest:
        names = ", ".join(item["name"] for item in largest)
        priorities.append(f"용량이 큰 최상위 폴더부터 분리 기준 수립: {names}")
    return priorities


def build_readme_summary(top_summaries: dict[str, dict]) -> dict:
    active = []
    management = []
    needs_review = []
    for item in sorted(top_summaries.values(), key=lambda value: value["name"].lower()):
        classification = item.get("classification", "분류 불명")
        entry = {
            "name": item["name"],
            "classification": classification,
            "file_count": item["file_count"],
            "size": format_size(item["size_bytes"]),
        }
        if classification in {"문서/관리", "빈 폴더"}:
            management.append(entry)
        elif classification in {"정리 필요", "분류 불명"}:
            needs_review.append(entry)
        else:
            active.append(entry)
    return {
        "active_or_study_folders": active,
        "management_folders": management,
        "needs_review": needs_review,
    }


def markdown_table_row(values: list[str | int]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def render_markdown(report: dict) -> str:
    lines = [
        "# AI Study Repository Cleaner Report",
        "",
        f"생성일: {report['generated_at']}",
        f"분석 루트: `{report['root']}`",
        "",
        "## 1. 전체 요약",
        f"- 총 파일 수: {report['total_files']}",
        f"- 총 폴더 수: {report['total_dirs']}",
        f"- 전체 용량: {format_size(report['total_size_bytes'])}",
        f"- Git 제외 후보: {len(report['git_exclude_candidates'])}",
        f"- 민감정보 후보: {len(report['sensitive_candidates'])}",
        f"- 대용량 파일 후보: {len(report['large_files'])}",
        "",
        "주요 확장자:",
        "",
    ]
    for extension, count in report["top_extensions"][:10]:
        lines.append(f"- `{extension}`: {count}")

    lines.extend(
        [
            "",
            "## 2. 최상위 폴더별 요약",
            "",
            "| 폴더 | 분류 | 파일 수 | 하위 폴더 수 | 용량 | 주요 확장자 | 위험 후보 | 대용량 |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for item in report["top_folders"]:
        extensions = ", ".join(f"{ext}:{count}" for ext, count in item["top_extensions"][:5]) or "-"
        lines.append(
            markdown_table_row(
                [
                    f"`{item['name']}`",
                    item["classification"],
                    item["file_count"],
                    item["dir_count"],
                    format_size(item["size_bytes"]),
                    extensions,
                    item["git_exclude_count"] + item["sensitive_count"],
                    item["large_file_count"],
                ]
            )
        )

    lines.extend(["", "## 3. Git 제외 후보", ""])
    if report["git_exclude_candidates"]:
        for item in report["git_exclude_candidates"][:200]:
            lines.append(f"- `{item['path']}` - {item['type']}")
        if len(report["git_exclude_candidates"]) > 200:
            lines.append(f"- ...and {len(report['git_exclude_candidates']) - 200} more")
    else:
        lines.append("- 없음")

    lines.extend(["", "## 4. 민감정보 후보", "", "주의: 실제 키/비밀번호 값은 출력하지 않음.", ""])
    if report["sensitive_candidates"]:
        for item in report["sensitive_candidates"][:200]:
            lines.append(f"- `{item['path']}` - {item['type']}; 값은 출력하지 않음")
        if len(report["sensitive_candidates"]) > 200:
            lines.append(f"- ...and {len(report['sensitive_candidates']) - 200} more")
    else:
        lines.append("- 없음")

    lines.extend(["", "## 5. 대용량 파일 후보", ""])
    if report["large_files"]:
        lines.append("| 경로 | 크기 | 확장자 | 추천 조치 |")
        lines.append("|---|---:|---|---|")
        for item in report["large_files"]:
            lines.append(
                markdown_table_row(
                    [
                        f"`{item['path']}`",
                        format_size(item["size_bytes"]),
                        item["extension"],
                        item["recommendation"],
                    ]
                )
            )
    else:
        lines.append("- 없음")

    lines.extend(["", "## 6. 정리 우선순위 추천", ""])
    if report["cleanup_priorities"]:
        for index, item in enumerate(report["cleanup_priorities"], start=1):
            lines.append(f"{index}. {item}")
    else:
        lines.append("- 현재 자동 추천할 정리 우선순위가 없습니다.")

    lines.extend(
        [
            "",
            "## 7. 다음 작업 추천",
            "",
            "- 민감정보 후보를 사람이 직접 열어 실제 값 포함 여부를 확인합니다.",
            "- `.gitignore` 규칙을 정리하기 전에 이미 추적 중인 파일 목록을 별도로 확인합니다.",
            "- 대용량 데이터/모델은 Git LFS, 외부 보관, 재생성 스크립트 중 하나로 관리 기준을 정합니다.",
            "- README 초안은 JSON 보고서의 `readme_summary` 항목을 바탕으로 작성합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(root: Path, report: dict) -> tuple[Path, Path]:
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)
    markdown_path = reports_dir / "repo_cleaner_report.md"
    json_path = reports_dir / "repo_cleaner_report.json"
    markdown_path.write_text(render_markdown(report), encoding="utf-8-sig")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return markdown_path, json_path


def print_summary(report: dict, markdown_path: Path, json_path: Path) -> None:
    print("AI Study Repository Cleaner")
    print("===========================")
    print(f"Root: {report['root']}")
    print(f"Generated: {report['generated_at']}")
    print(f"Total files: {report['total_files']}")
    print(f"Total folders: {report['total_dirs']}")
    print(f"Total size: {format_size(report['total_size_bytes'])}")
    print(f"Git exclude candidates: {len(report['git_exclude_candidates'])}")
    print(f"Sensitive candidates: {len(report['sensitive_candidates'])} (values are never printed)")
    print(f"Large file candidates: {len(report['large_files'])}")
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only cleaner report generator for AI study repositories.")
    parser.add_argument("--root", default=".", help="Repository root to scan. Defaults to current directory.")
    parser.add_argument("--large-threshold-mb", type=float, default=10.0, help="Large file threshold in MB. Defaults to 10.")
    parser.add_argument("--json", action="store_true", help="Print JSON report to stdout after writing report files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root does not exist or is not a directory: {root}")

    report = scan_repository(root, args.large_threshold_mb)
    markdown_path, json_path = write_reports(root, report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_summary(report, markdown_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
