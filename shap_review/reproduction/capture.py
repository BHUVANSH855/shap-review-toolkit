from __future__ import annotations

import re


def classify_output(stdout: str, stderr: str, returncode: int | None) -> dict:
    text = (stdout + "\n" + stderr).lower()
    return {
        "crash": any(
            x in text
            for x in [
                "segmentation fault",
                "addresssanitizer",
                "undefinedbehavior",
                "fatal python error",
            ]
        ),
        "timeout": returncode is None,
        "exception": bool(re.search(r"traceback \(most recent call last\)", text)),
    }
