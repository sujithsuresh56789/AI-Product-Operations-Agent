# Known Issue: Crash on Uploads Over 50MB

**Symptom:** The browser tab freezes and eventually crashes (white screen) when uploading files larger than approximately 50MB.

**Root cause (tracked as ENG-901):** Large files are currently processed synchronously on the main thread instead of via chunked/background upload, which blocks the UI and can exhaust browser memory on larger files.

**Severity:** High — this blocks data migration workflows for customers with large datasets.

**Workaround:** Split files into parts under 50MB using a spreadsheet tool or CLI split utility, and upload sequentially. Escalate to Engineering if the customer's migration is time-sensitive.

**Status:** A chunked upload rewrite is in progress; ETA end of Q3 2026.
