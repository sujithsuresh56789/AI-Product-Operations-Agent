# Troubleshooting: Export to CSV Fails Silently

**Symptom:** Clicking "Export to CSV" on the Reports dashboard shows a brief loading spinner but no file is downloaded.

**Root cause (known issue, tracked as ENG-884):** The export endpoint silently fails when a browser's pop-up/download blocker intercepts the generated file for reports containing more than 10,000 rows. No error is surfaced to the user.

**Workaround for customers:**
1. Narrow the date range to reduce row count below 10,000, or
2. Check the browser's download/pop-up blocker icon in the address bar and allow downloads from the app domain, then retry.

**Status:** Fix scheduled for release v4.3. Support should apply the workaround and link the customer to the status page for v4.3.
