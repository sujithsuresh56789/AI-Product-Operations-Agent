# API Rate Limits and Increasing Them

Default API rate limit is 100 requests/minute per API key on all plans.

Enterprise customers can request a rate limit increase by having their account owner submit a request through the Enterprise support channel; increases up to 1,000 requests/minute are typically approved within 2 business days.

For high-volume sync use cases, recommend the batch endpoint (`POST /v2/batch`) instead of many individual requests — it accepts up to 500 records per call and counts as a single request against the rate limit.
