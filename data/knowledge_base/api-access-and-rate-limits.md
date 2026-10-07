---
id: kb-api-access
title: API Access and Rate Limits
category: it_support
---

# API Access and Rate Limits

The Voltix Business API lets you manage users, devices and alerts programmatically.

## API keys

- Administrators create API keys in **Admin > Developers > API keys**.
- Each key can have **read-only** or **read-write** scope.
- Keys are shown only once, when created. Store them in a secrets manager; if a key is lost, revoke it and create a new one.

## Rate limits

| Plan | Requests per minute per key |
|---|---|
| Business | 600 |
| Business Plus | 3,000 |

When the limit is exceeded, the API returns **HTTP 429 Too Many Requests** with a `Retry-After` header indicating how many seconds to wait. Use exponential backoff in your client.

## Errors

- **401 Unauthorized:** the key is missing, invalid or revoked.
- **403 Forbidden:** the key does not have the required scope.
- **5xx errors:** retry with backoff and check the status page (see "Service Status and Maintenance").
