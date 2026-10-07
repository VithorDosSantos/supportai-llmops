---
id: kb-security-and-data-protection
title: Security and Data Protection
category: it_support
---

# Security and Data Protection

## Two-factor authentication

Voltix supports two-factor authentication (2FA) with:

- **authenticator apps** (TOTP codes);
- **security keys** (FIDO2 / WebAuthn).

SMS codes are not supported. Voltix Business administrators can require 2FA for all users in **Admin > Security > Policies**.

## Encryption

- Data is encrypted **in transit** with TLS 1.2 or higher.
- Data is encrypted **at rest** with AES-256.
- Cloud Backup files are also encrypted on your device before upload.

## Data location

Customer data is stored in data centers in the **United States**.

## Reporting a security incident

If you suspect unauthorized access to your account:

1. Change your password and sign out of all sessions in **Account > Security**.
2. Report it to **security@voltix.example**. Business customers can also use the 24/7 critical incident channel in the Admin console.

Voltix staff will **never** ask for your password or 2FA codes.
