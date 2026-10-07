---
id: kb-business-sso
title: Single Sign-On for Voltix Business
category: it_support
---

# Single Sign-On for Voltix Business

Single sign-on (SSO) lets employees sign in to Voltix Business with your company identity provider.

## Availability

SSO is available on the **Business Plus** plan only. It is not available on the standard Business plan.

## Supported protocols and providers

- **SAML 2.0** and **OpenID Connect (OIDC)**.
- Tested providers: Microsoft Entra ID, Okta, Google Workspace and OneLogin.

## Setting up SAML

1. In **Admin > Security > Single sign-on**, click **Add SAML provider** and copy the ACS URL and Entity ID.
2. Create a SAML application in your identity provider with those values.
3. Upload the provider's metadata XML to Voltix.
4. Click **Test connection** with an administrator account.
5. Turn on **Enforce SSO** only after the test succeeds.

Keep at least one administrator able to sign in with a password (break-glass account) in case the identity provider is unavailable.

## Automatic provisioning

Business Plus also supports **SCIM 2.0** to create and deactivate users automatically from your identity provider (see "User Seats and Licenses").
