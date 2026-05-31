# Security Policy

## Reporting a vulnerability

If you find a security issue in ai-eval-runner, please report it privately by email to security@sarmalinux.com rather than opening a public issue. Include a clear description of the problem, the steps to reproduce it, the commit SHA you tested against, and any proof-of-concept code or output that helps me confirm it. I will credit you in the release notes once a fix ships, unless you ask me not to.

## Response policy

I respond to every report within 7 days. Once I have confirmed an issue I patch it on `main`, cut a tagged release, and keep you updated on progress until it is resolved.

## Supported versions

I ship security fixes for the latest minor release and the latest commit on `main`. Older minor releases do not receive backported fixes, so pin to a current tagged release if you need a stable surface to track.

| Version | Supported |
|---------|-----------|
| 1.1.x   | Yes       |
| 1.0.x   | No        |
| < 1.0   | No        |
