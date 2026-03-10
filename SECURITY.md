# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
responsibly. **Do not open a public issue.**

### Preferred Method

Use [GitHub Security Advisories](https://github.com/rohanvinaik/Wayfinder/security/advisories/new) to report vulnerabilities privately.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Assessment**: Within 7 days
- **Fix timeline**: Depends on severity; critical issues prioritized

## Supported Versions

Security updates are applied to the latest release on the default branch.

## Security Tools

This project uses automated security scanning:
- **gitleaks** — secrets detection in commits
- **Bandit** — Python SAST analysis
- **pip-audit** — supply chain vulnerability scanning
- **SonarCloud** — continuous code quality and security analysis
