# Security Policy

## Supported versions

Only the latest tagged release is officially supported. If you find a
vulnerability in an older release, please still report it — we'll try to
back-port any state-transition-safety fix to the last two tags.

## Reporting a vulnerability

**Do not** open a public issue for suspected security bugs. Send the
report by email to:

- <1148395497@qq.com>

Please include:

1. The exact release tag (or the commit SHA if it's on `main`).
2. A minimal scenario YAML / seed that reproduces the issue.
3. What you expected to happen vs. what actually happened.
4. If the vulnerability could let an LLM/policy write a state the rules
   should have rejected, a one-sentence description of the rule gap
   (action legality? conflict-resolution ordering? cost settlement?).

Security-relevant issues in this project tend to be about **state
transition safety**: e.g. an agent action bypasses a cost check, hash
chain verification passes on corrupted traces, or replay produces a
different state than the original tick. Those get high priority.

## Paid support / integration

Security advisories and priority patches for commercial deployments are
covered under the **paid integration support / technical support**
channel — same email <1148395497@qq.com> with "WorldLoop support SLA" in
the subject.
