<!-- Thank you for the PR! Please fill in the checklist below so we can
review quickly. -->

## Summary

<!-- One-line summary + 2-4 sentences of context. -->

## Changes

- …
- …

## Verification

- [ ] `pytest worldloop-kernel/tests -q` → green
- [ ] `pytest worldloop-scenarios/tests -q` → green
- [ ] `pytest worldloop-adapters/tests -q` → green
- [ ] `pytest worldloop-data/tests -q` → green
- [ ] New scenario YAMLs compile via `test_compiler.test_compile_yaml_file`
      (or explain which rule was added / removed).
- [ ] No new upstream-layer imports that violate the
      kernel → scenarios → adapters → data direction.

## Release note

<!-- One short line that would make sense in a GitHub Release body, or
"None" if this is a pure internal refactor. -->
