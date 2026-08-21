# Release Checklist — WorldLoop v0.1.3-beta.1

> Follow in order. Each step has a green/red check against the audit
> output from `docs/06_audits/2026-08-21/`.

## 1. Clone-green (pre-flight, *already covered by Actions*)

- [ ] `git clone https://github.com/Crqse/worldloop.git` (fresh checkout,
      no PYTHONPATH set to mother repo).
- [ ] `python -m pip install -e ./worldloop-kernel -e ./worldloop-scenarios
      -e ./worldloop-data -e ./worldloop-adapters` → no errors.
- [ ] `pytest worldloop-kernel/tests worldloop-scenarios/tests
      worldloop-adapters/tests worldloop-data/tests -q` → all green (this
      is exactly what `.github/workflows/ci.yml` runs).

## 2. Docs clean

- [ ] No README link points at `current/`, `research/`, or other
      mother-repo paths.
- [ ] `docs/CLAIMS.md` only references files present in the release tree.
- [ ] Commercial segment in README uses the "MIT already allows commercial
      use; this channel covers paid integration support, technical
      support subscriptions, and custom development work" wording — NOT
      the contradictory "MIT + commercial licensing" phrasing.

## 3. Topics + Pages

Set these on the GitHub **Settings / About** panel:

- [ ] Description (short):
      `Environment-authoritative multi-agent simulation & trajectory-data system. Agents propose; a deterministic world adjudicates every consequence.`
- [ ] Homepage: `https://crqse.github.io/worldloop/` (after Pages deploy
      from the `examples/` static HTML below).
- [ ] Topics (comma-separated; copy-paste exactly):
      ```
      multi-agent-systems, llm-agents, agent-simulation, deterministic-simulation, counterfactual-reasoning, pettingzoo, gymnasium, synthetic-data, world-models
      ```

Pages deploy (lightweight, no build step needed):

- [ ] Repository **Settings → Pages → Build and deployment → Source** =
      `Deploy from a branch`.
- [ ] Branch = `main`, folder = `/ (root)`. Save.
- [ ] After the first deploy, confirm
      `https://crqse.github.io/worldloop/examples/emergency_demo.html`
      and `quickstart.html` load.

## 4. Create the Release (do this BEFORE flipping visibility)

- [ ] Run locally:
      ```bash
      git tag -a v0.1.3-beta.1 -m "WorldLoop v0.1.3-beta.1 — first public beta"
      git push origin v0.1.3-beta.1
      ```
- [ ] Then on GitHub: **Releases → Draft a new release**
  - Tag: `v0.1.3-beta.1`
  - Title: `WorldLoop v0.1.3-beta.1 — environment-authoritative public beta`
  - Release Notes (use the bullets in [RELEASE_NOTES_BETA1.md] OR the
    condensed copy below):
    > Agents propose, world adjudicates — first public beta of the
    > deterministic four-package layout. Highlights:
    > - 4 editable packages (kernel / scenarios / adapters / data), all
    >   green on Ubuntu+Windows × Python 3.10/3.12 via GitHub Actions.
    > - ScenarioSpec v0 YAML schema → compiler → validator →
    >   parametrized world execution.
    > - PettingZoo Parallel and Gymnasium adapters with tests green on
    >   the release layout.
    > - Trajectory exporter + leakage reporter + counterfactual brancher
    >   (no training-gain claim asserted).
    > - Interactive demo HTMLs shipped under `examples/` and deployed to
    >   GitHub Pages.
    > - Counterfactual & replay notebook demo.
  - [ ] Tick "Set as a pre-release".
  - [ ] "Publish release".

## 5. Flip visibility to Public

- [ ] Repository **Settings → Danger Zone → Change repository visibility →
      Make public →** confirm twice.
- [ ] Publish the announcement channels (see §6).

## 6. Announcement copy (minimal checklist)

- [ ] **30–60 s GIF / MP4** — re-record `examples/emergency_demo.html` or
      the counterfactual 2-panel Jupyter cell; attach it to the Release.
- [ ] **English article** (Post 1) titled *Why LLM agents should propose,
      not adjudicate*; post to your personal blog + relevant community
      (Reddit r/LocalLLaMA / r/MachineLearning, Hacker News Show HN,
      Hugging Face forum, PettingZoo/MARL Discords).
- [ ] **Chinese article(s)** — 知乎专栏 + 掘金 + v2ex 原创区，同步英文主线：
      「Agent/LLM 只负责提议，确定性世界负责裁决」
      + 最小 PettingZoo 重放 / 反事实示例 gif。
- [ ] **Minimum PettingZoo replay/counterfactual snippet** — paste the
      3-cell example from `quickstart.ipynb` into a self-contained
      GitHub Gist and link it from both announcements.

## 7. Phase-1 success metrics (not Stars)

Count these weekly for the first 4 weeks:

- [ ] N strangers successfully `pip install` + run `pytest …/tests -q`
      → target ≥ 10.
- [ ] N external bug reports / feature discussions opened (non-author)
      → target ≥ 3.
- [ ] N community-contributed scenario YAML files merged
      (pass `test_compiler.py`) → target ≥ 2.
- [ ] Stars only count after the three user-counts above are nonzero.
