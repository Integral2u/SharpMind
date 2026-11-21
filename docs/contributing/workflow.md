# Branch and release workflow

This is the shared workflow for SharpMind and SharpMind Studio. It is deliberately
local-first: commits are cheap, pushes are checkpoints, and `master` is the
release-grade line.

## Principles

- Work and commit locally. Push only when a push buys something.
- `master` is for released code and urgent fixes, not day-to-day work.
- A PR is optional. Use one when you need a pre-merge gate or a review record.
- Keep one logical change per commit and per branch.

## Branches

| Branch | Purpose |
| --- | --- |
| `master` | Released line. Release merges and urgent fixes only. |
| `release/x.y.z` | Integration branch for the current cycle (for example `release/1.1.3`). |
| `feat/<topic>` | Feature work, branched from the current `release/x.y.z`. |
| `fix/<topic>` | Bug fix for the current cycle, branched from `release/x.y.z`. |
| `docs/<topic>` | Documentation-only work, branched from the current `release/x.y.z`. |

- Keep work branches short-lived.
- Rebase a work branch on `release/x.y.z` before merging it back.
- Delete the work branch after it is merged.
- Never rewrite history that is already pushed and shared.

## Daily loop

```powershell
git switch release/x.y.z
git pull --ff-only
git switch -c feat/my-change

# edit, build, test, commit locally

git switch feat/my-change
git rebase release/x.y.z
git switch release/x.y.z
git merge --ff-only feat/my-change
```

If the branch is already pushed and shared, merge it instead of rebasing it.

## When to push

Push a work or release branch when:

- you want a backup or safe point;
- you are handing the work to someone else;
- you need a gate that cannot run locally (arm64, CodeQL, a clean machine);
- you are about to start a risky refactor.

```powershell
git push -u origin <branch>
```

Do not push `master` except for a release merge or an urgent fix.

## When to open a PR

No PR is required for normal local work. Open one only when:

- you want the full CI matrix and CodeQL to run before the change reaches `master`;
- you want another person to review the diff;
- branch protection requires it;
- the change comes from an external contributor.

PRs must target `master` for CodeQL to run.

## Gates

Run focused tests while working. Run the full build, test suite and encoding guard
before merging into `release/x.y.z` or `master`.

```powershell
dotnet build SharpMind.slnf -c Debug
dotnet test SharpMind.slnf -c Debug --no-build
pwsh ./scripts/check-encoding.ps1
```

Use `SharpMind.sln` only when the `wasm-tools-net10` workload is installed.
Use the Release configuration for performance-sensitive changes.

## Commit messages

Use `type(scope): imperative summary`, for example:

```text
fix(mla): validate slot indices
feat(server): add batched completions
docs(workflow): document the local-first release process
```

The body should explain why the change was needed, what changed, what was
verified, and any known limits. Do not mix a version bump with feature work.

## Versioning and changelog

- The version lives only in `Directory.Build.props`:
  `Version`, `AssemblyVersion` and `FileVersion`.
- Add release notes under `## [Unreleased]` in `CHANGELOG.md`.
- At release time, rename that section to `## [x.y.z.0] - YYYY-MM-DD` and add a
  fresh `## [Unreleased]`.
- Only the release owner bumps the version.

## Release

```powershell
git switch master
git pull --ff-only
git merge --no-ff release/x.y.z

# run the full gates on master

git push origin master
git tag vx.y.z.0
git push origin vx.y.z.0
```

Tag only when installers or packages are actually published. Merge to `master`
only when the cycle is complete or a fix must ship early.

## Urgent fix

Start from `master`, not from the cycle branch:

```powershell
git switch master
git switch -c fix/my-hotfix
# fix, verify, commit
git switch master
git merge --no-ff fix/my-hotfix
git push origin master
```

Then merge `master` back into `release/x.y.z`, or cherry-pick the fix, so the
current cycle does not diverge.

## Do not commit

- Model files and GGUF files. Keep them outside the repository, for example in
  `C:\Users\<you>\SharpMind\Models`.
- Datasets and benchmark artifacts.
- `bin/`, `obj/`, logs, local settings and secrets.
- Large binaries that are not part of a release.

## SharpMind Studio

SharpMind Studio follows the same branch, commit and push rules, with its own
version and changelog.

- If Studio consumes SharpMind packages, pin a published version.
- Bump the package pin only after SharpMind's version is final.
- Test against the local SharpMind build before changing the pin.
- Coordinate API changes: land the SharpMind side first, then update Studio.

## CI triggers

| Workflow | Runs on |
| --- | --- |
| `encoding` | Any push and any PR |
| `CI` (test matrix) | Push to `master`, any PR |
| `CodeQL` | Push to `master`, PR targeting `master`, weekly schedule |

A backup push of a work or release branch runs the encoding check only. If you
need the full matrix before merging, open a PR to `master` or run it locally.

## Handoff note

Include this in every handoff:

```text
Branch:
HEAD commit:
Changed:
Verified:
Not verified:
Next step:
```
