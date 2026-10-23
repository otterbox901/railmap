# Releasing

`.github/workflows/release.yml` runs on a `v*` tag. It checks that the tag matches `version` in `pyproject.toml`, runs
the tests, publishes to PyPI, pushes `ghcr.io/otterbox901/railmap:{version,major.minor,latest}` and creates a GitHub
release with the wheel and sdist.

One-time setup:

1. On PyPI, add a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
   for project `railmap`: owner `otterbox901`, repository `railmap`, workflow `release.yml`, environment `pypi`.
2. In the GitHub repo settings, create the environment `pypi` (optionally with required reviewers).
3. After the first image push, make the GHCR package public (Package settings → Change visibility).

Each release:

```bash
# bump version in pyproject.toml, then
uv lock && git commit -am "Release 0.1.0" && git tag v0.1.0 && git push origin master v0.1.0
```
