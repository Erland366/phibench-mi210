# Contributors build task images with Docker; Enroot is the fallback

Status: accepted (2026-11-19)

Φ-Bench builds its images with `docker buildx`, contributors most likely have Docker, and the merge into Φ-Bench
needs a Docker build anyway, so the procedure builds with Docker. Our Enroot builder (`enroot.py`), which follows
each Dockerfile step by step, stays documented for machines without Docker, such as our MI210 nodes.

## Consequences

- The procedure's build step has not yet been run with Docker by us; the worked example is where it is first
  tried (rootless Docker on node 303 is installed but untested).
