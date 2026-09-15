# Post-move actions — GPULlama3.java → jllm

Everything in this file happens **outside this repository**. The rename PR cannot do any of it;
each item needs a person with credentials on another system.

This is a working checklist. Tick the boxes as they land, then delete the file.

> **Naming recap.** `io.github.beehive-lab` (groupId) is unchanged. `gpu-llama3` → `jllm`
> (artifactId). `org.beehive.gpullama3` → `org.beehive.jllm` (Java package).
> `beehive-lab/GPULlama3.java` → `beehive-lab/jllm` (repo). `llama-tornado` → `jllm`,
> `llamaTornado` → `jllm4j` (launchers). This is a **clean break**: no relocation POM, no aliases.

---

## 1. Blocking — before or with the merge

| ✓ | Item | System | Action | Owner |
|---|---|---|---|---|
| [ ] | quarkus-langchain4j fork branch | GitHub `orionpapadakis/quarkus-langchain4j` | Branch `jllm/facade-1.0.0` off `gpu-llama3/facade-1.0.0`, change the dependency artifactId to `jllm`. `build-and-run.yml` already points at the new ref, so **the Quarkus leg is red until this exists**. | |
| [ ] | Fixture cache on self-hosted runners | each runner host | `ln -s ~/.gpullama3 ~/.jllm` — **symlink, not move**. A move turns every pre-rename branch's golden tests into silent skips too. | |
| [ ] | Record required status checks | GitHub repo settings | `gh api repos/beehive-lab/GPULlama3.java/rulesets`, or branch protection `required_status_checks.contexts`. Write the exact strings down **before** renaming anything. | |
| [ ] | Rename the repo | GitHub repo settings | `beehive-lab/GPULlama3.java` → `beehive-lab/jllm`. **Do this before merging — it is a prerequisite, not a preference.** The 13 slug guards now read `beehive-lab/jllm`, so on the old slug every guarded job evaluates false and *skips*, and skipped jobs report success: the PR goes green with no build, no spotless, no tests and no inference run. | |
| [ ] | Re-point required status checks | GitHub repo settings | From the list recorded above. A context stored as `GPULlama3 Build & Run / build-linux` can never be satisfied again and **blocks every future PR on a check that never reports**. Highest-risk item here. | |
| [ ] | Confirm CI actually ran | GitHub Actions | After merge: `gh run view --json jobs -q '.jobs[]\|select(.conclusion=="skipped").name'` must be empty. Skipped jobs report success. | |

## 2. Maven Central

| ✓ | Item | Action |
|---|---|---|
| [x] | Namespace verification | **No action needed.** Verification is per-groupId, and `io.github.beehive-lab` is already verified. A new artifactId under it publishes with no new registration. Do not open a Sonatype ticket. |
| [x] | Old coordinate | `io.github.beehive-lab:gpu-llama3` is immutable and stays resolvable forever. Nothing to deprecate, nothing to delete. |
| [ ] | First `jllm` release | Dry-run `prepare-release` on a fork. Confirm the regenerated `README.md` DEPENDENCY-SNIPPETS block and the `CHANGELOG.md` header emit `jllm`, and that `deploy-maven-central` actually fires (it triggers on the literal workflow name `Finalize jllm Release`). |
| [ ] | Automatic module name | Now pinned to `org.beehive.jllm` in the shade manifest. Previously derived from the jar filename (`gpu.llama3`); any downstream `requires gpu.llama3;` breaks. |
| [ ] | Forwarding address *(optional, currently declined)* | A terminal `gpu-llama3` release carrying only a `<relocation>` stanza would turn downstream `ClassNotFoundException` into a Maven warning naming the new coordinates. Recorded here in case the clean-break call is revisited. |

## 3. Downstream integrations — each needs an upstream PR we cannot make

| ✓ | Integration | What points at us | Action |
|---|---|---|---|
| [ ] | **LangChain4j** (`dev.langchain4j`) | module `langchain4j-gpu-llama3`; dep `io.github.beehive-lab:gpu-llama3`; docs page `docs.langchain4j.dev/integrations/language-models/gpullama3-java` | Upstream PR bumping the dependency to `io.github.beehive-lab:jllm`. Whether their *module* also renames is their call — it is a breaking change for their users, so propose **dependency-only** first. |
| [ ] | **Quarkus LangChain4j** (Quarkiverse) | `model-providers/gpu-llama3/{runtime,deployment}`; `<gpu-llama3.version>`; config keys `quarkus.langchain4j.gpu-llama3.*`; classes `GPULlama3ChatModel`, `GPULlama3StreamingChatModel`, `GPULlama3BaseModel`, `GPULlama3ResponseParser`; docs `docs.quarkiverse.io/.../gpullama3-chat-model.html` | Upstream PR bumping the dependency. **Their config keys are public API** — renaming `quarkus.langchain4j.gpu-llama3.*` breaks every `application.properties` in the wild. Dependency-only first; let them sequence any key rename with a deprecation cycle. |
| [ ] | Our own update skills | `.claude/skills/update-{langchain4j,quarkus-langchain4j}-integration` | `inspect-release.sh` now resolves `artifact=jllm` and javaps `org.beehive.jllm.model.Model`. **Neither exists in any published jar until the first `jllm` release** — expect these scripts to fail until then. |

## 4. Other beehive-lab-owned properties

| ✓ | Property | Action |
|---|---|---|
| [ ] | `beehive-lab/docker-gpullama3.java` | Follow-up PR **after** the repo rename (it clones this repo and runs `./llama-tornado`). Update the clone URL and the launcher invocation to `./jllm`. Image names `beehivelab/gpullama3.java-*` stay per the clean-break decision; note the in-image path `/gpullama3/GPULlama3.java/llama-tornado` changes when that repo rebuilds. |
| [ ] | jbang catalog (`beehive-lab/jbang-catalog`) | Alias `gpullama3@beehive-lab` rides the GitHub redirect. Verify with `jbang gpullama3@beehive-lab --help`; patch `script-ref` if it 404s. Decide separately whether to add a `jllm@beehive-lab` alias. |
| [ ] | Hugging Face collections (8, `*-gpullama3java`) | Slugs stay — renaming 404s the old ones. Update descriptions and repo links only. |
| [ ] | Docker Hub (`beehivelab/gpullama3.java-*`) | Tags are immutable. Decide whether new tags go to a new `beehivelab/jllm` repo; if so, add a deprecation note on the old repo's description. |
| [ ] | CLA-assistant | Registered against `beehive-lab/llama3.java-tornadovm` — **already wrong from the previous rename** (`CONTRIBUTING.md`). Re-register against `beehive-lab/jllm`. |
| [ ] | Repo metadata | Description, topics, social preview. |

## 5. Third-party indexes and passive references

| ✓ | Item | Action |
|---|---|---|
| [ ] | DeepWiki (`deepwiki.com/beehive-lab/GPULlama3.java`) | Auto-generated and keyed on the slug. Re-index; the badge in `README.md` already points at the new slug. |
| [ ] | Citation | `CITATION.cff` now reads `jllm (formerly GPULlama3.java)` so existing citations stay findable. There is **no DOI**, so nothing else links them. Consider minting a Zenodo DOI at the first `jllm` release, which makes the *next* rename a non-event. |
| [ ] | `docs/performance.png` | "GPULlama3.java" is rendered into the pixels. Currently orphaned (nothing references it) — re-render or delete. |
| [ ] | `docs/ll.gif` (32 MB, README hero) | A terminal recording showing `./llama-tornado`. Re-record, or drop the embed — a hero GIF demoing a command that no longer exists is worse than no GIF. |
| — | Upstream lineage links | `mukel/llama3.java`, `mikepapadim/llama2.tornadovm.java`, `mikepapadim/devoxx25-demo-gpullama3-langchain4j` — third-party, informational only. |
| — | Talks, blog posts, slides | Anything published referencing `llama-tornado` now names a command that does not exist. No action possible; listed so it is not a surprise. |

---

## Do not do

- **Never create a repo at the old slug.** GitHub's redirect from `beehive-lab/GPULlama3.java`
  survives indefinitely — until someone claims that name, at which point every old link,
  clone URL and issue reference breaks permanently.
- **Never rewrite `docs/perf-history.jsonl` or `perf-results/**`.** They are append-only
  historical records. The 1614 existing rows carry `"workflow": "GPULlama3 Build & Run"`
  because that is the workflow that produced them; the perf gate keys on a tuple that
  excludes `workflow`, so the old value is harmless. New rows pick up the new name on their own.
  `perf-results/baseline-rtx5090-tvm520-20260803/README.md` contains a captured stack trace
  with old package frames — that is evidence, not a stale reference.
- **Never rename the `beehive-lab` org.** It also owns TornadoVM and backs the verified
  Maven Central namespace `io.github.beehive-lab`.
- **Never `mv ~/.gpullama3 ~/.jllm` on a runner** — symlink it, so pre-rename branches keep working.
