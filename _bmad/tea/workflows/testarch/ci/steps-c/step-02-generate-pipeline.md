---
name: 'step-02-generate-pipeline'
description: 'Generate CI pipeline configuration'
nextStepFile: './step-03-configure-quality-gates.md'
pipelineOutputFile: '{project-root}/.github/workflows/test.yml'
outputFile: '{test_artifacts}/ci-pipeline-progress.md'
---

# Step 2: Generate CI Pipeline

## STEP GOAL

Create platform-specific CI configuration with test execution, sharding, burn-in, and artifacts.

## MANDATORY EXECUTION RULES

- 📖 Read the entire step file before acting
- ✅ Speak in `{communication_language}`

---

## EXECUTION PROTOCOLS:

- 🎯 Follow the MANDATORY SEQUENCE exactly
- 💾 Record outputs before proceeding
- 📖 Load the next step only when instructed

## CONTEXT BOUNDARIES:

- Available context: config, loaded artifacts, and knowledge fragments
- Focus: this step's goal only
- Limits: do not execute future steps
- Dependencies: prior steps' outputs (if any)

## MANDATORY SEQUENCE

**CRITICAL:** Follow this sequence exactly. Do not skip, reorder, or improvise.

## 1. Select Template

Choose template based on platform:

- GitHub Actions → `.github/workflows/test.yml`
- GitLab CI → `.gitlab-ci.yml`
- Circle CI → `.circleci/config.yml`
- Jenkins → `Jenkinsfile`

Use templates from `{installed_path}` when available (e.g., `github-actions-template.yaml`, `gitlab-ci-template.yaml`).

---

## 2. Pipeline Stages

Include stages:

- lint
- test (parallel shards)
- burn-in (flaky detection)
- report (aggregate + publish)

---

## 3. Test Execution

- Parallel sharding enabled
- CI retries configured
- Capture artifacts (HTML report, JUnit XML, traces/videos on failure)
- Cache dependencies (node_modules / pnpm / npm cache)

Write the selected pipeline configuration to `{pipelineOutputFile}` (or adjust the path if a non-GitHub platform was chosen).

---

### 4. Save Progress

**Save this step's accumulated work to `{outputFile}`.**

- **If `{outputFile}` does not exist** (first save), create it with YAML frontmatter:

  ```yaml
  ---
  stepsCompleted: ['step-02-generate-pipeline']
  lastStep: 'step-02-generate-pipeline'
  lastSaved: '{date}'
  ---
  ```

  Then write this step's output below the frontmatter.

- **If `{outputFile}` already exists**, update:
  - Add `'step-02-generate-pipeline'` to `stepsCompleted` array (only if not already present)
  - Set `lastStep: 'step-02-generate-pipeline'`
  - Set `lastSaved: '{date}'`
  - Append this step's output to the appropriate section of the document.

Load next step: `{nextStepFile}`

## 🚨 SYSTEM SUCCESS/FAILURE METRICS:

### ✅ SUCCESS:

- Step completed in full with required outputs

### ❌ SYSTEM FAILURE:

- Skipped sequence steps or missing outputs
  **Master Rule:** Skipping steps is FORBIDDEN.
