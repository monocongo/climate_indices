---
name: 'step-01-preflight'
description: 'Verify prerequisites and gather project context'
nextStepFile: './step-02-select-framework.md'
outputFile: '{test_artifacts}/framework-setup-progress.md'
---

# Step 1: Preflight Checks

## STEP GOAL

Verify the project is ready for framework scaffolding and gather key context.

## MANDATORY EXECUTION RULES

- 📖 Read the entire step file before acting
- ✅ Speak in `{communication_language}`
- 🚫 Halt if preflight requirements fail

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

## 1. Validate Prerequisites

- `package.json` exists in project root
- No existing E2E framework (`playwright.config.*`, `cypress.config.*`, `cypress.json`)
- Architecture/stack context available (project type, bundler, dependencies)

If any fail, **HALT** and report the missing requirement.

---

## 2. Gather Project Context

- Read `package.json` to identify framework, bundler, dependencies
- Check for architecture docs (`architecture.md`, `tech-spec*.md`) if available
- Note auth requirements and APIs (if documented)

---

## 3. Confirm Findings

Summarize:

- Project type and bundler
- Whether a framework is already installed
- Any relevant context docs found

---

### 4. Save Progress

**Save this step's accumulated work to `{outputFile}`.**

- **If `{outputFile}` does not exist** (first save), create it with YAML frontmatter:

  ```yaml
  ---
  stepsCompleted: ['step-01-preflight']
  lastStep: 'step-01-preflight'
  lastSaved: '{date}'
  ---
  ```

  Then write this step's output below the frontmatter.

- **If `{outputFile}` already exists**, update:
  - Add `'step-01-preflight'` to `stepsCompleted` array (only if not already present)
  - Set `lastStep: 'step-01-preflight'`
  - Set `lastSaved: '{date}'`
  - Append this step's output to the appropriate section of the document.

Load next step: `{nextStepFile}`

## 🚨 SYSTEM SUCCESS/FAILURE METRICS:

### ✅ SUCCESS:

- Step completed in full with required outputs

### ❌ SYSTEM FAILURE:

- Skipped sequence steps or missing outputs
  **Master Rule:** Skipping steps is FORBIDDEN.
