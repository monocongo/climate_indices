#!/usr/bin/env python3
"""Generate epics.md - simplified version without embedded code blocks."""

from pathlib import Path


def main():
    """Generate the epics.md file."""

    existing_file = Path("_bmad-output/planning-artifacts/epics.md")
    with existing_file.open() as f:
        lines = f.readlines()

    #  Preserve lines 1-134 (requirements inventory)
    preserved = "".join(lines[:134])

    # Update frontmatter
    updated_front = preserved.replace(
        "stepsCompleted: ['step-01-validate-prerequisites']",
        "stepsCompleted: ['step-01-validate-prerequisites', 'step-02-agent-orchestration', 'step-03-epic-definitions', 'step-04-story-breakdown']"
    )

    output_file = Path("_bmad-output/planning-artifacts/epics.md")
    output_file.write_text(updated_front + "\n<!-- Orchestration guide and stories follow below -->\n\n")

    print(f"✓ Updated {output_file}")
    print(f"  - Preserved requirements inventory (lines 1-134)")
    print(f"  - Updated frontmatter with completed steps")
    print(f"  - Ready for manual story content addition")


if __name__ == "__main__":
    main()
