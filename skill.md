# AI Learning Skill

## Objective

This skill functions as an AI learning engine that explains AI concepts and progressively builds a knowledge graph.

---

## Directory Structure

The project structure is organized as follows:

skill.md
scripts/
reference/

Automation scripts are located in the **scripts** directory.

Reference files are located in the **reference** directory.

---

## Reference Files

The skill must read the following files.

reference/knowledge_map.md

reference/concept_template.md

reference/rules.md

Additional knowledge structures are located in:

reference/knowledge_map/

---

## Input

The system determines the next concept based on the learning order defined in:

reference/knowledge_map.md

---

## Execution Process

1 Read reference/knowledge_map.md
2 Determine the current learning position
3 Load the corresponding knowledge structure from reference/knowledge_map/
4 Generate learning content following reference/concept_template.md
5 Apply rules defined in reference/rules.md
6 Update the knowledge graph

---

## Learning Mode

Each execution generates AI learning content based on the knowledge structure,
expands the knowledge graph, and delivers the learning result.
