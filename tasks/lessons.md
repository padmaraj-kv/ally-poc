# Lessons

- When the user has already shared a requirements doc, scope = that doc. Don't ask them to re-pick features from it; state the scope and move on.
- ruff >= 0.17 formats Python code blocks inside Markdown: exclude plan/spec dirs (`extend-exclude`) or `ruff format --check .` fails on docs.
- Extracting code from Markdown: match closing fences only at line start; code can contain ``` inside strings.
- Don't assume inputs carry metadata the real source won't have (e.g. call language at upload). Ask how the data arrives in production before making it a required field.
- Code-switched calls can change language mid-call: any detection that looks only at the opening (first N seconds) is wrong for this domain; sample the whole call.
