## Skill Catalog

The following Skills are metadata only. To use one, call the ordinary read_file Tool with its advertised path.
If a read result is paginated, continue with another read_file call when more instruction content is needed.
These autonomous read_file calls read the live file rather than the host's frozen Skill state. The model is not required to prove that it reached end of file.
Each following non-empty line is one Skill metadata JSON object with name, description, and path fields.

```jsonl
{entries}
```
