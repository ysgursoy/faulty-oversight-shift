import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from oversight_study.records.schema_export import export_schemas, schema_models
from oversight_study.scaffold import repository_root

COMMITTED = repository_root(Path(__file__)) / "schemas"


class SchemaDriftTests(unittest.TestCase):
    def test_committed_schemas_match_models(self) -> None:
        with TemporaryDirectory() as temp:
            for path in export_schemas(Path(temp)):
                committed = COMMITTED / path.name
                self.assertTrue(committed.exists(), f"missing {committed}; run "
                                "`uv run oversight-study schemas`")
                self.assertEqual(path.read_text(), committed.read_text(),
                                 f"{path.name} is stale; run `uv run oversight-study schemas`")

    def test_no_extra_schema_files(self) -> None:
        expected = {f"{name}.schema.json" for name in schema_models()}
        self.assertEqual({p.name for p in COMMITTED.glob("*.json")}, expected)

    def test_schemas_carry_categorical_values(self) -> None:
        turn = json.loads((COMMITTED / "turn_record.schema.json").read_text())
        text = json.dumps(turn)
        for value in ("near_measurement", "refusal", "development", "agriculture"):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
