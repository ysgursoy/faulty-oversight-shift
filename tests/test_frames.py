import unittest

from pydantic import ValidationError

from oversight_study.records.enums import Frame
from oversight_study.task.frames import FrameSpec, load_frame, parse_frame
from oversight_study.task.templates import load_template, placeholders


class FrameTests(unittest.TestCase):
    def test_all_frames_load(self) -> None:
        specs = {frame: load_frame(frame) for frame in Frame}
        self.assertEqual(specs[Frame.neutral].unit, "USD")
        self.assertEqual(specs[Frame.agriculture].weight_label, "Area (hectares)")
        self.assertIn("Rendaxin", specs[Frame.human].role)
        self.assertEqual(len({s.role for s in specs.values()}), 3)

    def test_fields_cover_template_placeholders(self) -> None:
        frame_fields = set(FrameSpec.model_fields)
        for wording in (1, 2):
            used = placeholders(load_template(f"system/wording_{wording}.txt")) - {"evaluator"}
            self.assertTrue(used <= frame_fields, used - frame_fields)

    def test_frames_differ_only_in_placeholder_values(self) -> None:
        template = load_template("system/wording_2.txt")
        rendered = {
            frame: template.format(evaluator="<E>", **load_frame(frame).model_dump())
            for frame in Frame
        }
        marker = template.format(evaluator="<E>", **{f: f"<{f}>" for f in FrameSpec.model_fields})
        for frame, text in rendered.items():
            restored = text
            for field, value in sorted(load_frame(frame).model_dump().items(),
                                       key=lambda kv: -len(kv[1])):
                restored = restored.replace(value, f"<{field}>")
            self.assertEqual(restored, marker, frame)

    def test_no_braces_in_values(self) -> None:
        for frame in Frame:
            for value in load_frame(frame).model_dump().values():
                self.assertNotIn("{", value)
                self.assertNotIn("}", value)

    def test_missing_or_extra_key_raises(self) -> None:
        text = load_template("frames/human.yaml")
        with self.assertRaises(ValidationError):
            parse_frame("\n".join(line for line in text.splitlines()
                                  if not line.startswith("unit:")))
        with self.assertRaises(ValidationError):
            parse_frame(text + "\nextra: x")
        with self.assertRaises(ValueError):
            load_frame("finance")


if __name__ == "__main__":
    unittest.main()
