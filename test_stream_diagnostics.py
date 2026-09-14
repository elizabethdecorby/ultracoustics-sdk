import json
import tempfile
import unittest
from pathlib import Path

from ultracoustics._internal.stream_proc import (
    PacketHeaderDiagnostics,
    classify_sequence_transition,
)


class SequenceClassificationTests(unittest.TestCase):
    def test_normal_stream(self):
        self.assertEqual(classify_sequence_transition(None, 10), ("first", 0, False))
        self.assertEqual(classify_sequence_transition(10, 11), ("next", 0, False))

    def test_small_and_large_gaps(self):
        self.assertEqual(classify_sequence_transition(10, 13), ("small_forward_gap", 2, False))
        self.assertEqual(classify_sequence_transition(10, 5012), ("large_forward_gap", 5001, False))

    def test_duplicate_wrap_and_backwards(self):
        self.assertEqual(classify_sequence_transition(10, 10), ("duplicate", 0, False))
        self.assertEqual(classify_sequence_transition(0xFFFFFFFF, 0), ("wrap", 0, False))
        self.assertEqual(classify_sequence_transition(0xFFFFFFFE, 1), ("wrap_gap", 2, False))
        self.assertEqual(classify_sequence_transition(100, 90), ("backwards_new_epoch", 0, True))

    def test_stale_first_header_starts_new_epoch_on_backwards_transition(self):
        first = classify_sequence_transition(None, 50000)
        second = classify_sequence_transition(50000, 25)
        self.assertEqual(first[0], "first")
        self.assertEqual(second, ("backwards_new_epoch", 0, True))


class RecorderTests(unittest.TestCase):
    def test_missing_truncated_and_bounded_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "headers.jsonl"
            recorder = PacketHeaderDiagnostics(
                str(path), attach_duration_s=0.001,
                max_attach_records=2, max_anomaly_records=1,
            )
            recorder.record(receive_monotonic_ns=1_000_000, length=8, seq=None,
                            drops_fw=None, classification="missing_or_truncated_frame")
            recorder.record(receive_monotonic_ns=1_000_100, length=16392, seq=4,
                            drops_fw=2, classification="first")
            recorder.record(receive_monotonic_ns=1_000_200, length=16392, seq=5,
                            drops_fw=2, classification="next")
            recorder.record(receive_monotonic_ns=3_000_000, length=16392, seq=5,
                            drops_fw=2, classification="duplicate")
            recorder.record(receive_monotonic_ns=4_000_000, length=16392, seq=3,
                            drops_fw=2, classification="backwards_new_epoch",
                            starts_new_epoch=True)
            recorder.write()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]["omitted_attach_records"], 1)
            self.assertEqual(rows[0]["omitted_anomaly_records"], 1)
            self.assertEqual(rows[1]["classification"], "missing_or_truncated_frame")
            self.assertEqual(rows[-1]["classification"], "duplicate")


if __name__ == "__main__":
    unittest.main()
