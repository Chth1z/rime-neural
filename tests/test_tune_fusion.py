import importlib.util
import json
import math
import pathlib
import tempfile
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TUNER_PATH = REPO_ROOT / "scripts" / "tune-fusion.py"


def load_tuner():
    spec = importlib.util.spec_from_file_location("tune_fusion", TUNER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tf = load_tuner()


class TestTuneFusion(unittest.TestCase):
    def test_parse_strict_int_rejects_bool_float_and_unicode_digits(self):
        self.assertEqual(tf._parse_strict_int(3), 3)
        self.assertEqual(tf._parse_strict_int(-2), -2)
        self.assertEqual(tf._parse_strict_int(" 42 "), 42)
        self.assertEqual(tf._parse_strict_int(5.0), 5)
        self.assertIsNone(tf._parse_strict_int(True))
        self.assertIsNone(tf._parse_strict_int(False))
        self.assertIsNone(tf._parse_strict_int(1.5))
        self.assertIsNone(tf._parse_strict_int(float("nan")))
        # Unicode superscript '²' returns True for str.isdigit() but fails int()
        self.assertIsNone(tf._parse_strict_int("\u00b2"))
        self.assertEqual(tf._parse_strict_int("\u00b2", -1), -1)

    def test_legacy_v1_respects_top1_sample_rate_before_and_after_scrub(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "feedback.jsonl"
            p.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "cands": ["甲", "乙"],
                                "logits": [0.0, 1.0],
                                "gram_deltas": [0.0, 0.2],
                                "char_lps": [0.0, 0.1],
                                "chosen_text": "乙",
                                "shown_top1": "乙",
                                "chosen_rank": 1,
                                "schema": "rime_frost",
                            }
                        ),
                        json.dumps(
                            {
                                "cands": ["甲", "乙"],
                                "logits": [0.0, 1.0],
                                "chosen_text": "甲",
                                "shown_top1": "乙",
                                "chosen_rank": 2,
                                "schema": "rime_frost",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            X, y, w, summary = tf.load_samples(p, top1_sample_rate=0.10)
            self.assertEqual(len(X), 2)
            self.assertEqual(summary["legacy_v1"], 2)
            self.assertAlmostEqual(float(w[0]), 10.0, places=6)
            self.assertAlmostEqual(float(w[1]), 1.0, places=6)

            # Scrubbing preserves absent sample_weight so --top1-sample-rate continues to work
            conv, kept = tf.scrub_legacy_log_files(p)
            self.assertEqual((conv, kept), (2, 0))
            raw_after = p.read_text(encoding="utf-8")
            self.assertNotIn('"chosen_text"', raw_after)
            self.assertNotIn('"cands"', raw_after)
            X2, y2, w2, s2 = tf.load_samples(p, top1_sample_rate=0.10)
            self.assertEqual(len(X2), 2)
            self.assertEqual(s2["legacy_v1"], 0)
            self.assertAlmostEqual(float(w2[0]), 10.0, places=6)
            self.assertAlmostEqual(float(w2[1]), 1.0, places=6)

    def test_load_samples_filters_and_scrubs_rotated_and_hybrid_records(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "feedback.jsonl"
            p1 = pathlib.Path(str(p) + ".1")
            p1.write_text(
                json.dumps(
                    {
                        "v": 2,
                        "schema": "rime_ice",
                        "model": "laya.one.w4.onnx",
                        "has_gram": False,
                        "has_ctx": False,
                        "logits": [1.0, 2.0],
                        "gram_deltas": [0.0, 0.0],
                        "char_lps": [0.0, 0.0],
                        "shown_top1_idx": 1,
                        "chosen_idx": 1,
                        "raw_input": "stray_plaintext",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            p.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "v": 2,
                                "schema": "rime_frost",
                                "model": "laya.one.fused.int8g.onnx",
                                "has_gram": True,
                                "has_ctx": True,
                                "logits": [1.0, 2.0],
                                "gram_deltas": [0.0, 0.3],
                                "char_lps": [0.0, 0.1],
                                "shown_top1_idx": 1,
                                "chosen_idx": 1,
                            }
                        ),
                        # Invalid boolean chosen_idx must be skipped cleanly
                        json.dumps({"v": 2, "logits": [1.0, 2.0], "chosen_idx": True, "shown_top1_idx": False}),
                        # Invalid Unicode digit chosen_idx must be skipped cleanly
                        json.dumps({"v": 2, "logits": [1.0, 2.0], "chosen_idx": "\u00b2", "shown_top1_idx": 0}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            X_all, _, _, s_all = tf.load_samples(p)
            self.assertEqual(len(X_all), 2)
            self.assertEqual(s_all["has_gram"][True], 1)
            self.assertEqual(s_all["has_gram"][False], 1)

            X_gram, _, _, _ = tf.load_samples(p, has_gram_filter=True)
            self.assertEqual(len(X_gram), 1)
            X_schema, _, _, _ = tf.load_samples(p, schema_filter="rime_ice", model_filter="laya.one.w4.onnx")
            self.assertEqual(len(X_schema), 1)

            conv, kept = tf.scrub_legacy_log_files(p, include_rotated=True)
            self.assertEqual(conv, 1)
            self.assertGreaterEqual(kept, 1)
            self.assertNotIn("stray_plaintext", p1.read_text(encoding="utf-8"))

    def test_newton_solver_is_weight_scale_invariant_and_freezes_collinear_ctx_lp(self):
        rng = np.random.default_rng(7)
        X_list = []
        y_list = []
        for _ in range(80):
            lp = tf.log_softmax(rng.normal(0.0, 1.5, size=4).tolist())
            X = np.array(
                [[lp[i], math.log(i + 1.0), rng.normal(0, 0.5), rng.normal(0, 0.5), lp[i]] for i in range(4)],
                dtype=np.float64,
            )
            # True weights favor higher logit weight (2.0) with rank prior (-1.0)
            scores = X @ np.array([2.0, -1.0, 0.2, 0.2, 0.0])
            y = int(np.argmax(scores))
            X_list.append(X)
            y_list.append(y)

        w_unit = np.ones(len(X_list), dtype=np.float64)
        w_scaled = w_unit * 10.0

        fit1 = tf.fit_newton_offset_ridge(X_list, y_list, w_unit, lam=15.0, freeze_ctx_lp=True)
        fit10 = tf.fit_newton_offset_ridge(X_list, y_list, w_scaled, lam=15.0, freeze_ctx_lp=True)
        np.testing.assert_allclose(fit1, fit10, atol=1e-9)
        self.assertEqual(float(fit1[4]), float(tf.W_PRIOR[4]))
        self.assertGreater(float(fit1[0]), float(tf.W_PRIOR[0]))

    def test_domain_sanity_z_score_and_neg_zero_formatting(self):
        self.assertTrue(tf.weights_pass_domain_sanity(tf.W_PRIOR)[0])
        self.assertFalse(tf.weights_pass_domain_sanity(np.array([1.0, 0.5, 0.0, 0.0, 0.0]))[0])
        self.assertFalse(tf.weights_pass_domain_sanity(np.array([0.05, -1.0, 0.0, 0.0, 0.0]))[0])
        self.assertFalse(tf.weights_pass_domain_sanity(np.array([0.5, -1.0, 0.0, 0.0, -0.45]))[0])

        self.assertTrue(math.isinf(tf.paired_z_score(0.05, 0.0)))
        self.assertEqual(tf.paired_z_score(0.0, 0.0), 0.0)
        fmt = tf.format_weights(np.array([1.0, -1.4, 0.05, 0.15, -0.0001]))
        self.assertEqual(str(fmt[4]), "0.0")


if __name__ == "__main__":
    unittest.main()
