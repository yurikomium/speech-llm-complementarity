"""Offline unit tests for the released core utilities."""

import ast
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "prompts"))

from evaluation_metrics import (  # noqa: E402
    ccc,
    pairwise_accuracy,
    pearson_r,
    top1_accuracy,
)
from llm_jsonl_to_predictions import (  # noqa: E402
    DIRECTION_TO_TARGET,
    convert_jsonl,
)
from main_single_modal import (  # noqa: E402
    _OLD_PT_RE,
    SPEAKER_CONFIG,
    build_emb_index,
    load_samples,
    run_inference_single,
    validate_participant_disjoint_splits,
)
from nway_fusion import (  # noqa: E402
    _build_selection_mask,
    _load_prediction_sources,
    _merge_prediction_sources,
    _per_participant_ccc,
)
from _api_response import (  # noqa: E402
    ApiResponseError,
    serialize_api_response,
    write_attempt_error,
)
from convert_csv_to_pt import convert as convert_feature_csv  # noqa: E402
from training import (  # noqa: E402
    SingleModalModel,
    _ccc_loss_from_scores,
    fit_single,
    train_one_epoch_single,
)


class ReleasedModelScopeTests(unittest.TestCase):
    def test_training_module_contains_only_the_reported_model_path(self):
        training_tree = ast.parse(
            (ROOT / "src" / "training.py").read_text(encoding="utf-8")
        )
        class_names = {
            node.name
            for node in training_tree.body
            if isinstance(node, ast.ClassDef)
        }
        self.assertEqual(
            class_names,
            {"AdditiveAttentionPooling", "PredictionHead", "SingleModalModel"},
        )
        training_source = (
            ROOT / "src" / "training.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("_make_" + "sample_batches", training_source)
        self.assertNotIn("loss_fn == " + '"mse"', training_source)
        self.assertNotIn("F." + "mse_loss", training_source)

        main_source = (ROOT / "src" / "main_single_modal.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("partner" + "-mode", main_source)
        self.assertNotIn("ccc" + "_global", main_source)

        evaluation_source = (
            ROOT / "src" / "evaluation_metrics.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"' + "global" + '"', evaluation_source)

        gpt_audio_source = (
            ROOT / "prompts" / "predict_liking_gpt_audio.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("SYSTEM_PROMPT_" + "T =", gpt_audio_source)
        self.assertNotIn("USER_PROMPT_" + "T =", gpt_audio_source)
        self.assertNotIn("--" + "modality", gpt_audio_source)
        self.assertNotIn("modality == " + '"T"', gpt_audio_source)

        release_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in [
                ROOT / "README.md",
                ROOT / "src" / "convert_csv_to_pt.py",
                ROOT / "scripts" / "run_modality_comparison.sh",
            ]
        ).lower()
        self.assertNotIn("open" + "face", release_text)
        self.assertNotIn("ridge", (ROOT / "src" / "nway_fusion.py").read_text(
            encoding="utf-8"
        ).lower())

    def test_unreported_mse_training_is_rejected(self):
        model = SingleModalModel(4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with self.assertRaisesRegex(ValueError, "CCC only"):
            train_one_epoch_single(
                model,
                train_samples=[],
                optimizer=optimizer,
                device=torch.device("cpu"),
                loss_fn="mse",
            )

    def test_ccc_loss_rejects_degenerate_batches(self):
        with self.assertRaisesRegex(ValueError, "two or more samples"):
            _ccc_loss_from_scores(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                ["F900"],
            )

    def test_self_only_model_trains_on_synthetic_samples(self):
        torch.manual_seed(0)
        samples = []
        for index, (female_id, target) in enumerate([
            ("F900", -1.0),
            ("F900", 1.0),
            ("F901", -0.5),
            ("F901", 0.5),
        ]):
            samples.append({
                "speaker": "female",
                "female_id": female_id,
                "male_id": f"M{900 + index}",
                "embeddings": torch.tensor([
                    [float(index), 0.0],
                    [float(index) + 0.5, 1.0],
                ]),
                "_target_scaled": target,
            })

        model = SingleModalModel(d=2, dropout=0.0)
        fitted = fit_single(
            model,
            samples,
            samples,
            torch.device("cpu"),
            loss_fn="ccc",
            max_epochs=1,
            min_epochs=1,
            patience=1,
            batch_size=2,
            verbose=False,
        )
        embeddings = samples[0]["embeddings"].unsqueeze(0)
        mask = torch.ones(1, embeddings.shape[1], dtype=torch.bool)
        prediction = fitted(embeddings, mask)
        self.assertTrue(torch.isfinite(prediction).all())


class FilenameParsingTests(unittest.TestCase):
    def test_old_pt_fields_keep_kaisu_session_and_conv_distinct(self):
        match = _OLD_PT_RE.match("12_2_3_A_F900_M900_female.pt")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "12")  # kaisu
        self.assertEqual(match.group(2), "2")   # session
        self.assertEqual(match.group(3), "3")   # conv
        self.assertEqual(match.group(4), "A")
        self.assertEqual(match.group(5), "F900")
        self.assertEqual(match.group(6), "M900")
        self.assertEqual(match.group(7), "female")

    def test_session_filter_uses_second_field_and_kaisu_uses_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            emb_dir = Path(tmp)
            payload = {
                "pair_key": "F900__M900",
                "female_id": "F900",
                "male_id": "M900",
                "speaker": "female",
                "fold_group": 1,
                "session": 2,
                "embeddings": torch.zeros(2, 3),
                "like_score": 5.0,
            }
            torch.save(
                payload,
                emb_dir / "12_2_1_A_F900_M900_female.pt",
            )
            payload_sess1 = {**payload, "session": 1}
            torch.save(
                payload_sess1,
                emb_dir / "12_1_2_A_F900_M900_female.pt",
            )

            index = build_emb_index(emb_dir, session=2)
            self.assertEqual(len(index["F900__M900"]), 1)
            samples = load_samples(
                ["F900__M900"],
                index,
                target="like_score",
                speaker="female",
            )
            self.assertEqual(samples[0]["_kaisu"], 12)


class DirectionSemanticsTests(unittest.TestCase):
    def test_supervised_speaker_mapping_matches_paper_directions(self):
        self.assertEqual(
            SPEAKER_CONFIG["female"],
            {
                "direction": "F2M",
                "id_col": "female_id",
                "partner_col": "male_id",
            },
        )
        self.assertEqual(
            SPEAKER_CONFIG["male"],
            {
                "direction": "M2F",
                "id_col": "male_id",
                "partner_col": "female_id",
            },
        )

    def test_llm_direction_targets_match_rater_to_partner_semantics(self):
        self.assertEqual(
            DIRECTION_TO_TARGET["f_to_m"],
            "like_F_to_M_sess{session}",
        )
        self.assertEqual(
            DIRECTION_TO_TARGET["m_to_f"],
            "like_M_to_F_sess{session}",
        )

    def test_all_prompt_direction_configs_use_the_same_roles(self):
        prompt_dir = ROOT / "prompts"
        prompt_files = sorted(prompt_dir.glob("predict_liking*.py"))
        self.assertEqual(len(prompt_files), 6)
        for path in prompt_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            config = None
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "DIRECTION_CONFIG"
                        for target in node.targets
                    )
                ):
                    config = ast.literal_eval(node.value)
                    break
            self.assertIsNotNone(config, path.name)
            self.assertEqual(config["f_to_m"]["evaluator_role"], "女性")
            self.assertEqual(config["f_to_m"]["target_role"], "男性")
            self.assertEqual(config["m_to_f"]["evaluator_role"], "男性")
            self.assertEqual(config["m_to_f"]["target_role"], "女性")
            if "evaluator_prefix" in config["f_to_m"]:
                self.assertEqual(config["f_to_m"]["evaluator_prefix"], "F")
                self.assertEqual(config["m_to_f"]["evaluator_prefix"], "M")
            else:
                self.assertIn("like_F_to_M", config["f_to_m"]["liking_col"])
                self.assertIn("like_M_to_F", config["m_to_f"]["liking_col"])


class SplitValidationTests(unittest.TestCase):
    def test_participant_disjoint_splits_are_accepted(self):
        validate_participant_disjoint_splits(
            ["F900__M900", "F901__M901"],
            ["F902__M902"],
            ["F903__M903"],
        )

    def test_participant_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Participant overlap"):
            validate_participant_disjoint_splits(
                ["F900__M900"],
                ["F900__M901"],
                ["F902__M902"],
            )

    def test_pair_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Pair overlap"):
            validate_participant_disjoint_splits(
                ["F900__M900"],
                ["F901__M901"],
                ["F900__M900"],
            )


class RankingMetricTests(unittest.TestCase):
    def test_pearson_and_ccc_match_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_same = np.array([1.0, 2.0, 3.0])
        y_shifted = np.array([2.0, 3.0, 4.0])
        r, _ = pearson_r(y_true, y_same)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(ccc(y_true, y_same), 1.0)
        self.assertAlmostEqual(ccc(y_true, y_shifted), 4.0 / 7.0)

    def test_top1_gives_fractional_credit_for_predicted_tie(self):
        frame = pd.DataFrame(
            {
                "rater": ["R1"] * 3,
                "partner": ["P1", "P2", "P3"],
                "y_true": [9.0, 8.0, 7.0],
                "y_pred": [5.0, 4.0, 5.0],
            }
        )
        accuracy, n_participants = top1_accuracy(
            frame, "rater", "partner"
        )
        self.assertEqual(n_participants, 1)
        self.assertAlmostEqual(accuracy, 0.5)

    def test_pairwise_is_participant_macro_and_excludes_true_ties(self):
        frame = pd.DataFrame(
            {
                "rater": ["R1"] * 3 + ["R2"] * 2,
                "partner": ["P1", "P2", "P3", "P4", "P5"],
                "y_true": [9.0, 9.0, 1.0, 1.0, 9.0],
                "y_pred": [8.0, 7.0, 1.0, 9.0, 1.0],
            }
        )
        accuracy, n_participants, n_pairs = pairwise_accuracy(
            frame, "rater", "partner"
        )
        # R1: 2/2 after excluding the tied ground-truth pair; R2: 0/1.
        self.assertEqual(n_participants, 2)
        self.assertEqual(n_pairs, 3)
        self.assertAlmostEqual(accuracy, 0.5)


class LlmConversionTests(unittest.TestCase):
    @staticmethod
    def _record(pair_key: str, session: int = 2) -> dict:
        return {
            "pair_key": pair_key,
            "direction": "f_to_m",
            "session": session,
            "scores": {f"Q{i}": 5 for i in range(1, 14)},
        }

    def test_thirteen_item_scores_are_averaged_on_one_to_nine_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_csv = tmp_path / "pairs.csv"
            input_jsonl = tmp_path / "outputs.jsonl"
            output_csv = tmp_path / "predictions.csv"

            pd.DataFrame(
                [{
                    "pair_key": "F900__M900",
                    "female_id": "F900",
                    "male_id": "M900",
                    "fold_group": 7,
                    "kaisu": 12,
                    "like_F_to_M_sess2": 65.0,
                    "exclude_reason": "",
                }]
            ).to_csv(pairs_csv, index=False)
            record = {
                "pair_key": "F900__M900",
                "direction": "f_to_m",
                "scores": {f"Q{i}": 5 for i in range(1, 14)},
            }
            input_jsonl.write_text(json.dumps(record) + "\n")

            output = convert_jsonl(
                input_jsonl,
                pairs_csv,
                output_csv,
                direction="f_to_m",
                session=2,
            )
            self.assertAlmostEqual(output.loc[0, "y_true"], 5.0)
            self.assertAlmostEqual(output.loc[0, "y_pred"], 5.0)
            self.assertEqual(output.loc[0, "kaisu"], 12)

    def test_constant_rater_exclusion_is_scoped_to_fold_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_csv = tmp_path / "pairs.csv"
            input_jsonl = tmp_path / "outputs.jsonl"
            output_csv = tmp_path / "predictions.csv"
            pairs = pd.DataFrame([
                {
                    "pair_key": "F900__M900", "female_id": "F900",
                    "male_id": "M900", "fold_group": 1, "kaisu": 1,
                    "like_F_to_M_sess2": 65.0, "exclude_reason": "",
                },
                {
                    "pair_key": "F900__M901", "female_id": "F900",
                    "male_id": "M901", "fold_group": 1, "kaisu": 1,
                    "like_F_to_M_sess2": 65.0, "exclude_reason": "",
                },
                {
                    "pair_key": "F900__M902", "female_id": "F900",
                    "male_id": "M902", "fold_group": 2, "kaisu": 2,
                    "like_F_to_M_sess2": 52.0, "exclude_reason": "",
                },
                {
                    "pair_key": "F900__M903", "female_id": "F900",
                    "male_id": "M903", "fold_group": 2, "kaisu": 2,
                    "like_F_to_M_sess2": 78.0, "exclude_reason": "",
                },
            ])
            pairs.to_csv(pairs_csv, index=False)
            input_jsonl.write_text(
                "".join(
                    json.dumps(self._record(pair_key)) + "\n"
                    for pair_key in pairs["pair_key"]
                ),
                encoding="utf-8",
            )

            output = convert_jsonl(
                input_jsonl, pairs_csv, output_csv,
                direction="f_to_m", session=2,
            )
            self.assertEqual(
                output["pair_key"].tolist(),
                ["F900__M902", "F900__M903"],
            )

    def test_record_session_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_csv = tmp_path / "pairs.csv"
            input_jsonl = tmp_path / "outputs.jsonl"
            pd.DataFrame([{
                "pair_key": "F900__M900", "female_id": "F900",
                "male_id": "M900", "fold_group": 1, "kaisu": 1,
                "like_F_to_M_sess2": 65.0, "exclude_reason": "",
            }]).to_csv(pairs_csv, index=False)
            input_jsonl.write_text(
                json.dumps(self._record("F900__M900", session=1)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match --session"):
                convert_jsonl(
                    input_jsonl, pairs_csv, tmp_path / "out.csv",
                    direction="f_to_m", session=2,
                )

    def test_missing_canonical_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs_csv = tmp_path / "pairs.csv"
            input_jsonl = tmp_path / "outputs.jsonl"
            pd.DataFrame([
                {
                    "pair_key": "F900__M900", "female_id": "F900",
                    "male_id": "M900", "fold_group": 1, "kaisu": 1,
                    "like_F_to_M_sess2": 52.0, "exclude_reason": "",
                },
                {
                    "pair_key": "F900__M901", "female_id": "F900",
                    "male_id": "M901", "fold_group": 1, "kaisu": 1,
                    "like_F_to_M_sess2": 78.0, "exclude_reason": "",
                },
            ]).to_csv(pairs_csv, index=False)
            input_jsonl.write_text(
                json.dumps(self._record("F900__M900")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing 1 canonical pairs"):
                convert_jsonl(
                    input_jsonl, pairs_csv, tmp_path / "out.csv",
                    direction="f_to_m", session=2,
                )


class FeatureConversionPathTests(unittest.TestCase):
    def test_explicit_input_csv_does_not_depend_on_staging_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_csv = tmp_path / "features-at-an-arbitrary-path.csv"
            pairs_csv = tmp_path / "fold-assignments.csv"
            output_dir = tmp_path / "converted"

            pd.DataFrame([
                {
                    "pair_key": "F900__M900", "speaker": "female",
                    "minute_idx": 1, "utt_id": 0,
                    "emb_0": 0.1, "emb_1": 0.2,
                },
                {
                    "pair_key": "F900__M900", "speaker": "male",
                    "minute_idx": 1, "utt_id": 1,
                    "emb_0": 0.3, "emb_1": 0.4,
                },
            ]).to_csv(feature_csv, index=False)
            pd.DataFrame([{
                "pair_key": "F900__M900",
                "exclude_reason": "",
                "fold_group": 7,
                "kaisu": 12,
                "female_id": "F900",
                "male_id": "M900",
                "like_F_to_M_sess2": 65.0,
                "love_F_to_M_sess2": 52.0,
                "like_M_to_F_sess2": 39.0,
                "love_M_to_F_sess2": 26.0,
            }]).to_csv(pairs_csv, index=False)

            convert_feature_csv(
                "bert", 2, output_dir, pairs_csv,
                input_csv=feature_csv,
            )

            female_path = output_dir / "F900__M900_female.pt"
            male_path = output_dir / "F900__M900_male.pt"
            self.assertTrue(female_path.is_file())
            self.assertTrue(male_path.is_file())
            female = torch.load(
                female_path, map_location="cpu", weights_only=False,
            )
            male = torch.load(
                male_path, map_location="cpu", weights_only=False,
            )
            self.assertEqual(tuple(female["embeddings"].shape), (1, 2))
            self.assertEqual(female["kaisu"], 12)
            self.assertEqual(male["kaisu"], 12)
            self.assertAlmostEqual(female["like_score"], 5.0)
            self.assertAlmostEqual(male["like_score"], 3.0)

            index = build_emb_index(output_dir, session=2)
            samples = load_samples(
                ["F900__M900"], index,
                target="like_score", speaker="female",
            )
            self.assertEqual(samples[0]["_kaisu"], 12)

            class ZeroModel(torch.nn.Module):
                def forward(self, embeddings, mask):
                    return torch.zeros(
                        embeddings.shape[0], device=embeddings.device,
                    )

            supervised_rows = run_inference_single(
                ZeroModel(), samples, torch.device("cpu"),
                mu=5.0, sig=1.0, fold_group=7,
            )
            self.assertEqual(supervised_rows[0]["kaisu"], 12)
            llm_rows = pd.DataFrame([{
                **supervised_rows[0],
                "y_pred": 4.5,
            }])
            merged = _merge_prediction_sources({
                "supervised": pd.DataFrame(supervised_rows),
                "llm": llm_rows,
            })
            self.assertEqual(merged.loc[0, "kaisu_supervised"], 12)
            self.assertEqual(merged.loc[0, "kaisu_llm"], 12)


class FusionSelectionTests(unittest.TestCase):
    def setUp(self):
        folds = np.array([1, 2, 2, 3])
        female_ids = np.array(["F900", "F900", "F901", "F902"])
        male_ids = np.array(["M900", "M901", "M900", "M902"])
        self.folds = folds
        self.female_ids = female_ids
        self.male_ids = male_ids

    def test_pooled_protocol_uses_every_non_test_row(self):
        pooled = _build_selection_mask(
            self.folds, self.female_ids, self.male_ids, test_fold=1,
            strict_participant_disjoint=False,
        )
        np.testing.assert_array_equal(pooled, [False, True, True, True])

    def test_strict_protocol_removes_rows_involving_test_participants(self):
        strict = _build_selection_mask(
            self.folds, self.female_ids, self.male_ids, test_fold=1,
            strict_participant_disjoint=True,
        )
        np.testing.assert_array_equal(strict, [False, False, False, True])

    def test_constant_prediction_series_contributes_zero_ccc(self):
        score = _per_participant_ccc(
            np.array([1.0, 2.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 1.5, 1.5]),
            np.array(["R1", "R1", "R2", "R2"]),
        )
        self.assertAlmostEqual(score, 0.5)


class FusionInputValidationTests(unittest.TestCase):
    @staticmethod
    def _frame(**overrides):
        data = {
            "pair_key": ["F900__M900", "F901__M901"],
            "y_true": [4.0, 6.0],
            "y_pred": [4.5, 5.5],
            "fold_group": [1, 2],
            "female_id": ["F900", "F901"],
            "male_id": ["M900", "M901"],
            "kaisu": [12, 13],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_valid_sources_merge_on_common_pairs(self):
        rounded = self._frame(y_true=[4.0000004, 5.9999996])
        merged = _merge_prediction_sources(
            {"llm": self._frame(), "hubert": rounded}
        )
        self.assertEqual(merged["pair_key"].tolist(), [
            "F900__M900", "F901__M901",
        ])

    def test_substantive_target_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "y_true mismatch"):
            _merge_prediction_sources({
                "llm": self._frame(),
                "hubert": self._frame(y_true=[4.0, 6.01]),
            })

    def test_pair_set_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pair_key set mismatch"):
            _merge_prediction_sources({
                "llm": self._frame(),
                "hubert": self._frame(
                    pair_key=["F900__M900", "F902__M902"],
                ),
            })

    def test_kaisu_mismatch_requires_explicit_legacy_opt_in(self):
        sources = {
            "canonical": self._frame(),
            "legacy": self._frame(kaisu=[1, 2]),
        }
        with self.assertRaisesRegex(ValueError, "kaisu mismatch"):
            _merge_prediction_sources(sources)
        merged = _merge_prediction_sources(
            sources, allow_kaisu_mismatch=True,
        )
        self.assertEqual(merged["kaisu_canonical"].tolist(), [12, 13])

    def test_metadata_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fold_group mismatch"):
            _merge_prediction_sources({
                "llm": self._frame(),
                "hubert": self._frame(fold_group=[1, 3]),
            })

    def test_duplicate_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path_a = Path(tmp) / "a.csv"
            path_b = Path(tmp) / "b.csv"
            self._frame().to_csv(path_a, index=False)
            self._frame().to_csv(path_b, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate prediction label"):
                _load_prediction_sources([
                    f"model:{path_a}", f"model:{path_b}",
                ])

    def test_duplicate_pair_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path_a = Path(tmp) / "a.csv"
            path_b = Path(tmp) / "b.csv"
            duplicated = self._frame(
                pair_key=["F900__M900", "F900__M900"],
            )
            duplicated.to_csv(path_a, index=False)
            self._frame().to_csv(path_b, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate pair_key"):
                _load_prediction_sources([
                    f"a:{path_a}", f"b:{path_b}",
                ])


class ApiResponsePreservationTests(unittest.TestCase):
    def test_sdk_response_is_converted_and_failed_attempt_is_written(self):
        class FakeResponse:
            def model_dump(self, mode="python"):
                return {"mode": mode, "content": [{"text": "raw"}]}

        raw = serialize_api_response(FakeResponse())
        self.assertEqual(raw["content"][0]["text"], "raw")

        handle = io.StringIO()
        error = ApiResponseError("invalid JSON", raw)
        write_attempt_error(
            handle,
            {"pair_key": "F900__M900", "condition": "synthetic"},
            attempt=0,
            max_retries=2,
            error=error,
        )
        write_attempt_error(
            handle,
            {"pair_key": "F900__M900", "condition": "synthetic"},
            attempt=1,
            max_retries=2,
            error=RuntimeError("network failure before response"),
            raw_output=None,
        )
        record, no_response_record = [
            json.loads(line) for line in handle.getvalue().splitlines()
        ]
        self.assertEqual(record["attempt"], 1)
        self.assertEqual(record["max_attempts"], 3)
        self.assertTrue(record["will_retry"])
        self.assertEqual(record["raw_output"], raw)
        self.assertEqual(no_response_record["attempt"], 2)
        self.assertIsNone(no_response_record["raw_output"])


if __name__ == "__main__":
    unittest.main()
