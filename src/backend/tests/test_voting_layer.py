"""The batch voting entry point: judge selection and the saved JSON."""

import asyncio
import json

from voting.models import (
    EvaluatedOutput,
    JudgeConfig,
    ProviderFeedback,
    RubricAverage,
    RubricFeedback,
    VotingResult,
)
from voting.voting_layer import run_voting_layer

JUDGES = [
    JudgeConfig(id="gemini", provider="google", model="gemini-3.6-flash"),
    JudgeConfig(
        id="prometheus",
        provider="ollama",
        model="ggozad/prometheus2:latest",
        style="per_rubric",
    ),
    JudgeConfig(id="claude", provider="anthropic", model="claude-sonnet-5-5"),
]


def test_run_voting_layer_returns_results_and_saves_file(tmp_path, monkeypatch):
    async def fake_evaluate_inputs(evaluation_inputs):
        assert [
            [judge.id for judge in evaluation_input.judges]
            for evaluation_input in evaluation_inputs
        ] == [["gemini", "prometheus"]] * 2
        return [
            VotingResult(
                ai="TestAI",
                model="test-model",
                prompt="Generate a requirement.",
                output=[
                    EvaluatedOutput(
                        output="A passable output",
                        feedback=[
                            ProviderFeedback(
                                ai="Gemini",
                                model="gemini-flash-latest",
                                feedback=[
                                    RubricFeedback(
                                        rubric="correctness", value=4, feedback="Good"
                                    ),
                                    RubricFeedback(
                                        rubric="coverage", value=3, feedback="Okay"
                                    ),
                                    RubricFeedback(
                                        rubric="relevance", value=4, feedback="Relevant"
                                    ),
                                    RubricFeedback(
                                        rubric="understandability",
                                        value=5,
                                        feedback="Clear",
                                    ),
                                ],
                                overall_score=4.0,
                            )
                        ],
                        rubric_averages=[
                            RubricAverage(rubric="correctness", value=4.0),
                            RubricAverage(rubric="coverage", value=3.0),
                            RubricAverage(rubric="relevance", value=4.0),
                            RubricAverage(rubric="understandability", value=5.0),
                        ],
                        overall_score=4.0,
                    )
                ],
                overall_score=4.0,
                rank=1,
            ),
            VotingResult(
                ai="AnotherAI",
                model="other-model",
                prompt="Generate another requirement.",
                output=[
                    EvaluatedOutput(
                        output="Another output",
                        feedback=[
                            ProviderFeedback(
                                ai="Prometheus",
                                model="ggozad/prometheus2:latest",
                                feedback=[
                                    RubricFeedback(
                                        rubric="correctness", value=3, feedback="Okay"
                                    ),
                                    RubricFeedback(
                                        rubric="coverage", value=2, feedback="Limited"
                                    ),
                                    RubricFeedback(
                                        rubric="relevance", value=4, feedback="Relevant"
                                    ),
                                    RubricFeedback(
                                        rubric="understandability",
                                        value=4,
                                        feedback="Readable",
                                    ),
                                ],
                                overall_score=3.25,
                            )
                        ],
                        rubric_averages=[
                            RubricAverage(rubric="correctness", value=3.0),
                            RubricAverage(rubric="coverage", value=2.0),
                            RubricAverage(rubric="relevance", value=4.0),
                            RubricAverage(rubric="understandability", value=4.0),
                        ],
                        overall_score=3.25,
                    )
                ],
                overall_score=3.25,
                rank=2,
            ),
        ]

    monkeypatch.setattr("voting.voting_layer.evaluate_inputs", fake_evaluate_inputs)
    monkeypatch.setattr("voting.voting_layer.load_judges", lambda: list(JUDGES))

    inputs = [
        {
            "ai": "TestAI",
            "model": "test-model",
            "prompt": "Generate a requirement.",
            "output": ["A passable output"],
        },
        {
            "ai": "AnotherAI",
            "model": "other-model",
            "prompt": "Generate another requirement.",
            "output": ["Another output"],
        },
    ]

    output_path = tmp_path / "votingLayerOutput.json"
    payload = asyncio.run(
        run_voting_layer(inputs, ["gemini", "prometheus"], output_path=output_path)
    )

    assert len(payload) == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
