import asyncio
import json

from voting.models import EvaluatedOutput, ProviderFeedback, RubricAverage, RubricFeedback, VotingResult


def test_run_voting_layer_returns_results_and_saves_file(tmp_path, monkeypatch):
    async def fake_evaluate_inputs(items):
        assert [item.providers for item in items] == [["gemini", "prometheus"], ["gemini", "prometheus"]]
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
                                    RubricFeedback(rubric="correctness", value=4, feedback="Good"),
                                    RubricFeedback(rubric="coverage", value=3, feedback="Okay"),
                                    RubricFeedback(rubric="relevance", value=4, feedback="Relevant"),
                                    RubricFeedback(rubric="understandability", value=5, feedback="Clear"),
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
                                    RubricFeedback(rubric="correctness", value=3, feedback="Okay"),
                                    RubricFeedback(rubric="coverage", value=2, feedback="Limited"),
                                    RubricFeedback(rubric="relevance", value=4, feedback="Relevant"),
                                    RubricFeedback(rubric="understandability", value=4, feedback="Readable"),
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

    monkeypatch.setattr("voting.votingLayer.evaluate_inputs", fake_evaluate_inputs)

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
    result = asyncio.run(
        __import__("voting.votingLayer", fromlist=["run_voting_layer"]).run_voting_layer(
            inputs,
            ["gemini", "prometheus"],
            output_path=output_path,
        )
    )

    assert len(result) == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
