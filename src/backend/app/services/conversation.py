from app.models_conversation import (
    AcceptanceCriterion,
    AcceptanceCriterionScores,
    ConversationMessage,
    ConversationRequest,
    GenerateResult,
)

_CANNED_ASSISTANT_REPLY = (
    "From viewing the user stories there seems to be enough context, "
    "feel free to generate the acceptance criteria."
)


def _build_example_generate_result() -> GenerateResult:
    return GenerateResult(
        acceptance_criteria=[
            AcceptanceCriterion(
                id=1,
                title="Successful login → /home",
                given="the user is on the login page with a valid registered account",
                when="the user submits their correct username and password",
                then="the user is redirected to /home and a welcome message is shown",
                scores=AcceptanceCriterionScores(
                    relevance=9.4, correctness=9.0, understandability=9.2, coverage=8.8
                ),
                overall_score=9.1,
            ),
            AcceptanceCriterion(
                id=2,
                title="Failed login shows inline error",
                given="the user is on the login page",
                when="the user submits an incorrect username or password",
                then="an inline error message is shown and the user stays on the login page",
                scores=AcceptanceCriterionScores(
                    relevance=9.0, correctness=8.7, understandability=8.9, coverage=8.3
                ),
                overall_score=8.7,
            ),
            AcceptanceCriterion(
                id=3,
                title="Account lockout after repeated failures",
                given="the user has failed to log in 4 times in a row",
                when="the user submits a 5th consecutive incorrect login attempt",
                then="the account is temporarily locked and a lockout message with a retry time is shown",
                scores=AcceptanceCriterionScores(
                    relevance=8.6, correctness=8.5, understandability=8.4, coverage=8.0
                ),
                overall_score=8.4,
            ),
        ]
    )


class ConversationService:
    """Stubbed conversation logic, independent of HTTP concerns.

    No AI is wired up yet, so both methods ignore the conversation content
    and return fixed example data.
    """

    def send_message(self, data: ConversationRequest) -> ConversationMessage:
        return ConversationMessage(role="assistant", text=_CANNED_ASSISTANT_REPLY)

    def generate(self, data: ConversationRequest) -> GenerateResult:
        return _build_example_generate_result()
