"""jevlet — a small, calibrated System One decision layer for edge agents.

The wire shape follows TypeSafe's Jev (state + typed questions -> typed answers with
probabilities and a confidence), so any backend here can be dropped behind the same
application code: a cloud Jev, an open encoder decision model, a next-token logprob readout
over a local llama.cpp server, or plain rules. The point of the package is not the wire
shape; it is that every backend returns the *same epistemic object* (a distribution the
application thresholds) and that the calibration of that distribution is measured, not assumed.
"""

from .schema import Question, Answer, Noul, Choice, Score, confidence, normalize_questions
from .backends.base import Backend, DecisionResult

__all__ = ["Question", "Answer", "Noul", "Choice", "Score", "confidence", "normalize_questions",
           "Backend", "DecisionResult"]
__version__ = "0.1.0"
