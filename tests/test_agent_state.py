import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent_state import AgentState, LoopDecision, QueryPlan


class AgentStateTest(unittest.TestCase):
    def test_query_plan_defaults_to_private_qa(self):
        plan = QueryPlan()

        self.assertEqual(plan.intent, "private_qa")
        self.assertTrue(plan.need_rag)
        self.assertTrue(plan.need_memory)
        self.assertEqual(tuple(plan.retrieval_modes), ("basic",))

    def test_agent_state_tracks_errors(self):
        state = AgentState(session_id="demo", raw_input="问题")

        state.add_error("node", "boom", recoverable=False)

        self.assertEqual(state.question, "")
        self.assertEqual(state.errors[0]["node"], "node")
        self.assertFalse(state.errors[0]["recoverable"])

    def test_loop_decision_defaults_tool_requests(self):
        decision = LoopDecision(action="build_context")

        self.assertEqual(decision.tool_requests, [])


if __name__ == "__main__":
    unittest.main()
