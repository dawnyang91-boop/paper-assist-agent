import os
import sys
import unittest
from types import SimpleNamespace


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.answer_verifier import AnswerVerifier
from rag.context_builder import ContextDocument


class AnswerVerifierTest(unittest.TestCase):
    def build_context(self, documents=None, memories=None):
        return SimpleNamespace(documents=documents or [], memories=memories or [])

    def test_missing_document_citation_is_error(self):
        verifier = AnswerVerifier()
        context = self.build_context([
            ContextDocument(doc_id="D1", content="A", score=1.0),
        ])

        result = verifier.verify("答案引用了不存在的来源。[D9]", context)

        self.assertFalse(result.passed)
        self.assertEqual(result.status, "citation_error")
        self.assertEqual(result.missing_citations, ["D9"])
        self.assertTrue(result.need_repair)

    def test_empty_answer_needs_repair(self):
        verifier = AnswerVerifier()

        result = verifier.verify("", self.build_context())

        self.assertFalse(result.passed)
        self.assertEqual(result.status, "answer_empty")
        self.assertTrue(result.need_repair)

    def test_claiming_documents_without_documents_is_risk(self):
        verifier = AnswerVerifier()

        result = verifier.verify("根据文档，SENet 使用 SE block。", self.build_context())

        self.assertFalse(result.passed)
        self.assertEqual(result.status, "hallucination_risk")

    def test_documents_without_citation_is_warning(self):
        verifier = AnswerVerifier()
        context = self.build_context([
            ContextDocument(doc_id="D1", content="A", score=1.0),
        ])

        result = verifier.verify("SENet 使用 SE block。", context)

        self.assertTrue(result.passed)
        self.assertEqual(result.status, "warning")
        self.assertTrue(result.warnings)


if __name__ == "__main__":
    unittest.main()
