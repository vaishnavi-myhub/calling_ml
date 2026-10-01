import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import Base
from backend.services.documents import DocumentRAG


def _isolated_session_factory():
    """A private in-memory SQLite DB per test, so tests never touch the
    real backend/data/aura.db and can run in parallel/offline."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


class DocumentRagTests(unittest.TestCase):
    def test_keywords_match_uploaded_document_context(self):
        rag = DocumentRAG(session_factory=_isolated_session_factory())
        rag.add_document("policy.txt", b"Refunds are processed on the 15th of May. Policy: customers get a 30 day window for claims.")

        context = rag.get_context("When are refunds processed?")

        self.assertIn("15th", context)
        self.assertIn("May", context)

    def test_documents_persist_across_instances(self):
        session_factory = _isolated_session_factory()
        first = DocumentRAG(session_factory=session_factory)
        first.add_document("faq.txt", b"Support hours are 9am to 6pm, Monday to Saturday.")

        second = DocumentRAG(session_factory=session_factory)

        self.assertEqual(second.count(), 1)
        self.assertIn("9am", second.get_context("What are your support hours?"))

    def test_unrelated_query_returns_no_context(self):
        rag = DocumentRAG(session_factory=_isolated_session_factory())
        rag.add_document("policy.txt", b"Refunds are processed on the 15th of May.")

        self.assertEqual(rag.get_context("What is the capital of France?"), "")


if __name__ == "__main__":
    unittest.main()
