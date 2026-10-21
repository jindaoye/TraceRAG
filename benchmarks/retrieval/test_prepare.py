"""Check that data conversion preserves answers and evidence relationships."""

import unittest

from prepare import cmrc_records, hotpot_records


class ConversionTests(unittest.TestCase):
    def test_cmrc_keeps_source_label_and_deduplicates_answers(self):
        corpus, questions = cmrc_records([
            {"context_id": "d1", "title": "设备", "context_text": "E03 是过热保护。",
             "qas": [{"query_id": "q1", "query_text": "E03 是什么？",
                      "answers": ["过热保护", "过热保护", 39764.0]}]}
        ])
        self.assertEqual(questions[0]["answers"], ["过热保护"])
        self.assertEqual(questions[0]["non_text_answers_skipped"], 1)
        self.assertEqual(questions[0]["relevant_documents"], [corpus[0]["id"]])

    def test_hotpot_preserves_different_versions_of_the_same_title(self):
        rows = []
        for i, sentence in enumerate(["Alice was born in Paris.", "Alice lives in London."]):
            rows.append({"row": {
                "id": f"q{i}", "question": "Where?", "answer": "Paris" if i == 0 else "London",
                "context": {"title": ["Alice"], "sentences": [[sentence]]},
                "supporting_facts": {"title": ["Alice"], "sent_id": [0]},
            }})
        corpus, questions = hotpot_records({"rows": rows})
        self.assertEqual(len(corpus), 2)
        self.assertNotEqual(questions[0]["relevant_documents"], questions[1]["relevant_documents"])
        documents = {doc["id"]: doc["text"] for doc in corpus}
        for question in questions:
            fact = question["evidence"][0]
            self.assertEqual(documents[fact["document_id"]], fact["sentence"])

    def test_hotpot_rejects_a_supporting_sentence_outside_the_document(self):
        raw = {"rows": [{"row": {
            "id": "q1", "question": "Where?", "answer": "Paris",
            "context": {"title": ["Alice"], "sentences": [["Paris"]]},
            "supporting_facts": {"title": ["Alice"], "sent_id": [3]},
        }}]}
        with self.assertRaises(ValueError):
            hotpot_records(raw)


if __name__ == "__main__":
    unittest.main()
