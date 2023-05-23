from transformers import pipeline


class RobertaQAModel(object):
    """Roberta LLM"""

    def __init__(self, context):
        self.context = context
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/xlm-roberta-large-squad2",
            tokenizer="deepset/xlm-roberta-large-squad2",
        )

    def query(self, question):
        return self.qa_pipeline({"context": self.context, "question": question})[
            "answer"
        ]
