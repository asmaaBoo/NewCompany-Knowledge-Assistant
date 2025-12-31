"""
RAGAS Evaluation - Full Pipeline Integration
Uses all RAGPipeline features: caching, web search, config settings
"""

from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from src.config import Config
from src.retriever import Retriever
from src.generator import Generator
from src.rag_pipeline import RAGPipeline
from datasets import Dataset
import numpy as np
from langchain_openai import ChatOpenAI
from ragas import evaluate


# 12 evaluation questions
QUESTIONS = [
    {
        "question": "ما هو الغرض من سياسة الاستقطاب والتوظيف؟",
        "ground_truth": "تحديد أساس استقطاب وتوظيف الأشخاص وضمان تكافؤ الفرص على أساس الجدارة والمؤهلات"
    },
    {
        "question": "كم مدة فترة الاختبار؟",
        "ground_truth": "90 يوماً من تاريخ توقيع عقد العمل"
    },
    {
        "question": "كم مدة الإجازة السنوية؟",
        "ground_truth": "33 يوماً مدفوعة الأجر مع تذكرة سفر"
    },
    {
        "question": "ما عواقب عدم الالتزام بقواعد السلوك؟",
        "ground_truth": "إجراءات تأديبية تشمل التوبيخ أو تعليق العمل أو خفض المرتبة أو إنهاء الخدمة"
    },
    {
        "question": "ما الغرض من سياسة الصحة والسلامة؟",
        "ground_truth": "الحفاظ على أفضل ظروف عمل وضمان شعور الموظفين بالأمان"
    },
    {
        "question": "ما الهدف من سياسة تعارض المصالح؟",
        "ground_truth": "حماية النزاهة ومنع تأثير المصالح الشخصية على أداء العاملين"
    },
    {
        "question": "ما الهدف من سياسة المحاسبة المالية؟",
        "ground_truth": "تحديد قواعد مسك الدفاتر وتنظيم القيود المحاسبية والموازنات"
    },
    {
        "question": "ما ضوابط سياسة الاستثمار الآمن؟",
        "ground_truth": "تشكيل لجنة استثمار ومشروعية الاستثمار والتخطيط المحكم والابتعاد عن المخاطر العالية"
    },
    {
        "question": "ما هي الموارد المالية للمركز؟",
        "ground_truth": "رسوم العضوية والتبرعات والإعانات الحكومية وعائدات الاستثمار"
    },
    {
        "question": "هل توجد إجازة طارئة؟",
        "ground_truth": "نعم، يحق للموظف الحصول على إجازة طارئة في حالات معينة"
    },
    {
        "question": "ما سياسة الإبلاغ عن المخالفات؟",
        "ground_truth": "توفير قنوات للإبلاغ مع حماية المبلغين من الانتقام"
    },
    {
        "question": "كيف يتعامل المركز مع غسل الأموال؟",
        "ground_truth": "إجراءات وقائية تشمل تقييم المخاطر والقنوات غير النقدية والتدريب"
    }
]




def run_evaluation():
    # Setup
    Config.setup()

    retriever = Retriever(
        embedding_model=Config.EMBEDDING_MODEL,
        top_k=Config.TOP_K,
        vector_store_path=Config.VECTOR_STORE_PATH
    )

    if not retriever.load_vector_store():
        raise RuntimeError("Vector store not found. Run the app first.")

    generator = Generator(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        provider=Config.LLM_PROVIDER
    )

    pipeline = RAGPipeline(
        retriever,
        generator,
        enable_cache=Config.ENABLE_CACHE,
        enable_web_search=True
    )

    # Generate answers
    rows = []
    for q in QUESTIONS:
        result = pipeline.query(q["question"], return_contexts=True)
        rows.append({
            "question": q["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": q["ground_truth"],
            "reference": q["ground_truth"]
        })

    dataset = Dataset.from_list(rows)

    # RAGAS setup
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    ragas_embeddings = LangchainEmbeddingsWrapper(retriever.embeddings)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge_llm,
        embeddings=ragas_embeddings
    )

    print("\n RAGAS Evaluation Results")
    print(results)


if __name__ == "__main__":
    run_evaluation()