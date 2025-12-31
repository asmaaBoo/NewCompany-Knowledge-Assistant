"""
Generator Module - Multi-LLM Support 
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class Generator:
    """Generate answers using GPT or Groq"""
    
    def __init__(self, model="gpt-4o-mini", temperature=0, provider=None):
        """
        Initialize generator
        
        Args:
            model: Model name (e.g., "gpt-4o-mini" or "llama-3.3-70b-versatile")
            temperature: Model temperature (0-1)
            provider: "openai" or "groq" (auto-detected if None)
        """
        self.model = model
        self.temperature = temperature
        
        # Auto-detect provider if not specified
        if provider is None:
            if "gpt" in model.lower():
                provider = "openai"
            elif "llama" in model.lower() in model.lower():
                provider = "groq"
            else:
                provider = "openai"  # Default
        
        self.provider = provider.lower()
        
        # Initialize LLM based on provider
        if self.provider == "openai":
            self.llm = ChatOpenAI(model=model, temperature=temperature)
            print(f"✅ Using OpenAI: {model}")
            
        elif self.provider == "groq":
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(model=model, temperature=temperature)
                print(f"✅ Using Groq: {model}")
            except ImportError:
                print("⚠️  Warning: langchain-groq not installed. Falling back to OpenAI.")
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
                self.provider = "openai"
        
        else:
            # Fallback to OpenAI
            print(f"⚠️  Warning: Unknown provider '{self.provider}'. Using OpenAI.")
            self.llm = ChatOpenAI(model=model, temperature=temperature)
            self.provider = "openai"
        
        self.prompt = ChatPromptTemplate.from_template("""
أنت مساعد ذكي مختص بالإجابة على الاستفسارات.

السياق المتوفر:
{context}

السؤال:
{question}

التعليمات:
- إذا كانت الإجابة موجودة بوضوح في سياسات المركز، أجب منها مباشرة.
- إذا لم يتم ذكر الإجابة في سياسات المركز، وُجدت معلومات داعمة من الإنترنت:
  • وضّح أولاً أن المعلومة غير مذكورة في السياسات.
  • ثم أجب اعتمادًا على المعلومات المتوفرة من الإنترنت بصيغة عامة وغير ملزمة.
- لا تعرض المعلومات المستخرجة من الإنترنت على أنها سياسة رسمية للمركز.
- إذا لم تتوفر أي معلومات مفيدة لا في السياسات ولا في الإنترنت، اذكر ذلك بوضوح.

الإجابة:
""")
    
    def generate(self, question, context):
        """Generate answer from question and context"""
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})