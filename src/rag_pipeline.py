"""
RAG Pipeline 
"""

import time
import redis
import json



class RAGPipeline:
    """Simple RAG: Retriever + Generator + Cache + Web Search"""
    
    def __init__(self, retriever, generator, enable_cache=True, enable_web_search=True):
        self.retriever = retriever
        self.generator = generator
        self.enable_cache = enable_cache
        self.enable_web_search = enable_web_search
        self.cache = redis.Redis(
          host="localhost",
          port=6379,
           db=0,
          decode_responses=True)
                                                                                                                 
        self.web_search = None
        
        # Initialize web search (try multiple methods for different versions)
        if enable_web_search:
            try:
                # Try new version (8.x)
                from duckduckgo_search import DDGS
                self.web_search = DDGS()
                self.web_search_version = "new"
                print("✅ Web search enabled (DDGS)")
            except:
                try:
                    # Try LangChain wrapper
                    from langchain_community.tools import DuckDuckGoSearchResults
                    self.web_search = DuckDuckGoSearchResults()
                    self.web_search_version = "langchain"
                    print("✅ Web search enabled (LangChain)")
                except Exception as e:
                    print(f"⚠️ Web search disabled: {e}")
                    self.enable_web_search = False
                    self.web_search_version = None
    
    def _do_web_search(self, question):
        """Perform web search using available method"""
        try:
            if self.web_search_version == "new":
                # DDGS version 8.x
                results = self.web_search.text(question, max_results=3)
                # Format results
                formatted = []
                for r in results:
                    title = r.get('title', '')
                    body = r.get('body', '')
                    formatted.append(f"{title}: {body}")
                return "\n".join(formatted)
            
            elif self.web_search_version == "langchain":
                # LangChain wrapper
                return self.web_search.invoke(question)
            
        except Exception as e:
            print(f"❌ Web search error: {e}")
            return None
    
    def query(self, question, return_contexts=False, use_web_search=False):
        """Answer a question with optional web search"""
        start = time.time()
        
        # Check cache
        if self.enable_cache:
           cached = self.cache.get(question)
           if cached:
              result = json.loads(cached)
              result["cached"] = True
              result["latency"] = time.time() - start
              return result
        
        # Retrieve contexts from vector store
        docs = self.retriever.retrieve(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Generate initial answer
        answer = self.generator.generate(question, context)
        
        # Check if answer is unclear
        unclear_indicators = [
            "لا تتوفر إجابة",
            "غير واضح", 
            "لم يتم ذكر",
            "لا يوجد",
            "لم أجد"
        ]
        
        needs_web_search = any(indicator in answer for indicator in unclear_indicators)
        
        # Try web search if needed
        web_used = False
        if (needs_web_search or use_web_search) and self.enable_web_search and self.web_search:
            print(f"🔍 Searching web for: {question[:50]}...")
            web_results = self._do_web_search(question)
            
            if web_results:
                web_context = f"\n\n**معلومات من الإنترنت:**\n{web_results}"
                
                # Regenerate with web context
                full_context = context + web_context
                answer = self.generator.generate(question, full_context)
                web_used = True
                print("✅ Web search successful")
            else:
                print("❌ Web search returned no results")
        
        # Build result
        result = {
            "question": question,
            "answer": answer,
            "contexts": [d.page_content for d in docs],
            "num_contexts": len(docs),
            "latency": time.time() - start,
            "cached": False,
            "web_search_used": web_used
        }
        
        if return_contexts:
            result["contexts"] = [d.page_content for d in docs]
        
        # Cache it
        if self.enable_cache:
            self.cache.setex(
               question,
                3600,  
                json.dumps(result)
    )

        
        return result
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.flushdb()
    
    def get_cache_stats(self):
        """Get cache stats"""
        return {
            "enabled": self.enable_cache,
            "size": self.cache.dbsize(),
            "web_search_enabled": self.enable_web_search
        }