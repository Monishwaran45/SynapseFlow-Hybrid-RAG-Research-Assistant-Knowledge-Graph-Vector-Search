"""
SynapseFlow — LLM Service
Configurable LLM backend supporting Google Gemini and Groq.
Handles answer generation, entity/relation extraction, and insight analysis.
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config


class LLMService:
    """Unified LLM interface supporting Gemini and Groq."""
    
    def __init__(self):
        self.provider = config.LLM_PROVIDER
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the LLM client based on provider config."""
        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            self._client = genai.GenerativeModel("gemini-2.0-flash")
        
        elif self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=config.GROQ_API_KEY)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated text string
        """
        try:
            if self.provider == "gemini":
                response = self._client.generate_content(prompt)
                return response.text
            
            elif self.provider == "groq":
                response = self._client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
        
        except Exception as e:
            return f"[LLM Error: {str(e)}]"
    
    def extract_entities_relations(self, text: str) -> list:
        """
        Extract entities and relationships from text using LLM.
        
        Args:
            text: Text to extract from (truncated to 2000 chars)
        
        Returns:
            List of [entity1, relation, entity2] triples
        """
        prompt = f"""Extract key entities and their relationships from the following text.

Return ONLY a JSON array of triples in this exact format:
[["Entity1", "relationship", "Entity2"], ...]

Rules:
- Entities should be specific concepts, methods, tools, people, or organizations
- Relationships should be verbs like "uses", "improves", "proposes", "compares_with", "is_part_of", "developed_by", "applied_to", "outperforms"
- Extract 3-8 triples maximum
- Keep entity names concise (2-4 words max)
- Return ONLY the JSON array, no explanation

Text:
{text[:2000]}"""
        
        response = self.generate(prompt)
        
        try:
            # Find JSON array in response
            match = re.search(r'\[[\s\S]*\]', response)
            if match:
                triples = json.loads(match.group())
                # Validate: each triple must be a list of 3 strings
                valid = []
                for t in triples:
                    if isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) for x in t):
                        valid.append(t)
                return valid
        except (json.JSONDecodeError, Exception):
            pass
        
        return []
    
    def generate_answer(self, query: str, text_context: str, graph_context: str) -> str:
        """
        Generate a research-quality answer using both text chunks and graph relations.
        
        Args:
            query: User's question
            text_context: Retrieved text chunks
            graph_context: Knowledge graph relations
        
        Returns:
            Formatted answer string
        """
        prompt = f"""You are SynapseFlow, an advanced AI research assistant. Answer the user's question using BOTH the text context from documents AND the knowledge graph relations.

═══ TEXT CONTEXT (from uploaded documents) ═══
{text_context[:4000]}

═══ KNOWLEDGE GRAPH RELATIONS ═══
{graph_context[:1500] if graph_context else "No graph relations available for this query."}

═══ QUESTION ═══
{query}

═══ INSTRUCTIONS ═══
- Provide a comprehensive, well-structured answer
- Use **bold** for key terms and concepts
- Use bullet points and numbered lists for clarity
- Reference specific findings from the context
- If graph relations provide cross-document insights, highlight them
- If the context is insufficient, say so honestly
- Keep the answer focused and actionable
- Add a "Key Takeaways" section at the end with 2-3 bullet points"""
        
        return self.generate(prompt, max_tokens=3000)
    
    def extract_insights(self, text: str) -> dict:
        """
        Extract structured research insights from text.
        
        Args:
            text: Combined text from document chunks
        
        Returns:
            Dict with contributions, methodology, limitations, future_work
        """
        prompt = f"""Analyze this research text and extract structured insights.

Return a JSON object with these exact keys, each containing a list of concise strings:

{{
    "contributions": ["list of key contributions or novelties"],
    "methodology": ["list of approaches/methods used"],
    "limitations": ["list of acknowledged weaknesses or constraints"],
    "future_work": ["list of suggested future directions"]
}}

Rules:
- Each item should be 1-2 sentences max
- Extract 2-5 items per category
- If a category has no relevant info, use an empty list
- Return ONLY the JSON, no other text

Text:
{text[:4000]}"""
        
        response = self.generate(prompt)
        
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                result = json.loads(match.group())
                # Ensure all keys exist
                for key in ["contributions", "methodology", "limitations", "future_work"]:
                    if key not in result:
                        result[key] = []
                    elif not isinstance(result[key], list):
                        result[key] = [str(result[key])]
                return result
        except (json.JSONDecodeError, Exception):
            pass
        
        return {
            "contributions": ["Could not extract — try with more text"],
            "methodology": [],
            "limitations": [],
            "future_work": []
        }
