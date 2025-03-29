from pydantic import Field
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
import google.generativeai as genai
import ollama

# Configure Gemini API with Google AI Studio Key
GOOGLE_AI_STUDIO_API_KEY = "AIzaSyAe8rheF4wv2ZHJB2YboUhyyVlM2y0vmla"

genai.configure(api_key=GOOGLE_AI_STUDIO_API_KEY)

#  Set up Global Parameters for AI Configuration
GENAI_PARAMS = {
    "temperature": 0,
    "top_p": 0.95,
    "max_tokens": 8192,
}

class HybridLLM(BaseLLM):
    primary_model_name: str = Field(default="gemini-2.0-flash")
    #fallback_model: str = Field(default="gemini-2.0-flash-001")
    #primary_model_name: str = Field(default="deepseek-r1:7b")
    fallback_model: str = Field(default="deepseek-r1:14b")
    used_model: str = Field(default=None)
    primary_model: any = Field(default=None, exclude=True)

    def __init__(self, primary_model="gemini-2.0-flash", fallback_model="deepseek-r1:7b", **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'primary_model_name', primary_model)
        object.__setattr__(self, 'fallback_model', fallback_model)
        object.__setattr__(self, 'primary_model', genai.GenerativeModel(self.primary_model_name))
        
    def _call(self, prompt: str, stop=None):
        result = self._generate([prompt], stop)
        return result.generations[0][0].text
    
    def _generate(self, prompts, stop=None):
        generations = []
        for prompt in prompts:
            try:
                response = self.primary_model.generate_content(prompt)
                if response and hasattr(response, "text"):
                    object.__setattr__(self, 'used_model', "Gemini")
                    output = response.text.strip()
                else:
                    raise ValueError("No valid response from Gemini")
            except Exception as e:
                print(f"Gemini failed: {e}, switching to DeepSeek...")
                try:
                    response = ollama.chat(
                        model=self.fallback_model, 
                        messages=[{"role": "user", "content": prompt}]
                    )
                    object.__setattr__(self, 'used_model', "DeepSeek")
                    output = response["message"]["content"]
                except Exception as e:
                    print(f"DeepSeek also failed: {e}")
                    output = "Both Gemini and DeepSeek failed to generate a response."
            
            generations.append([Generation(text=output)])
        
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self):
        return "hybrid_llm"
    
    def get_used_model(self):
        return self.used_model

