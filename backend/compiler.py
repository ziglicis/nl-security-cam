import ollama
import json
import re

class QueryCompiler:
    def __init__(self, model: str = "llava:7b"):
        self.model = model

    def compile(self, natural_language_query: str) -> dict:
        prompt = f"""Convert this security camera alert instruction into a JSON condition object.
Instruction: "{natural_language_query}"

Respond with ONLY a valid JSON object, no explanation. Example:
{{"event": "unattended_bag", "zone": "entrance", "description": "a bag left without a person nearby"}}

Now convert the instruction above:"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response["message"]["content"].strip()
        # Strip markdown code fences if present
        text = re.sub(r"```json|```", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: return raw description
            return {"description": natural_language_query}