import logging
import ollama
import json
import re

logger = logging.getLogger(__name__)

class QueryCompiler:
    def __init__(self, model: str = "mistral"):
        self.model = model

    def compile(self, natural_language_query: str) -> dict:
        prompt = f"""Convert this security camera alert instruction into a JSON condition object.
Instruction: "{natural_language_query}"

Respond with ONLY a valid JSON object, no explanation. Example:
{{"event": "unattended_bag", "zone": "entrance", "description": "a bag left without a person nearby"}}

Now convert the instruction above:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            logger.error("Compilation inference failed: %s", e)
            return {"description": natural_language_query, "_compiled": False}

        text = response["message"]["content"].strip()
        # Strip markdown code fences if present
        text = re.sub(r"```json|```", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse compiler output: %s", text)
            return {"description": natural_language_query, "_compiled": False}