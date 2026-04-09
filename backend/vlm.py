import ollama
import base64
from dataclasses import dataclass

@dataclass
class VLMResult:
    triggered: bool
    explanation: str

class VLMChecker:
    def __init__(self, model: str = "llava:7b"):
        self.model = model

    def check(self, frame_b64: str, condition: dict) -> VLMResult:
        prompt = f"""You are a security camera analyst.
Condition to check: {condition}

Look at this image carefully. Answer ONLY in this exact format:
TRIGGERED: yes or no
REASON: one sentence explanation

Does the condition apply to this image?"""

        response = ollama.chat(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [frame_b64]
            }]
        )

        text = response["message"]["content"].strip()
        triggered = "triggered: yes" in text.lower()
        reason = ""
        for line in text.splitlines():
            if line.lower().startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
        return VLMResult(triggered=triggered, explanation=reason)