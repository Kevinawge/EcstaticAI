import os
from openai import OpenAI
from typing import List, Dict

# Set your API key here (replace with your actual key)
OPENAI_API_KEY = "sk-proj-..."  # replace with your actual key
client = OpenAI(api_key=OPENAI_API_KEY)


class GPTFinanceAssistant:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_prompt: str = "You are using EcstaticAI, a financial assistant. I provide accurate, clear, and reliable financial explanations, strategies, and summaries.",
        temperature: float = 0.2,
        max_tokens: int = 800,
        verbose: bool = True,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self._message_log: List[Dict[str, str]] = []

        if self.verbose:
            print(f"[GPT INIT] Model: {self.model}")
            print(f"[GPT INIT] System Prompt: {self.system_prompt}")

    def reset_chat(self):
        self._message_log = []
        if self.verbose:
            print("[GPT] Chat memory reset.")

    def ask(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt}] + self._message_log
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            result = response.choices[0].message.content.strip()
            self._message_log.append({"role": "user", "content": user_prompt})
            self._message_log.append({"role": "assistant", "content": result})

            if self.verbose:
                print(
                    f"[GPT RESPONSE] {result[:100]}{'...' if len(result) > 100 else ''}")

            return result

        except Exception as e:
            return f"[GPT Error] {str(e)}"

    def explain_term(self, term: str) -> str:
        return self.ask(f"Explain the financial term '{term}' with an example and mathematical formulation.")

    def analyze_strategy(self, description: str) -> str:
        return self.ask(f"Analyze this trading or investment strategy:\n\n{description}")


# Testing/Example
if __name__ == "__main__":
    print("\n[Running GPT Agent Test...]\n")
    assistant = GPTFinanceAssistant()
    response = assistant.ask("What is the Capital Asset Pricing Model (CAPM)?")
    print("\n[Full Answer]\n")
    print(response)
