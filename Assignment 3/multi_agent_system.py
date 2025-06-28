import asyncio
import os
import logging
from typing import Dict, Optional, Any
from openai import AsyncOpenAI

# === Setup OpenAI Client ===
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if client.api_key is None:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

# === Logging Config ===
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MultiAgentSystem")

# === Async LLM Call ===
async def openai_llm_call(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 200) -> str:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return "[ERROR]"

# === Message & Agent Classes ===
class Message:
    def __init__(self, sender: str, task: str, payload: Optional[Dict[str, Any]] = None):
        self.sender = sender
        self.task = task
        self.payload = payload or {}

    def __repr__(self):
        return f"Message(from={self.sender}, task={self.task})"

class Agent:
    def __init__(self, name: str, inbox: asyncio.Queue, outboxes: Dict[str, asyncio.Queue]):
        self.name = name
        self.inbox = inbox
        self.outboxes = outboxes
        self.running = True
        self.logger = logging.getLogger(self.name)

    async def send(self, to: str, task: str, payload: Dict[str, Any]):
        msg = Message(sender=self.name, task=task, payload=payload)
        await self.outboxes[to].put(msg)
        self.logger.info(f"Sent message to {to}: task={task}")

    async def receive(self) -> Optional[Message]:
        try:
            msg = await asyncio.wait_for(self.inbox.get(), timeout=10)
            self.logger.info(f"Received message from {msg.sender}: task={msg.task}")
            return msg
        except asyncio.TimeoutError:
            return None

    async def process_message(self, msg: Message):
        raise NotImplementedError

    async def run(self):
        self.logger.info("Started.")
        while self.running:
            msg = await self.receive()
            if msg:
                try:
                    await self.process_message(msg)
                except Exception as e:
                    self.logger.error(f"Error processing {msg.task}: {e}")

    def stop(self):
        self.running = False
        self.logger.info("Stopping agent.")

# === Specific Agents ===
class PlannerAgent(Agent):
    async def process_message(self, msg: Message):
        if msg.task == "plan":
            prompt = msg.payload.get("prompt", "")
            plan = await openai_llm_call(f"Create a step-by-step plan for: {prompt}")
            await self.send("ValidatorAgent", "validate_plan", {"plan": plan})

class ValidatorAgent(Agent):
    async def process_message(self, msg: Message):
        if msg.task == "validate_plan":
            plan = msg.payload.get("plan", "")
            if "[ERROR]" in plan or len(plan.split()) < 10:
                self.logger.error("Received invalid plan. Asking Planner to retry.")
                await self.send("PlannerAgent", "plan", {"prompt": "Retry planning with more detail."})
            else:
                await self.send("SummarizerAgent", "summarize_plan", {"plan": plan})

class SummarizerAgent(Agent):
    async def process_message(self, msg: Message):
        if msg.task == "summarize_plan":
            plan = msg.payload.get("plan", "")
            summary = await openai_llm_call(f"Summarize this plan briefly:\n{plan}")
            await self.send("AnswerAgent", "answer", {"summary": summary})

class AnswerAgent(Agent):
    async def process_message(self, msg: Message):
        if msg.task == "answer":
            summary = msg.payload.get("summary", "")
            answer = await openai_llm_call(f"Based on this summary, what is the key insight?\n\n{summary}")
            self.logger.info(f"🤖 Final Answer:\n{answer}")
            print(f"\n=== FINAL OUTPUT ===\n{answer}\n====================\n")
            self.stop()

# === Main Async Runner ===
async def main():
    inboxes = {
        "PlannerAgent": asyncio.Queue(),
        "ValidatorAgent": asyncio.Queue(),
        "SummarizerAgent": asyncio.Queue(),
        "AnswerAgent": asyncio.Queue(),
    }

    agents = [
        PlannerAgent("PlannerAgent", inboxes["PlannerAgent"], inboxes),
        ValidatorAgent("ValidatorAgent", inboxes["ValidatorAgent"], inboxes),
        SummarizerAgent("SummarizerAgent", inboxes["SummarizerAgent"], inboxes),
        AnswerAgent("AnswerAgent", inboxes["AnswerAgent"], inboxes),
    ]

    tasks = [asyncio.create_task(agent.run()) for agent in agents]

    # ✅ Take dynamic user input
    initial_prompt = input("📝 Enter your task/question: ").strip()
    if not initial_prompt:
        initial_prompt = "Explain how a multi-agent system works with LLMs."

    await inboxes["PlannerAgent"].put(Message("User", "plan", {"prompt": initial_prompt}))

    try:
        await asyncio.wait_for(tasks[-1], timeout=120)
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for final output.")

    for agent in agents:
        agent.stop()
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
