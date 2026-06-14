#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import AgentConfig
from src.workflow import SupportAgent


SCENARIOS = [
    ("Общий вопрос", "Какие кредиты вы предлагаете малому бизнесу?", None, "chat_site"),
    ("Статус заявки", "Что с моей заявкой на кредит?", "C-000002", "chat_intern"),
    ("Коллизия регламента", "У меня счёт открыт 4 месяца. Могу ли я оформить Бизнес-Старт?", "C-000012", "chat_intern"),
    ("Продажная эскалация", "Хочу подать заявку на Бизнес-Оборот, свяжитесь со мной.", "C-000001", "chat_intern"),
    ("Негативная эскалация", "Ваш банк опять тянет время, позовите оператора, я буду писать жалобу.", "C-000002", "chat_intern"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", help="Run a custom question")
    parser.add_argument("--client-id")
    parser.add_argument("--channel", default="chat_site")
    parser.add_argument("--online", action="store_true", help="Use GigaChat if dependencies and credentials are available")
    args = parser.parse_args()

    agent = SupportAgent(AgentConfig.from_env(force_local=not args.online))
    scenarios = [("Custom", args.question, args.client_id, args.channel)] if args.question else SCENARIOS

    for title, question, client_id, channel in scenarios:
        response = agent.answer(question, client_id=client_id, channel=channel)
        print("\n" + "=" * 80)
        print(title)
        print(f"Q: {question}")
        print(f"client_id={client_id}, channel={channel}")
        print(f"intent={response.intent}, outcome={response.outcome_type}, escalation={response.escalation.required}")
        if response.escalation.ticket_id:
            print(f"ticket={response.escalation.ticket_id}")
        print("A: " + response.answer)
        print("Sources: " + ", ".join(chunk.source for chunk in response.sources[:4]))
        print("Tool calls: " + (", ".join(call.name for call in response.tool_calls) if response.tool_calls else "none"))
        print(f"Trace: {response.trace_id}")


if __name__ == "__main__":
    main()
