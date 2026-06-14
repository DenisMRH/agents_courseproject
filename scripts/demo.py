#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import AgentConfig
from src.display import label_channel, label_intent, label_outcome, label_tool
from src.workflow import SupportAgent


SCENARIOS = [
    ("Общий вопрос", "Какие кредиты вы предлагаете малому бизнесу?", None, "chat_site"),
    ("Статус заявки", "Что с моей заявкой на кредит?", "C-000002", "chat_intern"),
    ("Коллизия регламента", "У меня счёт открыт 4 месяца. Могу ли я оформить Бизнес-Старт?", "C-000012", "chat_intern"),
    ("Продажная эскалация", "Хочу подать заявку на Бизнес-Оборот, свяжитесь со мной.", "C-000001", "chat_intern"),
    ("Негативная эскалация", "Ваш банк опять тянет время, позовите оператора, я буду писать жалобу.", "C-000002", "chat_intern"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо агента поддержки кредитования МСБ")
    parser.add_argument("--question", help="Запустить свой вопрос")
    parser.add_argument("--client-id", help="ID клиента для клиентских сценариев")
    parser.add_argument("--channel", default="chat_site", help="Код канала обращения")
    parser.add_argument("--online", action="store_true", help="Использовать GigaChat, если зависимости и учётные данные доступны")
    args = parser.parse_args()

    agent = SupportAgent(AgentConfig.from_env(force_local=not args.online))
    scenarios = [("Пользовательский вопрос", args.question, args.client_id, args.channel)] if args.question else SCENARIOS

    for title, question, client_id, channel in scenarios:
        response = agent.answer(question, client_id=client_id, channel=channel)
        print("\n" + "=" * 80)
        print(title)
        print(f"Вопрос: {question}")
        print(f"ID клиента={client_id or 'нет'}, канал={label_channel(channel)}")
        print(
            f"интент={label_intent(response.intent)}, результат={label_outcome(response.outcome_type)}, "
            f"эскалация={'да' if response.escalation.required else 'нет'}"
        )
        if response.escalation.ticket_id:
            print(f"тикет={response.escalation.ticket_id}")
        print("Ответ: " + response.answer)
        print("Источники: " + ", ".join(chunk.source for chunk in response.sources[:4]))
        print("Вызовы инструментов: " + (", ".join(label_tool(call.name) for call in response.tool_calls) if response.tool_calls else "нет"))
        print(f"Трассировка: {response.trace_id}")


if __name__ == "__main__":
    main()
