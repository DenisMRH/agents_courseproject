from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - проверка UI-зависимости
    raise SystemExit("Streamlit не установлен. Выполните: pip install -r requirements.txt") from exc

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import AgentConfig
from src.display import (
    CHANNEL_LABELS,
    label_channel,
    label_intent,
    label_outcome,
    label_safety,
    label_tool,
    localized_args,
    localized_escalation,
    localized_response,
)
from src.workflow import SupportAgent


SCENARIOS = {
    "Общий вопрос по продуктам": ("Какие кредиты вы предлагаете малому бизнесу?", None, "chat_site"),
    "Статус заявки клиента": ("Что с моей заявкой на кредит?", "C-000002", "chat_intern"),
    "Конфликт регламентов": ("У меня счёт открыт 4 месяца. Могу ли я оформить Бизнес-Старт?", "C-000012", "chat_intern"),
    "Продажная эскалация": ("Хочу подать заявку на Бизнес-Оборот, свяжитесь со мной.", "C-000001", "chat_intern"),
    "Негативная эскалация": ("Ваш банк опять тянет время, позовите оператора, я буду писать жалобу.", "C-000002", "chat_intern"),
}


@st.cache_resource(show_spinner=False)
def get_agent(online: bool) -> SupportAgent:
    return SupportAgent(AgentConfig.from_env(force_local=not online))


def show_sources(response) -> None:
    rows = [
        {
            "Источник": chunk.source,
            "Заголовок": chunk.title,
            "Оценка": round(chunk.score, 4),
            "ID фрагмента": chunk.chunk_id,
        }
        for chunk in response.sources
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def show_tool_calls(response) -> None:
    rows = [
        {
            "Инструмент": label_tool(call.name),
            "Аргументы": json.dumps(localized_args(call.args), ensure_ascii=False),
            "Краткий результат": call.result_summary,
        }
        for call in response.tool_calls
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Для этого запроса вызовы инструментов не потребовались.")


def main() -> None:
    st.set_page_config(page_title="Агент поддержки кредитования МСБ", layout="wide")
    st.title("Агент поддержки кредитования МСБ")

    left, right = st.columns([0.34, 0.66], gap="large")
    with left:
        online = st.toggle("Использовать GigaChat, если доступен", value=False)
        scenario_name = st.selectbox("Тестовый вопрос", list(SCENARIOS.keys()))
        default_question, default_client, default_channel = SCENARIOS[scenario_name]
        channel = st.selectbox(
            "Канал",
            list(CHANNEL_LABELS.keys()),
            index=["chat_site", "chat_intern", "mobile", "contact_center"].index(default_channel),
            format_func=label_channel,
        )
        client_id = st.text_input("ID клиента", value=default_client or "")
        question = st.text_area("Вопрос", value=default_question, height=140)
        run = st.button("Запустить агента", type="primary", use_container_width=True)

    with right:
        if run:
            agent = get_agent(online)
            response = agent.answer(question, client_id=client_id.strip() or None, channel=channel)
            st.subheader("Ответ")
            st.write(response.answer)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Интент", label_intent(response.intent))
            m2.metric("Результат", label_outcome(response.outcome_type))
            m3.metric("Безопасность", label_safety(response.safety_status))
            m4.metric("Эскалация", "да" if response.escalation.required else "нет")

            tab_sources, tab_tools, tab_escalation, tab_trace = st.tabs(["Источники", "Вызовы инструментов", "Эскалация", "Трассировка"])
            with tab_sources:
                show_sources(response)
            with tab_tools:
                show_tool_calls(response)
            with tab_escalation:
                st.json(localized_escalation(response.escalation))
            with tab_trace:
                st.code(response.trace_id or "трассировка отсутствует")
                st.json(localized_response(response))
        else:
            st.info("Выберите сценарий или измените запрос, затем запустите агента.")


if __name__ == "__main__":
    main()
