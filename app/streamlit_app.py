from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - UI dependency check
    raise SystemExit("Streamlit is not installed. Run: pip install -r requirements.txt") from exc

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import AgentConfig
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
            "source": chunk.source,
            "title": chunk.title,
            "score": round(chunk.score, 4),
            "chunk_id": chunk.chunk_id,
        }
        for chunk in response.sources
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def show_tool_calls(response) -> None:
    rows = [call.to_dict() for call in response.tool_calls]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Tool calls were not needed for this request.")


def main() -> None:
    st.set_page_config(page_title="MSB lending support agent", layout="wide")
    st.title("MSB lending support agent")

    left, right = st.columns([0.34, 0.66], gap="large")
    with left:
        online = st.toggle("Use GigaChat when available", value=False)
        scenario_name = st.selectbox("Test question", list(SCENARIOS.keys()))
        default_question, default_client, default_channel = SCENARIOS[scenario_name]
        channel = st.selectbox(
            "Channel",
            ["chat_site", "chat_intern", "mobile", "contact_center"],
            index=["chat_site", "chat_intern", "mobile", "contact_center"].index(default_channel),
        )
        client_id = st.text_input("client_id", value=default_client or "")
        question = st.text_area("Question", value=default_question, height=140)
        run = st.button("Run agent", type="primary", use_container_width=True)

    with right:
        if run:
            agent = get_agent(online)
            response = agent.answer(question, client_id=client_id.strip() or None, channel=channel)
            st.subheader("Answer")
            st.write(response.answer)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Intent", response.intent)
            m2.metric("Outcome", response.outcome_type)
            m3.metric("Safety", response.safety_status)
            m4.metric("Escalation", "yes" if response.escalation.required else "no")

            tab_sources, tab_tools, tab_escalation, tab_trace = st.tabs(["Sources", "Tool calls", "Escalation", "Trace"])
            with tab_sources:
                show_sources(response)
            with tab_tools:
                show_tool_calls(response)
            with tab_escalation:
                st.json(response.escalation.to_dict())
            with tab_trace:
                st.code(response.trace_id or "no trace")
                st.json(response.to_dict())
        else:
            st.info("Choose a scenario or edit the request, then run the agent.")


if __name__ == "__main__":
    main()
