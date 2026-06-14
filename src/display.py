from __future__ import annotations

from typing import Any, Dict

from .schemas import AgentResponse, EscalationDecision, RagChunk, ToolCall


INTENT_LABELS = {
    "application_status": "статус заявки",
    "business_rule_rejection": "отказ по бизнес-правилу",
    "communication_policy": "правила коммуникации",
    "documents": "документы",
    "early_repayment": "досрочное погашение",
    "general_info": "общая информация",
    "incompatible_products": "несовместимые продукты",
    "loan_status": "статус кредита",
    "manipulation": "манипуляция",
    "negative_escalation": "негативная эскалация",
    "offtopic": "вне темы",
    "operator_required": "нужен оператор",
    "out_of_scope_info": "вне нормативной базы",
    "out_of_segment": "вне сегмента МСБ",
    "product_info": "информация о продукте",
    "rate_change_rejection": "отказ в изменении ставки",
    "requirements": "требования",
    "restructuring": "реструктуризация",
    "sales_escalation": "продажная эскалация",
    "scoring_secret": "закрытые правила скоринга",
    "self_employed_ineligible": "самозанятый не подходит",
    "suspicious_prompt_injection": "подозрение на инъекцию промпта",
    "suspicious_third_party": "подозрение на запрос третьего лица",
    "third_party_data": "данные третьего лица",
}

OUTCOME_LABELS = {
    "calculation": "расчёт",
    "clarification": "нужно уточнение",
    "escalation": "эскалация",
    "info": "информационный ответ",
    "rejection": "отказ",
}

SAFETY_LABELS = {
    "blocked": "заблокировано",
    "needs_identification": "нужна идентификация",
    "ok": "безопасно",
}

CHANNEL_LABELS = {
    "chat_site": "чат на сайте",
    "chat_intern": "внутренний чат",
    "mobile": "мобильное приложение",
    "contact_center": "контакт-центр",
}

TOOL_LABELS = {
    "get_client_context": "получение контекста клиента",
}

TRIGGER_LABELS = {
    "negative": "негатив клиента",
    "operator": "оператор",
    "operator_required": "запрошен оператор",
    "routing": "маршрутизация",
    "sales_intent": "продажный интерес",
    "security": "безопасность",
}

PRIORITY_LABELS = {
    "high": "высокий",
    "normal": "обычный",
}

REASON_LABELS = {
    "business_start_overdraft_conflict": "конфликт Бизнес-Старта и овердрафта",
    "offtopic": "запрос вне темы",
    "prompt_injection": "инъекция промпта",
    "rate_change_requires_restructuring": "изменение ставки требует реструктуризации",
    "repeat_refinance_not_self_service": "повторное рефинансирование не решается в самообслуживании",
    "request_exception": "исключение из запроса",
    "scoring_secret": "закрытые правила скоринга",
    "self_employed_requirements": "ограничения для самозанятых",
    "stop_factor": "стоп-фактор",
    "third_party_data": "данные третьего лица",
}


def label_intent(value: str) -> str:
    return INTENT_LABELS.get(value, value)


def label_outcome(value: str) -> str:
    return OUTCOME_LABELS.get(value, value)


def label_safety(value: str) -> str:
    return SAFETY_LABELS.get(value, value)


def label_channel(value: str) -> str:
    return CHANNEL_LABELS.get(value, value)


def label_tool(value: str) -> str:
    return TOOL_LABELS.get(value, value)


def label_trigger(value: str | None) -> str | None:
    if value is None:
        return None
    return TRIGGER_LABELS.get(value, value)


def label_priority(value: str) -> str:
    return PRIORITY_LABELS.get(value, value)


def label_reason(value: str | None) -> str | None:
    if value is None:
        return None
    return REASON_LABELS.get(value, value)


def localized_source(chunk: RagChunk) -> Dict[str, Any]:
    return {
        "id_документа": chunk.document_id,
        "id_фрагмента": chunk.chunk_id,
        "источник": chunk.source,
        "заголовок": chunk.title,
        "текст": chunk.text,
        "оценка": chunk.score,
    }


def localized_args(args: Dict[str, Any]) -> Dict[str, Any]:
    field_labels = {
        "client_id": "id_клиента",
    }
    return {field_labels.get(key, key): value for key, value in args.items()}


def localized_tool_call(call: ToolCall) -> Dict[str, Any]:
    return {
        "инструмент": label_tool(call.name),
        "аргументы": localized_args(call.args),
        "краткий_результат": call.result_summary,
    }


def localized_escalation(escalation: EscalationDecision) -> Dict[str, Any]:
    return {
        "требуется": "да" if escalation.required else "нет",
        "триггер": label_trigger(escalation.trigger),
        "приоритет": label_priority(escalation.priority),
        "причина": label_reason(escalation.reason),
        "id_тикета": escalation.ticket_id,
    }


def localized_response(response: AgentResponse) -> Dict[str, Any]:
    return {
        "ответ": response.answer,
        "тип_результата": label_outcome(response.outcome_type),
        "интент": label_intent(response.intent),
        "безопасность": label_safety(response.safety_status),
        "источники": [localized_source(chunk) for chunk in response.sources],
        "вызовы_инструментов": [localized_tool_call(call) for call in response.tool_calls],
        "эскалация": localized_escalation(response.escalation),
        "id_трассировки": response.trace_id,
    }
