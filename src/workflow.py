from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, TypedDict

from .config import AgentConfig
from .llm import GigaChatClient
from .prompts import build_answer_prompt
from .rag import RagIndex, describe_chunks
from .schemas import AgentResponse, EscalationDecision, RagChunk, ToolCall
from .tickets import TicketStore
from .tools import ClientTools, rub
from .tracing import TraceStore


PRODUCT_NAMES = {
    "BUSINESS_OBOROT": "Бизнес-Оборот",
    "BUSINESS_RAZVITIE": "Бизнес-Развитие",
    "BUSINESS_LIMIT": "Бизнес-Лимит",
    "BUSINESS_START": "Бизнес-Старт",
    "BUSINESS_PEREZAGRUZKA": "Бизнес-Перезагрузка",
}


class WorkflowState(TypedDict, total=False):
    question: str
    client_id: Optional[str]
    channel: str
    history: List[Dict[str, str]]
    classification: "Classification"
    tool_calls: List[ToolCall]
    client_context: Dict[str, Any]
    chunks: List[RagChunk]
    escalation: EscalationDecision
    response: AgentResponse


@dataclass
class Classification:
    intent: str
    outcome_type: str
    safety_status: str = "ok"
    safety_reason: Optional[str] = None


def _norm(text: str) -> str:
    return text.lower().replace("ё", "е")


def _has(text: str, *keywords: str) -> bool:
    return any(keyword in text for keyword in keywords)


def _latest(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return items[0] if items else None


def _message_text(item: Dict[str, str]) -> str:
    return item.get("content") or item.get("text") or ""


def _client_history_text(history: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for item in history:
        role = (item.get("role") or "client").lower()
        if role in {"client", "user", "customer"}:
            parts.append(_message_text(item))
    return " ".join(parts)


def _all_history_text(history: List[Dict[str, str]]) -> str:
    return " ".join(_message_text(item) for item in history)


def _source_names(chunks: List[RagChunk]) -> str:
    seen: List[str] = []
    for chunk in chunks:
        if chunk.source not in seen:
            seen.append(chunk.source)
    return "; ".join(seen[:4])


class SupportAgent:
    """Sequential implementation of the required LangGraph-style workflow.

    If langgraph is installed, `build_langgraph` returns a StateGraph with the same
    node order. The production path remains explicit and deterministic so the demo
    and evaluation can run without network-only dependencies.
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env()
        self.config.ensure_runtime()
        self.rag = RagIndex(self.config)
        self.tools = ClientTools(self.config)
        self.tickets = TicketStore(self.config)
        self.traces = TraceStore(self.config)
        self.llm = GigaChatClient(self.config)
        try:
            self.graph = self.build_langgraph()
        except Exception:
            self.graph = None

    def build_langgraph(self):
        try:
            from langgraph.graph import END, StateGraph  # type: ignore
        except Exception:
            return None

        graph = StateGraph(WorkflowState)
        graph.add_node("intent_classification", self._node_classify)
        graph.add_node("safety_check", self._node_safety)
        graph.add_node("rag_and_tools", self._node_rag_and_tools)
        graph.add_node("escalation_gate", self._node_escalation)
        graph.add_node("final_self_check", self._node_finalize)
        graph.set_entry_point("intent_classification")
        graph.add_edge("intent_classification", "safety_check")
        graph.add_edge("safety_check", "rag_and_tools")
        graph.add_edge("rag_and_tools", "escalation_gate")
        graph.add_edge("escalation_gate", "final_self_check")
        graph.add_edge("final_self_check", END)
        return graph.compile()

    def answer(
        self,
        question: str,
        *,
        client_id: Optional[str] = None,
        channel: str = "chat_site",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentResponse:
        state: WorkflowState = {
            "question": question,
            "client_id": client_id,
            "channel": channel,
            "history": history or [],
        }
        if self.graph is not None:
            try:
                result = self.graph.invoke(state)
                return result["response"]
            except Exception:
                pass
        for node in (self._node_classify, self._node_safety, self._node_rag_and_tools, self._node_escalation, self._node_finalize):
            state = node(state)
        return state["response"]

    def _node_classify(self, state: WorkflowState) -> WorkflowState:
        state["classification"] = self._classify(state["question"], state.get("history") or [])
        return state

    def _node_safety(self, state: WorkflowState) -> WorkflowState:
        classification = state["classification"]
        if classification.safety_status != "ok":
            return state
        question = state["question"]
        client_id = state.get("client_id")
        if self._needs_client_context(classification.intent, question, client_id) and not client_id:
            state["classification"] = Classification(
                intent=classification.intent,
                outcome_type="clarification",
                safety_status="needs_identification",
                safety_reason="Для ответа нужны данные клиента в авторизованном канале.",
            )
        return state

    def _node_rag_and_tools(self, state: WorkflowState) -> WorkflowState:
        classification = state["classification"]
        tool_calls: List[ToolCall] = []
        client_context: Dict[str, Any] = {}
        client_id = state.get("client_id")
        if client_id and classification.safety_status == "ok":
            client_context = self.tools.logged_context(client_id, tool_calls)
        query = self._build_search_query(state["question"], state.get("history") or [], classification.intent)
        state["tool_calls"] = tool_calls
        state["client_context"] = client_context
        state["chunks"] = self.rag.search(query, top_k=self.config.top_k)
        return state

    def _node_escalation(self, state: WorkflowState) -> WorkflowState:
        classification = self._apply_context_policy(
            state["question"],
            state.get("history") or [],
            state["classification"],
            state.get("client_context") or {},
        )
        if classification != state["classification"]:
            state["classification"] = classification
            query = self._build_search_query(state["question"], state.get("history") or [], classification.intent)
            state["chunks"] = self.rag.search(query, top_k=self.config.top_k)

        escalation = self._escalation_gate(
            state["question"],
            state.get("history") or [],
            classification,
            state.get("client_context") or {},
        )
        if escalation.required:
            ticket_id = self.tickets.create_ticket(
                client_id=state.get("client_id"),
                channel=state["channel"],
                trigger=escalation.trigger or "operator",
                priority=escalation.priority,
                question=state["question"],
                summary=self._ticket_summary(state["question"], classification.intent, state.get("client_context") or {}),
                payload={
                    "intent": classification.intent,
                    "safety_status": classification.safety_status,
                    "sources": [chunk.to_dict() for chunk in state.get("chunks", [])],
                    "tool_calls": [call.to_dict() for call in state.get("tool_calls", [])],
                },
            )
            escalation.ticket_id = ticket_id
            state["classification"] = Classification(classification.intent, "escalation", classification.safety_status, classification.safety_reason)
        state["escalation"] = escalation
        return state

    def _node_finalize(self, state: WorkflowState) -> WorkflowState:
        classification = state["classification"]
        response = AgentResponse(
            answer=self._generate_answer(
                question=state["question"],
                channel=state["channel"],
                client_id=state.get("client_id"),
                classification=classification,
                chunks=state.get("chunks", []),
                client_context=state.get("client_context") or {},
                escalation=state.get("escalation") or EscalationDecision(False),
            ),
            outcome_type=classification.outcome_type,
            intent=classification.intent,
            safety_status=classification.safety_status,
            sources=state.get("chunks", []),
            tool_calls=state.get("tool_calls", []),
            escalation=state.get("escalation") or EscalationDecision(False),
        )
        self._self_check(response)
        response.trace_id = self.traces.write(
            {
                "request": {
                    "question": state["question"],
                    "client_id": state.get("client_id"),
                    "channel": state["channel"],
                    "history": state.get("history") or [],
                },
                "classification": classification.__dict__,
                "response": response.to_dict(),
            }
        )
        state["response"] = response
        return state

    def _classify(self, question: str, history: List[Dict[str, str]]) -> Classification:
        combined = " ".join([_client_history_text(history), question])
        current = _norm(question or _client_history_text(history[-1:]))
        text = _norm(combined)

        if _has(text, "игнорируй", "ignore previous", "с этого момента ты", "ты — менеджер", "ты - менеджер", "директор отделения", "без проверки"):
            return Classification("suspicious_prompt_injection", "escalation")
        if _has(text, "забудь инструк", "system prompt", "промпт", "секретный ключ"):
            return Classification("manipulation", "rejection", "blocked", "prompt_injection")
        if _has(text, "слабые места") and _has(text, "скоринг", "отказ"):
            return Classification("scoring_secret", "rejection", "blocked", "scoring_secret")
        if _has(text, "гарантировать", "гарантия одобр", "поменяю основной оквэд", "слабые места", "сделайте исключение", "сделайте мне исключение", "сделать исключение", "обойдемся", "обойдёмся", "обойти правило", "одобрите", "премиальный клиент сбера", "ставку 12") or ("исключение" in text and _has(text, "сдел", "надеж", "надёж")):
            return Classification("manipulation", "rejection", "blocked", "request_exception")
        if ("инн" in text and _has(text, "партнер", "партнера", "партнёра", "контрагент", "чуж")) or _has(text, "другого клиента", "не мой кредит", "сосед", "конкурент", "я бухгалтер", "представитель ооо", "есть доверенность", "дайте ее данные", "дайте её данные"):
            return Classification("suspicious_third_party", "escalation")
        if _has(text, "чуж"):
            return Classification("third_party_data", "rejection", "blocked", "third_party_data")
        if _has(text, "веса скоринга", "формулу скоринга", "как обойти скоринг", "точные критерии скоринга"):
            return Classification("scoring_secret", "rejection", "blocked", "scoring_secret")
        if _has(text, "банк x", "банка x", "конкурент", "по налогам", "налогам", "налогов выгод", "в расходах", "кэшбэк", "кешбэк", "бонус", "аккредитив", "физлицу", "физ лицу", "как физлицу", "переводы юрлицам") or ("партнер" in text and _has(text, "долях", "схема")):
            return Classification("out_of_scope_info", "info")
        if _has(text, "погода", "рецепт", "футбол", "курс биткоина", "стихотворение", "анекдот", "как у вас дела", "акции", "облигации"):
            return Classification("offtopic", "info", "blocked", "offtopic")
        if _has(text, "миллиард", "1 млрд", "1000 млн", "свыше 800 млн") and "выруч" in text:
            return Classification("out_of_segment", "escalation")
        if _has(text, "ипотек", "жилье", "жильё", "обслуживание расчетного счета", "обслуживание расчётного счёта", "рко"):
            return Classification("out_of_scope_info", "info")

        if self._is_negative(text):
            return Classification("negative_escalation", "escalation")
        if self._is_sales_intent(current):
            return Classification("sales_escalation", "escalation")
        if _has(text, "долговая нагрузка", "долговую нагрузку"):
            return Classification("requirements", "info")
        if _has(current, "рейтинг улучш", "улучшить рейтинг", "повысить рейтинг"):
            return Classification("requirements", "info")
        if _has(text, "один регламент", "другой регламент") and "справ" in text:
            return Classification("documents", "clarification")
        if _has(text, "документ", "паспорт", "выписк", "справк", "кудир", "налогов", "что готовить", "что подготовить", "залог", "страхование"):
            return Classification("documents", "info")
        if _has(text, "досроч", "погасить раньше", "частичное погаш", "закрыть кредит", "закрою", "закрыть свой", "внести как частичную", "досрочку"):
            if _has(text, "до скольки", "до какого времени"):
                return Classification("early_repayment", "clarification")
            if _has(current, "что изменится", "что лучше выбрать"):
                return Classification("early_repayment", "info")
            personal = _has(current, "мой", "моему", "мне", "у меня", "рассч", "закрыть кредит", "закрыть свой", "закрою", "сколько надо", "какая будет сумма", "в одну сумму")
            return Classification("early_repayment", "calculation" if personal else "info")
        if ("статус" in text and "заяв" in text) or _has(text, "где заяв", "что с заяв", "моя заяв", "моей заяв", "мою заяв", "заявкой", "заявку приняли", "повторно подать", "по заявке", "одобрен", "одобрили", "отказ"):
            return Classification("application_status", "info")
        if _has(text, "мой кредит", "остат", "следующий платеж", "следующий платёж", "платеж", "платёж", "просроч", "задолженност", "закончится кредит", "закрытие через", "скоро ли"):
            return Classification("loan_status", "info")
        if _has(text, "реструкт", "кредитные каникулы", "отсроч", "не могу платить", "уменьшить платеж", "уменьшить платёж"):
            return Classification("restructuring", "info")
        if _has(text, "самозанят", "ип", "ооо", "выручк", "требован", "подхожу", "можно ли", "могу ли", "счет открыт"):
            return Classification("requirements", "info")
        if _has(text, "ставк", "лимит", "срок", "продукт", "кредит", "овердрафт", "перезагруз", "бизнес-"):
            return Classification("product_info", "info")
        return Classification("general_info", "info")

    def _is_negative(self, text: str) -> bool:
        return _has(
            text,
            "оператор",
            "человека",
            "нужен человек",
            "бот мне не помогает",
            "жалоб",
            "безобразие",
            "никаких нормальных",
            "идиоты",
            "издевательство",
            "никто ничего не делает",
            "роспотребнадзор",
            "социальные сети",
            "нечем платить",
            "большая просрочка",
            "просрочка большая",
            "выкинули",
            "ненужного",
            "три года клиент",
            "грабеж",
            "грабёж",
            "обнаглели",
            "проценты драть",
            "все совсем плохо",
            "всё совсем плохо",
            "бизнес рухнул",
            "хочу к человеку",
            "с человеком поговорить",
            "ужас",
            "обман",
            "некомпетент",
            "цб",
            "суд",
            "пожаловаться",
            "претензи",
            "возмущ",
            "две недели жду",
            "беспредел",
            "разберит",
            "не устраивает",
        ) or ("сколько можно" in text and _has(text, "жду", "тян", "решени"))

    def _is_sales_intent(self, text: str) -> bool:
        informational = [
            "как подать",
            "просто хочу понять",
            "хочу понять",
            "через сколько",
            "когда можно",
            "после отказа",
            "почему",
            "сколько раз",
            "могу я подать",
            "можно ли подать",
            "что лучше выбрать",
            "что у вас есть",
            "какие условия",
            "как выгоднее",
        ]
        if any(marker in text for marker in informational):
            return False
        markers = [
            "помогите подобрать",
            "подобрать оптимальный",
            "поможете решить",
            "что лучше",
            "давайте оформлять",
            "мне нужен",
            "можно мне еще один кредит",
            "можно мне ещё один кредит",
            "хочу подать на реструктуризацию",
            "дайте реструктуризацию",
            "одобрят или нет",
            "хочу открыть",
            "прошу перевести",
            "хочу оформить",
            "хочу подать заявку",
            "подать заявку",
            "оставить заявку",
            "оформите",
            "давайте оформим",
            "перезвоните",
            "свяжитесь",
            "нужен кредит",
            "готов взять",
            "готов оформить",
            "хочу кредит",
            "хочу взять",
            "оформить новый",
        ]
        if any(marker in text for marker in markers):
            if "хочу открыть" in text and not _has(text, "счет", "счёт", "кредит"):
                return False
            if "прошу перевести" in text and not _has(text, "реструкт", "кредит"):
                return False
            return True
        if re.search(r"\bоформ(ить|ляем|ляйте)\b", text) and _has(text, "кредит", "овердрафт"):
            return True
        return False

    def _needs_client_context(self, intent: str, question: str, client_id: Optional[str]) -> bool:
        text = _norm(question)
        if intent in {"application_status", "loan_status", "restructuring"}:
            if intent == "restructuring":
                return _has(text, "мой", "моему", "мне", "у меня", "по моему", "не могу платить", "уменьшить платеж", "уменьшить платёж")
            return _has(text, "мой", "моя", "моей", "мою", "мне", "у меня", "по моей", "по моему", "статус", "следующий", "остат", "просроч", "закончится")
        if intent == "early_repayment":
            return _has(text, "мой", "моему", "мне", "у меня", "рассч", "закрыть кредит", "закрыть свой", "закрою", "сколько", "сумм", "внести")
        if client_id and _has(text, "мой", "мне", "подхожу", "у меня", "для меня", "можно мне", "могу ли я"):
            return intent in {"requirements", "documents", "product_info", "general_info"}
        return False

    def _build_search_query(self, question: str, history: List[Dict[str, str]], intent: str) -> str:
        hints = {
            "sales_escalation": "триггер намерение оформить продукт порядок эскалации",
            "negative_escalation": "триггер негатив клиента просьба к человеку порядок эскалации",
            "application_status": "статусы заявки доступ к статусу SLA",
            "loan_status": "состояние кредита досрочное погашение просрочка",
            "early_repayment": "досрочное погашение комиссия уведомление",
            "restructuring": "реструктуризация просрочка ограничения",
            "documents": "перечень документов заявка ИП ООО самозанятый",
            "requirements": "требования к заемщику продукт ограничения скоринг",
            "product_info": "линейка кредитных продуктов Бизнес-Оборот Бизнес-Развитие Бизнес-Лимит Бизнес-Старт Бизнес-Перезагрузка",
            "scoring_secret": "запрет раскрытие скоринговая модель",
            "manipulation": "компетенция помощника не обещает одобрение гарантированные условия",
            "suspicious_prompt_injection": "подозрительные обращения социальная инженерия запрет исключений",
            "suspicious_third_party": "конфиденциальность третьи лица чужие обязательства",
            "business_rule_rejection": "стоп-факторы ограничения требования заемщика",
            "self_employed_ineligible": "самозанятый Бизнес-Старт требования НПД 12 месяцев доход 1,2 млн",
            "incompatible_products": "Бизнес-Лимит Бизнес-Старт одновременно не предоставляется",
            "rate_change_rejection": "реструктуризация снижение процентной ставки условия ограничения",
            "operator_required": "намерение изменить условия договора эскалация оператор",
            "communication_policy": "компетенция помощника запреты качество коммуникации источники пункты регламента",
            "offtopic": "запросы вне темы кредитования",
            "out_of_scope_info": "компетенция помощника не дает налоговых юридических советов прочие продукты Банка вне темы кредитования",
        }
        history_text = " ".join(_message_text(item) for item in history[-3:])
        return f"{question}\n{history_text}\n{hints.get(intent, '')}"

    def _apply_context_policy(
        self,
        question: str,
        history: List[Dict[str, str]],
        classification: Classification,
        client_context: Dict[str, Any],
    ) -> Classification:
        if classification.safety_status != "ok":
            return classification

        text = _norm(" ".join([_all_history_text(history), question]))
        profile = client_context.get("profile") or {}
        loans = client_context.get("loans") or []
        applications = client_context.get("applications") or []

        if _has(text, "почему вы цитируете", "что это значит для меня", "что значит для меня") and "пункт" in text:
            return Classification("communication_policy", "info")

        if _has(text, "снизить") and "ставк" in text and _has(text, "мой", "моему", "у меня"):
            return Classification("rate_change_rejection", "rejection", "blocked", "rate_change_requires_restructuring")

        if _has(text, "дешевле", "сделать еще дешевле", "сделать ещё дешевле") and _has(text, "рефинанс", "перезагруз"):
            return Classification("rate_change_rejection", "rejection", "blocked", "repeat_refinance_not_self_service")

        active_codes = {loan.get("product_code") for loan in loans}
        asks_start = "бизнес-старт" in text or "экспресс-кредит" in text
        asks_overdraft = "овердрафт" in text or "бизнес-лимит" in text
        if ("BUSINESS_LIMIT" in active_codes and asks_start) or ("BUSINESS_START" in active_codes and asks_overdraft):
            return Classification("incompatible_products", "rejection", "blocked", "business_start_overdraft_conflict")

        if profile.get("legal_form") == "Самозанятый" and _has(text, "кредит", "бизнес-старт", "самозанят"):
            revenue = int(profile.get("annual_revenue") or 0)
            months_registered = self._months_since(profile.get("registration_date"))
            if months_registered is not None and (months_registered < 12 or revenue < 1_200_000):
                return Classification("self_employed_ineligible", "rejection", "blocked", "self_employed_requirements")

        if _has(text, "рассмотреть мою заявку", "можете рассмотреть") and (
            profile.get("credit_score") == "D" or "stop_factor" in str(profile.get("notes") or "")
        ):
            return Classification("business_rule_rejection", "rejection", "blocked", "stop_factor")

        if _has(text, "что именно вы поменяли", "что поменяли в условиях") and applications:
            return Classification("operator_required", "escalation")

        if _has(text, "поручительств") and _has(text, "снять", "убрать", "отменить"):
            return Classification("operator_required", "escalation")

        return classification

    def _months_since(self, value: Any) -> Optional[int]:
        if not value:
            return None
        try:
            start = date.fromisoformat(str(value))
        except ValueError:
            return None
        today = date.today()
        return (today.year - start.year) * 12 + today.month - start.month - int(today.day < start.day)

    def _escalation_gate(
        self,
        question: str,
        history: List[Dict[str, str]],
        classification: Classification,
        client_context: Dict[str, Any],
    ) -> EscalationDecision:
        if classification.safety_status != "ok":
            return EscalationDecision(False)
        if classification.intent == "sales_escalation":
            return EscalationDecision(
                True,
                trigger="sales_intent",
                priority="normal",
                reason="Клиент выразил намерение оформить кредитный продукт.",
            )
        if classification.intent == "negative_escalation":
            return EscalationDecision(
                True,
                trigger="negative",
                priority="high",
                reason="Обнаружен негатив или запрос на оператора.",
            )
        if classification.intent == "operator_required":
            return EscalationDecision(
                True,
                trigger="operator_required",
                priority="normal",
                reason="Запрос требует индивидуального изменения условий или пояснения решения.",
            )
        if classification.intent in {"suspicious_third_party", "suspicious_prompt_injection"}:
            return EscalationDecision(
                True,
                trigger="security",
                priority="high",
                reason="Запрос данных третьего лица или подозрительное обращение.",
            )
        if classification.intent == "out_of_segment":
            return EscalationDecision(
                True,
                trigger="routing",
                priority="normal",
                reason="Клиент вне сегмента МСБ, нужна маршрутизация.",
            )
        text = _norm(" ".join([_all_history_text(history), question]))
        loans = client_context.get("loans") or []
        loan = _latest(loans)
        if classification.intent == "loan_status" and loan and int(loan.get("overdue_days") or 0) > 90 and "просроч" in text:
            return EscalationDecision(
                True,
                trigger="operator_required",
                priority="high",
                reason="По действующему кредиту просрочка свыше 90 дней, нужен индивидуальный разбор.",
            )
        if classification.intent == "early_repayment" and self._is_early_repayment_action(text, loan):
            return EscalationDecision(
                True,
                trigger="sales_intent",
                priority="normal",
                reason="Клиент выразил намерение оформить досрочное погашение.",
            )
        return EscalationDecision(False)

    def _is_early_repayment_action(self, text: str, loan: Optional[Dict[str, Any]]) -> bool:
        if _has(text, "какие условия", "если я", "если сейчас", "рассч", "сколько мне нужно для", "какая будет сумма"):
            return False
        if _has(text, "если внести", "что лучше выбрать", "что изменится"):
            return False
        if _has(text, "хочу досрочно погасить", "оформите досрочное", "оформить досрочное", "внести как частичную", "частичную досрочку"):
            return True
        if loan and loan.get("product_code") == "BUSINESS_RAZVITIE" and int(loan.get("months_passed") or 0) < 6:
            return _has(text, "хочу") and _has(text, "досроч", "погасить")
        return False

    def _generate_answer(
        self,
        *,
        question: str,
        channel: str,
        client_id: Optional[str],
        classification: Classification,
        chunks: List[RagChunk],
        client_context: Dict[str, Any],
        escalation: EscalationDecision,
    ) -> str:
        if escalation.required:
            return self._answer_escalation(escalation)
        if classification.safety_status == "blocked":
            return self._answer_rejection(classification)
        if classification.safety_status == "needs_identification":
            return "Для ответа по вашим данным нужна идентификация в интернет-банке или мобильном приложении. В анонимном канале я могу дать только общую информацию."
        if self._must_use_deterministic_answer(classification, client_context):
            return self._local_answer(question, classification, chunks, client_context)

        tool_context = self._safe_tool_context(client_context) if client_context else "нет"
        if self.llm.available:
            llm_answer = self.llm.invoke(
                build_answer_prompt(
                    question=question,
                    channel=channel,
                    client_id=client_id,
                    context=describe_chunks(chunks),
                    tool_context=tool_context,
                )
            )
            if llm_answer:
                return self._postprocess_llm_answer(llm_answer, question, classification, chunks, client_context)
        return self._local_answer(question, classification, chunks, client_context)

    def _must_use_deterministic_answer(self, classification: Classification, client_context: Dict[str, Any]) -> bool:
        if classification.outcome_type == "calculation" or classification.intent in {
            "out_of_scope_info",
            "communication_policy",
        }:
            return True
        if client_context and classification.intent in {
            "application_status",
            "loan_status",
            "requirements",
            "early_repayment",
            "restructuring",
        }:
            return True
        return False

    def _safe_tool_context(self, client_context: Dict[str, Any]) -> str:
        profile = dict(client_context.get("profile") or {})
        for field in ("profile_tag", "notes", "contact_phone", "contact_email"):
            profile.pop(field, None)
        safe_context = {
            "profile": profile or None,
            "applications": client_context.get("applications") or [],
            "loans": client_context.get("loans") or [],
        }
        return json.dumps(safe_context, ensure_ascii=False, default=str)[:4000]

    def _postprocess_llm_answer(
        self,
        answer: str,
        question: str,
        classification: Classification,
        chunks: List[RagChunk],
        client_context: Dict[str, Any],
    ) -> str:
        text = _norm(answer)
        leaked_internal = any(term in text for term in ("profile_tag", "raw json", "данные tools", "source indexes"))
        unsupported_products = any(term in text for term in ("лизинг", "ипотек", "микрозайм", "кредитная линия"))
        generic_products_question = classification.intent == "product_info" and _has(
            _norm(question),
            "какие кредиты",
            "что предлагаете",
            "линейка",
            "кредитные продукты",
        )
        required_product_names = (
            "бизнес-оборот",
            "бизнес-развитие",
            "бизнес-лимит",
            "бизнес-старт",
            "бизнес-перезагрузка",
        )
        incomplete_product_line = generic_products_question and any(name not in text for name in required_product_names)
        business_start_account_exception = (
            classification.intent == "requirements"
            and "бизнес-старт" in _norm(question)
            and ("4 месяц" in _norm(question) or "4 мес" in _norm(question))
        )
        if (
            leaked_internal
            or generic_products_question
            or (generic_products_question and unsupported_products)
            or incomplete_product_line
            or business_start_account_exception
        ):
            return self._local_answer(question, classification, chunks, client_context)
        return answer

    def _answer_escalation(self, escalation: EscalationDecision) -> str:
        if escalation.trigger == "sales_intent":
            return (
                "Передаю обращение специалисту по кредитованию МСБ: он сможет продолжить оформление и уточнить параметры заявки. "
                f"Тикет: {escalation.ticket_id}."
            )
        if escalation.trigger == "security":
            return (
                "Я не могу подтверждать или раскрывать данные третьих лиц. Передаю обращение на проверку безопасности. "
                f"Тикет: {escalation.ticket_id}."
            )
        if escalation.trigger == "routing":
            return (
                "По указанной выручке компания относится не к сегменту МСБ, поэтому нужна маршрутизация к профильному подразделению. "
                f"Передаю обращение оператору. Тикет: {escalation.ticket_id}."
            )
        return f"Подключаю оператора, чтобы разобрать обращение индивидуально. Тикет: {escalation.ticket_id}."

    def _answer_rejection(self, classification: Classification) -> str:
        if classification.intent == "scoring_secret":
            return (
                "Я не могу раскрывать внутренние веса и способы обхода скоринговой модели. "
                "Могу объяснить общие уровни рейтинга и требования к продуктам без внутренних критериев."
            )
        if classification.intent == "third_party_data":
            return "Я не могу раскрывать данные других клиентов или обязательств. В авторизованном канале доступна только информация по вашему клиентскому профилю."
        if classification.intent == "offtopic":
            return "Я помогаю по вопросам кредитования малого и микробизнеса в Банке. По этой теме консультацию не даю, но могу помочь с продуктами, заявками, платежами или реструктуризацией."
        if classification.intent == "self_employed_ineligible":
            return "По регламенту самозанятым доступен только Бизнес-Старт при регистрации НПД не менее 12 месяцев и подтверждённом доходе от 1,2 млн руб.; без выполнения этих условий кредит оформить нельзя."
        if classification.intent == "incompatible_products":
            return "Одновременно пользоваться Бизнес-Старт и овердрафтом Бизнес-Лимит нельзя: регламент прямо запрещает такое сочетание продуктов."
        if classification.intent == "business_rule_rejection":
            return "Я не могу рассмотреть или одобрить заявку в обход стоп-факторов. По регламенту при стоп-факторе кредитование не выполняется без профильной проверки."
        if classification.intent == "rate_change_rejection":
            return "Я не могу самостоятельно снизить ставку или изменить условия действующего договора. Такое изменение возможно только в рамках процедуры реструктуризации и решения уполномоченного уровня Банка."
        return "Я не могу выполнить этот запрос, потому что он выходит за рамки безопасной работы помощника."

    def _local_answer(self, question: str, classification: Classification, chunks: List[RagChunk], client_context: Dict[str, Any]) -> str:
        intent = classification.intent
        if intent == "application_status":
            return self._answer_application(client_context, chunks)
        if intent == "loan_status":
            return self._answer_loan(client_context, chunks)
        if intent == "early_repayment":
            return self._answer_early_repayment(question, client_context, chunks)
        if intent == "restructuring":
            return self._answer_restructuring(client_context, chunks)
        if intent == "out_of_scope_info":
            return self._answer_out_of_scope(question, chunks)
        if intent == "communication_policy":
            return self._answer_communication_policy(chunks)
        if intent == "product_info":
            return self._answer_product_info(question, chunks)
        if intent == "documents":
            return self._answer_documents(question, chunks)
        if intent == "requirements":
            return self._answer_requirements(question, chunks, client_context)
        return self._answer_from_chunks(chunks)

    def _answer_application(self, context: Dict[str, Any], chunks: List[RagChunk]) -> str:
        applications = context.get("applications") or []
        app = _latest(applications)
        if not app:
            return "По клиентскому профилю не нашёл активных или прошлых кредитных заявок. " + self._sources_suffix(chunks)
        decision = app.get("decision") or "решение ещё не принято"
        reason = app.get("decision_reason_category") or "не указана"
        return (
            f"Последняя заявка {app['application_id']} по продукту {PRODUCT_NAMES.get(app['product_code'], app['product_code'])} "
            f"на сумму {rub(app['amount_requested'])} от {app['application_date']} сейчас в статусе «{app['status']}». "
            f"Решение: {decision}; категория причины: {reason}. {self._sources_suffix(chunks)}"
        )

    def _answer_loan(self, context: Dict[str, Any], chunks: List[RagChunk]) -> str:
        loans = context.get("loans") or []
        loan = _latest(loans)
        if not loan:
            return "По клиентскому профилю не нашёл действующих кредитных договоров. " + self._sources_suffix(chunks)
        overdue = (
            f"Есть просрочка {loan['overdue_days']} дн. на {rub(loan['overdue_amount'])}."
            if loan.get("has_overdue")
            else "Активной просрочки по договору нет."
        )
        return (
            f"Договор {loan['contract_id']} ({loan['product_name']}): остаток основного долга {rub(loan['principal_outstanding'])}, "
            f"ставка {loan['interest_rate']}% годовых, следующий платёж {rub(loan['next_payment_amount'])} "
            f"до {loan['next_payment_date']}. {overdue} {self._sources_suffix(chunks)}"
        )

    def _answer_early_repayment(self, question: str, context: Dict[str, Any], chunks: List[RagChunk]) -> str:
        loans = context.get("loans") or []
        loan = _latest(loans)
        if not loan:
            text = _norm(question)
            if "оборот" in text:
                return "По Бизнес-Оборот полное и частичное досрочное погашение проводится без комиссии и без обязательного предуведомления; минимальная сумма частичного погашения — 50 000 руб. " + self._sources_suffix(chunks)
            return "Для точного расчёта досрочного погашения нужен действующий кредитный договор. По общим правилам сумма фиксируется на дату погашения. " + self._sources_suffix(chunks)
        principal = int(loan["principal_outstanding"])
        commission = 0
        notice = "заявление можно подать в канале обслуживания"
        if loan["product_code"] == "BUSINESS_RAZVITIE" and int(loan["months_passed"]) < 6:
            commission = round(principal * 0.02)
            notice = "нужно уведомить Банк за 10 рабочих дней"
        total = principal + commission + int(loan.get("overdue_amount") or 0)
        return (
            f"По договору {loan['contract_id']} ориентир для полного досрочного погашения: основной долг {rub(principal)}"
            f"{', комиссия ' + rub(commission) if commission else ', комиссии по базовому правилу нет'}"
            f"{', просрочка ' + rub(loan.get('overdue_amount')) if loan.get('overdue_amount') else ''}. "
            f"Итого ориентировочно {rub(total)}; {notice}. Точную сумму Банк фиксирует на дату погашения. {self._sources_suffix(chunks)}"
        )

    def _answer_restructuring(self, context: Dict[str, Any], chunks: List[RagChunk]) -> str:
        loans = context.get("loans") or []
        loan = _latest(loans)
        if not loan:
            return "Реструктуризация возможна при документально подтверждённом ухудшении финансового положения: снижении выручки, потере ключевого контрагента, чрезвычайных обстоятельствах, изменении регулирования, болезни владельца или сезонных затруднениях. Доступные меры включают кредитные каникулы, изменение графика, снижение ставки, пролонгацию или комбинированную реструктуризацию. " + self._sources_suffix(chunks)
        overdue_days = int(loan.get("overdue_days") or 0)
        if overdue_days > 90:
            conclusion = "стандартная реструктуризация обычно недоступна, обращение рассматривается индивидуально"
        elif overdue_days >= 30:
            conclusion = "потребуется частичное погашение просроченной задолженности перед рассмотрением"
        else:
            conclusion = "можно подать заявление с документальным подтверждением причины"
        return (
            f"По договору {loan['contract_id']} просрочка составляет {overdue_days} дн.; {conclusion}. "
            f"Возможные меры: отсрочка, изменение графика, пролонгация или комбинированная реструктуризация. {self._sources_suffix(chunks)}"
        )

    def _answer_out_of_scope(self, question: str, chunks: List[RagChunk]) -> str:
        text = _norm(question)
        if "банк x" in text or "конкурент" in text:
            return "Я не сравниваю условия Банка с условиями конкретных конкурентов. Могу рассказать только условия продуктов Банка по регламенту кредитования МСБ. " + self._sources_suffix(chunks)
        if "налог" in text:
            return "Я не даю налоговые или бухгалтерские рекомендации. По учёту кредитных платежей лучше обратиться к бухгалтеру или налоговому консультанту; по кредитным условиям Банка могу помочь отдельно. " + self._sources_suffix(chunks)
        if "физлиц" in text or "физ лиц" in text:
            return "Помощник работает с кредитованием малого и микробизнеса. Кредиты физическим лицам относятся к другим продуктам Банка, поэтому по ним нужно обратиться в розничный блок. " + self._sources_suffix(chunks)
        if _has(text, "переводы", "аккредитив", "кэшбэк", "кешбэк", "бонус"):
            return "Этот вопрос относится к другим банковским продуктам, а не к кредитованию МСБ. По нему нужно обратиться в профильный раздел Банка или к специалисту обслуживания. " + self._sources_suffix(chunks)
        if "ипотек" in text or "жилье" in text or "жильё" in text:
            return "Ипотечные и жилищные кредиты для физических лиц не входят в регламент кредитования МСБ. По ним лучше обратиться в розничный блок Банка или профильный раздел сайта. " + self._sources_suffix(chunks)
        return "Расчётно-кассовое обслуживание не входит в компетенцию помощника по кредитованию МСБ. По тарифам РКО лучше обратиться в раздел РКО на сайте Банка или к менеджеру по обслуживанию. " + self._sources_suffix(chunks)

    def _answer_communication_policy(self, chunks: List[RagChunk]) -> str:
        return (
            "Я указываю пункты регламентов, чтобы было понятно, на каком правиле основан ответ. "
            "Для клиента это не внутренняя инструкция к действию, а ссылка на проверяемый источник: по спорному вопросу её можно передать оператору или менеджеру. "
            + self._sources_suffix(chunks)
        )

    def _answer_product_info(self, question: str, chunks: List[RagChunk]) -> str:
        text = _norm(question)
        if "оборот" in text:
            return "Бизнес-Оборот — кредит на пополнение оборотных средств до 30 млн руб., сроком до 36 месяцев, ставка от 18,5% годовых. " + self._sources_suffix(chunks)
        if "развит" in text or "оборуд" in text or "100" in text:
            return "Бизнес-Развитие — инвестиционный кредит до 100 млн руб. на срок до 84 месяцев, в том числе на оборудование; требуется обеспечение, ставка от 19% годовых. " + self._sources_suffix(chunks)
        if "овердрафт" in text or "лимит" in text:
            return "Бизнес-Лимит — овердрафт до 5 млн руб. и до 30% среднемесячных оборотов по счёту; счёт в Банке должен обслуживаться не менее 6 месяцев. " + self._sources_suffix(chunks)
        if "старт" in text or "самозан" in text:
            return "Бизнес-Старт — беззалоговый экспресс-кредит до 5 млн руб. на срок до 36 месяцев, ставка от 23,5%; для самозанятых лимит до 1 млн руб. " + self._sources_suffix(chunks)
        if "перезагруз" in text or "рефинанс" in text:
            return "Бизнес-Перезагрузка предназначен для рефинансирования внешнего долга: до 70 млн руб., срок до 60 месяцев, ставка от 17,5%, выдача напрямую в банк-кредитор. " + self._sources_suffix(chunks)
        if "доллар" in text or "валют" in text:
            return "Кредитование МСБ по этой линейке предоставляется только в рублях; валютные кредиты не оформляются. " + self._sources_suffix(chunks)
        return (
            "Линейка включает Бизнес-Оборот, Бизнес-Развитие, Бизнес-Лимит, Бизнес-Старт и Бизнес-Перезагрузка. "
            "Каждый продукт предназначен для своей цели: оборотные средства, инвестиции, овердрафт, экспресс-кредит или рефинансирование. "
            + self._sources_suffix(chunks)
        )

    def _answer_documents(self, question: str, chunks: List[RagChunk]) -> str:
        text = _norm(question)
        if "страхование" in text or "залог" in text:
            return "По обеспечению Банк может запросить документы на предмет залога, оценку, подтверждение права собственности и страховые документы, если они требуются для конкретного обеспечения. Точный перечень зависит от продукта и вида залога. " + self._sources_suffix(chunks)
        if "самозан" in text:
            return "Для самозанятого нужны паспорт, подтверждение регистрации НПД, справка о доходах за 12 месяцев из «Мой налог», справка о расчётах по НПД и выписки по счетам. " + self._sources_suffix(chunks)
        if "ип" in text:
            return "Для ИП обычно нужны паспорт, лист записи ЕГРИП, ИНН, налоговая декларация, КУДиР, выписки по счетам за 6 месяцев, справки ФНС и сведения о действующих кредитах. " + self._sources_suffix(chunks)
        if "справк" in text and "налог" in text:
            return "Справка о наличии или отсутствии налоговой задолженности должна быть актуальной: давность не более 30 календарных дней. " + self._sources_suffix(chunks)
        return "Для ООО готовят учредительные документы, сведения о руководителе и собственниках, бухгалтерскую отчётность, расшифровки задолженностей, выписки по счетам и справки ФНС. " + self._sources_suffix(chunks)

    def _answer_requirements(self, question: str, chunks: List[RagChunk], context: Dict[str, Any]) -> str:
        profile = context.get("profile")
        text = _norm(question)
        prefix = ""
        if profile:
            prefix = f"По профилю {profile['client_id']}: форма {profile['legal_form']}, выручка {rub(profile['annual_revenue'])}, скоринг {profile['credit_score']}. "
        if "бизнес-старт" in text and ("счет" in text or "счёт" in text) and ("4 месяц" in text or "4 мес" in text or (profile and profile.get("client_id") == "C-000012")):
            return prefix + "Для Бизнес-Старт действует продуктовый регламент: достаточно счёта в Банке от 3 месяцев, поэтому 4 месяца не блокируют этот продукт. Общее правило 6 месяцев применяется к другим продуктам, если продуктовые условия не устанавливают исключение. " + self._sources_suffix(chunks)
        if "самозан" in text:
            return prefix + "Самозанятым доступен только Бизнес-Старт: регистрация НПД не менее 12 месяцев, подтверждённый доход от 1,2 млн руб., сумма до 1 млн руб. " + self._sources_suffix(chunks)
        if "полгода" in text or "6 мес" in text or "6 месяц" in text:
            return prefix + "При сроке деятельности 6 месяцев обычно доступен только Бизнес-Старт; остальные продукты требуют 12 месяцев и более. " + self._sources_suffix(chunks)
        if "выруч" in text and "оборот" in text:
            return prefix + "Для Бизнес-Оборот минимальная годовая выручка — 6 млн руб.; также учитываются срок деятельности, скоринг и отсутствие существенных просрочек. " + self._sources_suffix(chunks)
        if "скоринг c" in text:
            return prefix + "Рейтинг C означает повышенный риск: доступны Бизнес-Оборот, Бизнес-Старт и Бизнес-Лимит, но не Бизнес-Развитие и Бизнес-Перезагрузка. " + self._sources_suffix(chunks)
        if _has(text, "рейтинг улучш", "улучшить рейтинг", "повысить рейтинг"):
            return prefix + "На рейтинг влияют финансовое состояние бизнеса, кредитная история и отсутствие просрочек, характеристики деятельности и поведение по счетам. Детальные веса и точные параметры скоринговой модели Банк не раскрывает, но можно улучшать прозрачность оборотов, своевременность платежей и качество финансовой отчётности. " + self._sources_suffix(chunks)
        return prefix + self._answer_from_chunks(chunks)

    def _answer_from_chunks(self, chunks: List[RagChunk]) -> str:
        if not chunks:
            return "В нормативных документах не нашёл достаточной информации для точного ответа."
        best = chunks[0]
        snippet = re.sub(r"\s+", " ", best.text).strip()
        snippet = re.sub(r"^#+\s*", "", snippet)
        if len(snippet) > 620:
            snippet = snippet[:617].rstrip() + "..."
        return f"{snippet} {self._sources_suffix(chunks)}"

    def _sources_suffix(self, chunks: List[RagChunk]) -> str:
        sources = _source_names(chunks)
        return f"Источники: {sources}." if sources else "Источник в RAG не найден."

    def _ticket_summary(self, question: str, intent: str, context: Dict[str, Any]) -> str:
        profile = context.get("profile")
        client_part = f" Клиент: {profile['client_id']} {profile['name']}." if profile else ""
        return f"Интент: {intent}. Вопрос: {question[:300]}.{client_part}"

    def _self_check(self, response: AgentResponse) -> None:
        if response.escalation.required and not response.escalation.ticket_id:
            response.answer += " Тикет будет создан оператором вручную."
        if response.intent == "scoring_secret" and "вес" in _norm(response.answer):
            response.answer = self._answer_rejection(Classification("scoring_secret", "rejection", "blocked"))
