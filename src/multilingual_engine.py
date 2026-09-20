"""Multilingual Support Engine.

Handles language detection, translation, and culturally appropriate responses
for customer service chatbot interactions using Google Gemini with fallback mechanisms.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_src_dir)
load_dotenv(os.path.join(_src_dir, ".env"))
load_dotenv(os.path.join(_project_dir, ".env"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language configurations
SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {"name": "English", "flag": "🇬🇧", "greeting": "Hello! How can I help you?", "direction": "ltr"},
    "es": {"name": "Spanish", "flag": "🇪🇸", "greeting": "¡Hola! ¿Cómo puedo ayudarte?", "direction": "ltr"},
    "fr": {"name": "French", "flag": "🇫🇷", "greeting": "Bonjour! Comment puis-je vous aider?", "direction": "ltr"},
    "de": {"name": "German", "flag": "🇩🇪", "greeting": "Hallo! Wie kann ich Ihnen helfen?", "direction": "ltr"},
    "hi": {"name": "Hindi", "flag": "🇮🇳", "greeting": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?", "direction": "ltr"}
}

# Common words/phrases for language detection
LANGUAGE_INDICATORS: Dict[str, List[str]] = {
    "es": [
        "hola", "gracias", "por favor", "cómo", "qué", "necesito", "ayuda",
        "bueno", "el", "la", "los", "las", "es", "está", "son", "tengo",
        "quiero", "dónde", "cuándo", "porque", "también", "pero", "como",
        "muy", "más", "pedido", "cuenta", "servicio"
    ],
    "fr": [
        "bonjour", "merci", "s'il vous plaît", "comment", "pourquoi", "je",
        "suis", "oui", "non", "le", "la", "les", "un", "une", "est", "sont",
        "avoir", "faire", "quel", "cette", "dans", "pour", "avec", "pas",
        "très", "commande", "compte", "aide"
    ],
    "de": [
        "hallo", "danke", "bitte", "wie", "warum", "ich", "bin", "ja",
        "nein", "der", "die", "das", "ein", "eine", "ist", "sind", "haben",
        "nicht", "und", "aber", "oder", "für", "mit", "kann", "möchte",
        "bestellung", "hilfe", "konto"
    ],
    "hi": [
        "नमस्ते", "धन्यवाद", "कृपया", "कैसे", "क्यों", "मैं", "हूँ", "है",
        "हैं", "क्या", "यह", "वह", "और", "में", "को", "का", "की", "के",
        "से", "पर", "मुझे", "कर", "हो", "था", "मदद", "ऑर्डर", "सहायता"
    ]
}

# Basic translation dictionaries for common chatbot phrases
COMMON_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "How can I help you?": {
        "es": "¿Cómo puedo ayudarte?",
        "fr": "Comment puis-je vous aider ?",
        "de": "Wie kann ich Ihnen helfen?",
        "hi": "मैं आपकी कैसे मदद कर सकता हूँ?"
    },
    "Thank you for contacting us.": {
        "es": "Gracias por contactarnos.",
        "fr": "Merci de nous avoir contactés.",
        "de": "Vielen Dank für Ihre Kontaktaufnahme.",
        "hi": "हमसे संपर्क करने के लिए धन्यवाद।"
    },
    "I don't know.": {
        "es": "No lo sé.",
        "fr": "Je ne sais pas.",
        "de": "Ich weiß es nicht.",
        "hi": "मुझे नहीं पता।"
    },
    "Please wait a moment.": {
        "es": "Por favor, espere un momento.",
        "fr": "Veuillez patienter un instant.",
        "de": "Bitte warten Sie einen Moment.",
        "hi": "कृपया एक क्षण प्रतीक्षा करें।"
    },
    "Can you please provide more details?": {
        "es": "¿Podría proporcionar más detalles, por favor?",
        "fr": "Pouvez-vous fournir plus de détails, s'il vous plaît ?",
        "de": "Können Sie bitte weitere Details angeben?",
        "hi": "क्या आप कृपया अधिक जानकारी दे सकते हैं?"
    },
    "You're welcome!": {
        "es": "¡De nada!",
        "fr": "De rien !",
        "de": "Gern geschehen!",
        "hi": "आपका स्वागत है!"
    },
    "Goodbye! Have a great day.": {
        "es": "¡Adiós! Que tenga un buen día.",
        "fr": "Au revoir ! Passez une excellente journée.",
        "de": "Auf Wiedersehen! Einen schönen Tag noch.",
        "hi": "अलविदा! आपका दिन शुभ हो।"
    },
    "An agent will be with you shortly.": {
        "es": "Un agente estará con usted en breve.",
        "fr": "Un agent sera avec vous sous peu.",
        "de": "Ein Mitarbeiter wird in Kürze bei Ihnen sein.",
        "hi": "एक एजेंट जल्द ही आपके साथ जुड़ेगा।"
    },
    "Your issue has been resolved.": {
        "es": "Su problema ha sido resuelto.",
        "fr": "Votre problème a été résolu.",
        "de": "Ihr Anliegen wurde gelöst.",
        "hi": "आपकी समस्या का समाधान हो गया है।"
    },
    "Please check your email for confirmation.": {
        "es": "Por favor revise su correo electrónico para la confirmación.",
        "fr": "Veuillez vérifier votre e-mail pour confirmation.",
        "de": "Bitte überprüfen Sie Ihre E-Mails zur Bestätigung.",
        "hi": "पुष्टि के लिए कृपया अपना ईमेल देखें।"
    },
    "What is your order number?": {
        "es": "¿Cuál es su número de pedido?",
        "fr": "Quel est votre numéro de commande ?",
        "de": "Wie lautet Ihre Bestellnummer?",
        "hi": "आपका ऑर्डर नंबर क्या है?"
    },
    "I apologize for the inconvenience.": {
        "es": "Pido disculpas por las molestias.",
        "fr": "Je m'excuse pour le désagrément.",
        "de": "Ich entschuldige mich für die Unannehmlichkeiten.",
        "hi": "असुविधा के लिए मुझे खेद है।"
    },
    "Is there anything else I can help you with?": {
        "es": "¿Hay algo más en lo que pueda ayudarle?",
        "fr": "Y a-t-il autre chose que je puisse faire pour vous aider ?",
        "de": "Kann ich Ihnen sonst noch bei etwas helfen?",
        "hi": "क्या कोई और चीज़ है जिसमें मैं आपकी मदद कर सकता हूँ?"
    },
    "Let me find that information for you.": {
        "es": "Permítame encontrar esa información para usted.",
        "fr": "Laissez-moi chercher cette information pour vous.",
        "de": "Lassen Sie mich diese Informationen für Sie heraussuchen.",
        "hi": "मुझे आपके लिए यह जानकारी खोजने दीजिए।"
    },
    "Could you please rephrase your question?": {
        "es": "¿Podría reformular su pregunta, por favor?",
        "fr": "Pourriez-vous reformuler votre question, s'il vous plaît ?",
        "de": "Könnten Sie Ihre Frage bitte umformulieren?",
        "hi": "क्या आप कृपया अपना प्रश्न दोबारा दोहरा सकते हैं?"
    },
    "Our business hours are 9 AM to 5 PM.": {
        "es": "Nuestro horario de atención es de 9:00 a 17:00.",
        "fr": "Nos heures d'ouverture sont de 9h à 17h.",
        "de": "Unsere Geschäftszeiten sind von 9:00 bis 17:00 Uhr.",
        "hi": "हमारे काम के घंटे सुबह 9 बजे से शाम 5 बजे तक हैं।"
    },
    "Thank you for your patience.": {
        "es": "Gracias por su paciencia.",
        "fr": "Merci pour votre patience.",
        "de": "Vielen Dank für Ihre Geduld.",
        "hi": "आपके धैर्य के लिए धन्यवाद।"
    },
    "I am here to help you.": {
        "es": "Estoy aquí para ayudarte.",
        "fr": "Je suis là pour vous aider.",
        "de": "Ich bin hier, um Ihnen zu helfen.",
        "hi": "मैं आपकी मदद के लिए यहाँ हूँ।"
    },
    "Please hold on.": {
        "es": "Por favor, manténgase a la espera.",
        "fr": "Veuillez patienter.",
        "de": "Bitte bleiben Sie dran.",
        "hi": "कृपया बने रहें।"
    },
    "Sorry, I could not find any results.": {
        "es": "Lo siento, no pude encontrar ningún resultado.",
        "fr": "Désolé, je n'ai trouvé aucun résultat.",
        "de": "Entschuldigung, ich konnte keine Ergebnisse finden.",
        "hi": "क्षमा करें, मुझे कोई परिणाम नहीं मिला।"
    }
}


class MultilingualEngine:
    """Handles language detection, translation, and culturally appropriate responses."""

    def __init__(self) -> None:
        """Initialize MultilingualEngine with detection and LLM translation capabilities."""
        self.current_language: str = "en"
        self.has_langdetect: bool = False
        self.has_gemini: bool = False
        self._detect_fn = None
        self._model = None

        # Initialize langdetect if available
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            self._detect_fn = detect
            self.has_langdetect = True
            logger.info("langdetect initialized successfully.")
        except (ImportError, Exception) as e:
            self.has_langdetect = False
            self._detect_fn = None
            logger.info("langdetect not available (%s); using rule-based indicator detection.", e)

        # Initialize Google Gemini for translation if available
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                candidates = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]
                for cand in candidates:
                    try:
                        self._model = genai.GenerativeModel(cand)
                        self.has_gemini = True
                        logger.info(f"Google Gemini translation model initialized with '{cand}'.")
                        break
                    except Exception as ex:
                        logger.warning(f"Translation candidate '{cand}' failed ({ex}); trying next...")
            except (ImportError, Exception) as e:
                self.has_gemini = False
                self._model = None
                logger.info("Gemini translation not initialized (%s); using fallback translation.", e)
        else:
            logger.info("No GOOGLE_API_KEY found; using dictionary and fallback translation.")

    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect the language of input text.

        Tries the langdetect library first, falls back to indicator word matching,
        and defaults to English if no confident match is found.

        Args:
            text: Input text string to detect language for.

        Returns:
            Dict containing language_code, language_name, confidence, and flag.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "language_code": "en",
                "language_name": "English",
                "confidence": 1.0,
                "flag": "🇬🇧"
            }

        # Check for Hindi (Devanagari Unicode block: U+0900 to U+097F)
        if re.search(r"[\u0900-\u097F]", text):
            return {
                "language_code": "hi",
                "language_name": SUPPORTED_LANGUAGES["hi"]["name"],
                "confidence": 0.98,
                "flag": SUPPORTED_LANGUAGES["hi"]["flag"]
            }

        # Primary detection using langdetect library
        if self.has_langdetect and self._detect_fn:
            try:
                detected_code = self._detect_fn(text)
                if detected_code in SUPPORTED_LANGUAGES:
                    return {
                        "language_code": detected_code,
                        "language_name": SUPPORTED_LANGUAGES[detected_code]["name"],
                        "confidence": 0.95,
                        "flag": SUPPORTED_LANGUAGES[detected_code]["flag"]
                    }
            except Exception as e:
                logger.debug("langdetect exception during detection: %s", e)

        # Fallback detection using LANGUAGE_INDICATORS
        clean_text = text.lower()
        words = re.findall(r"\b[\w']+\b", clean_text)
        word_set = set(words)

        scores: Dict[str, int] = {}
        for lang_code, indicators in LANGUAGE_INDICATORS.items():
            match_count = 0
            for indicator in indicators:
                if " " in indicator:
                    if indicator in clean_text:
                        match_count += 2
                elif indicator in word_set:
                    match_count += 1
            if match_count > 0:
                scores[lang_code] = match_count

        if scores:
            best_lang = max(scores, key=scores.get)
            max_matches = scores[best_lang]
            confidence = min(0.95, round(0.50 + 0.15 * max_matches, 2))
            return {
                "language_code": best_lang,
                "language_name": SUPPORTED_LANGUAGES[best_lang]["name"],
                "confidence": confidence,
                "flag": SUPPORTED_LANGUAGES[best_lang]["flag"]
            }

        # Default fallback to English
        return {
            "language_code": "en",
            "language_name": "English",
            "confidence": 0.6,
            "flag": "🇬🇧"
        }

    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English for processing.

        Checks common phrase translations first, then tries LLM-based translation
        via Gemini, and falls back to the original text.

        Args:
            text: Text to be translated into English.
            source_lang: Language code of the source text.

        Returns:
            Translated English string (or original string on fallback).
        """
        if not text or not text.strip() or source_lang == "en":
            return text

        # Check reverse match in COMMON_TRANSLATIONS
        clean_input = text.strip().lower()
        for en_phrase, translations in COMMON_TRANSLATIONS.items():
            if source_lang in translations and translations[source_lang].strip().lower() == clean_input:
                return en_phrase

        # Try Gemini LLM translation
        if self.has_gemini and self._model:
            try:
                source_name = SUPPORTED_LANGUAGES.get(source_lang, {}).get("name", source_lang)
                prompt = (
                    f"You are an expert translator. Translate the following text from {source_name} "
                    f"to natural English. Output only the translated English text without explanation "
                    f"or quotation marks:\n\n{text}"
                )
                response = self._model.generate_content(prompt)
                if response and response.text:
                    translated = response.text.strip().strip("\"'")
                    if translated:
                        return translated
            except Exception as e:
                logger.warning("Gemini translation to English failed: %s", e)

        logger.info("Translation to English fallback: returning original text.")
        return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate English response to target language.

        Checks common phrase translations first, then tries LLM-based translation
        via Gemini, and falls back to the original English text.

        Args:
            text: English text to translate.
            target_lang: Language code of target language.

        Returns:
            Translated response string in target language (or original on fallback).
        """
        if not text or not text.strip() or target_lang == "en":
            return text

        # Check direct match in COMMON_TRANSLATIONS
        clean_input = text.strip()
        for en_phrase, translations in COMMON_TRANSLATIONS.items():
            if en_phrase.lower() == clean_input.lower() and target_lang in translations:
                return translations[target_lang]

        # Try Gemini LLM translation
        if self.has_gemini and self._model:
            try:
                target_name = SUPPORTED_LANGUAGES.get(target_lang, {}).get("name", target_lang)
                prompt = (
                    f"You are an expert translator. Translate the following text from English "
                    f"to natural {target_name}. Output only the translated text in {target_name} "
                    f"without explanation or quotation marks:\n\n{text}"
                )
                response = self._model.generate_content(prompt)
                if response and response.text:
                    translated = response.text.strip().strip("\"'")
                    if translated:
                        return translated
            except Exception as e:
                logger.warning("Gemini translation from English failed: %s", e)

        logger.info("Translation from English fallback: returning original text.")
        return text

    def process_multilingual_query(self, user_input: str) -> Dict[str, Any]:
        """Process a user query: detect language and translate if needed.

        Args:
            user_input: Raw query text from user.

        Returns:
            Dict containing original_text, detected_language, english_text,
            and needs_translation flag.
        """
        detection = self.detect_language(user_input)
        lang_code = detection["language_code"]
        needs_translation = (lang_code != "en")

        if needs_translation:
            english_text = self.translate_to_english(user_input, lang_code)
        else:
            english_text = user_input

        return {
            "original_text": user_input,
            "detected_language": detection,
            "english_text": english_text,
            "needs_translation": needs_translation
        }

    def format_multilingual_response(self, english_response: str, target_lang: str) -> Dict[str, Any]:
        """Format response for target language with cultural adaptations.

        Args:
            english_response: Response generated in English.
            target_lang: Destination language code.

        Returns:
            Dict containing response_text, language, flag, and greeting.
        """
        if target_lang != "en":
            response_text = self.translate_from_english(english_response, target_lang)
        else:
            response_text = english_response

        lang_info = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES["en"])
        greeting = lang_info.get("greeting", "Hello! How can I help you?")

        return {
            "response_text": response_text,
            "language": target_lang,
            "flag": lang_info.get("flag", "🇬🇧"),
            "greeting": greeting
        }

    def get_greeting(self, lang_code: Optional[str] = None) -> str:
        """Return appropriate greeting for language.

        Args:
            lang_code: Optional language code. If None, uses current_language.

        Returns:
            Greeting string in the specified language.
        """
        code = lang_code if lang_code else self.current_language
        return SUPPORTED_LANGUAGES.get(code, SUPPORTED_LANGUAGES["en"]).get("greeting", "Hello! How can I help you?")

    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Return list of supported language info dicts.

        Returns:
            List of dicts with code, name, flag, greeting, and direction.
        """
        return [{"code": code, **info} for code, info in SUPPORTED_LANGUAGES.items()]

    def set_language(self, lang_code: str) -> bool:
        """Validate and set current language preference.

        Args:
            lang_code: Language code to activate.

        Returns:
            True if language was supported and set, False otherwise.
        """
        if lang_code in SUPPORTED_LANGUAGES:
            self.current_language = lang_code
            logger.info("Current language set to: %s (%s)", lang_code, SUPPORTED_LANGUAGES[lang_code]["name"])
            return True
        logger.warning("Attempted to set unsupported language: %s", lang_code)
        return False

    def answer_multilingual_query(
        self,
        query: str,
        qa_chain: Any = None,
        force_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Answer a user query in their native language using the FAQ Knowledge Base.

        Architecture:
        1. Language Detection: Auto-detect language (or use force_language)
        2. Translation to English: Translate query to English for optimal FAISS semantic retrieval
        3. Knowledge Base Query: Retrieve verified answer from QA chain
        4. Localization: Translate the verified answer back into the user's native language
        5. Greeting & Cultural Politeness: Add language-appropriate greeting

        Args:
            query: User's raw question in any supported language.
            qa_chain: Optional RetrievalQA chain instance. If None or fails, falls back to direct LLM.
            force_language: Optional language code override (e.g., 'es', 'fr', 'de', 'hi', 'en').

        Returns:
            Dict containing:
                - original_query: str
                - target_language: str
                - language_name: str
                - flag: str
                - confidence: float
                - english_query: str
                - raw_english_answer: str
                - localized_answer: str
                - sources: List[str]
        """
        if not query or not query.strip():
            return {
                "original_query": "",
                "target_language": "en",
                "language_name": "English",
                "flag": "🇬🇧",
                "confidence": 1.0,
                "english_query": "",
                "raw_english_answer": "",
                "localized_answer": "Please enter a question.",
                "sources": []
            }

        # 1. Determine Language
        if force_language and force_language in SUPPORTED_LANGUAGES:
            target_lang = force_language
            detection = {
                "language_code": target_lang,
                "language_name": SUPPORTED_LANGUAGES[target_lang]["name"],
                "confidence": 1.0,
                "flag": SUPPORTED_LANGUAGES[target_lang]["flag"]
            }
        else:
            detection = self.detect_language(query)
            target_lang = detection["language_code"]

        # 2. Normalize to English for vector search
        if target_lang != "en":
            english_query = self.translate_to_english(query, target_lang)
        else:
            english_query = query

        # 3. Query Knowledge Base
        raw_english_answer = ""
        sources = []
        if qa_chain:
            try:
                qa_res = qa_chain({"query": english_query})
                raw_english_answer = qa_res.get("result", "").strip()
                docs = qa_res.get("source_documents", [])
                for doc in docs:
                    content = getattr(doc, "page_content", str(doc))
                    if content:
                        sources.append(content)
            except Exception as e:
                logger.warning("QA chain execution failed in multilingual query: %s", e)

        # 4. Fallback to direct Gemini LLM if QA chain returned empty or unknown
        if not raw_english_answer or "I don't know" in raw_english_answer or "don't have" in raw_english_answer.lower():
            if self.has_gemini and self._model:
                try:
                    lang_name = SUPPORTED_LANGUAGES.get(target_lang, {}).get("name", "English")
                    prompt = (
                        f"You are a friendly customer service assistant for Nullclass (an online tech education "
                        f"platform providing courses in Python, Java, Data Science, Web Development, real-world project "
                        f"internships, certificates, and flexible EMI payment options).\n"
                        f"Answer the customer's question directly and concisely in {lang_name}:\n\n"
                        f"Customer Question: {query}"
                    )
                    resp = self._model.generate_content(prompt)
                    if resp and resp.text:
                        localized = resp.text.strip()
                        return {
                            "original_query": query,
                            "target_language": target_lang,
                            "language_name": detection.get("language_name", "English"),
                            "flag": detection.get("flag", "🌐"),
                            "confidence": detection.get("confidence", 0.95),
                            "english_query": english_query,
                            "raw_english_answer": raw_english_answer or localized,
                            "localized_answer": localized,
                            "sources": sources
                        }
                except Exception as ex:
                    logger.warning("Direct LLM multilingual answering failed: %s", ex)

        # 5. Localize English answer back to user's native language
        if target_lang != "en" and raw_english_answer:
            localized_answer = self.translate_from_english(raw_english_answer, target_lang)
        else:
            localized_answer = raw_english_answer or "I apologize, but I could not find information about that. Please contact support."

        return {
            "original_query": query,
            "target_language": target_lang,
            "language_name": detection.get("language_name", "English"),
            "flag": detection.get("flag", "🌐"),
            "confidence": detection.get("confidence", 0.95),
            "english_query": english_query,
            "raw_english_answer": raw_english_answer,
            "localized_answer": localized_answer,
            "sources": sources
        }

    def get_status(self) -> Dict[str, Any]:
        """Return status with available features and active configuration.

        Returns:
            Dict containing current_language, feature flags, and supported languages list.
        """
        return {
            "current_language": self.current_language,
            "current_language_name": SUPPORTED_LANGUAGES.get(self.current_language, {}).get("name", "English"),
            "has_langdetect": self.has_langdetect,
            "has_gemini": self.has_gemini,
            "supported_languages_count": len(SUPPORTED_LANGUAGES),
            "supported_languages": list(SUPPORTED_LANGUAGES.keys())
        }

