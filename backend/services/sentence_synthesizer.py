"""
KINEX ISL Sentence Synthesizer Service
Transforms sequences of discrete Indian Sign Language (ISL) gesture glosses/actions
into fluent, natural, grammatically correct English sentences.

Supports:
- Pre-defined idiomatic ISL phrase maps (Subject-Object-Verb to SVO translation)
- Contextual grammar assembly for multi-sign action sequences
- Single-sign natural expansion
- Dynamic rule-based fallback for custom signs (e.g., added via Camera Trainer)
- Sentence confidence aggregation & grammatical stability scoring
"""

import time
from typing import Any, Dict, List, Optional, Tuple


class ISLSentenceSynthesizer:
    """
    Translates sequences of discrete sign glosses into complete, natural sentences.
    Designed specifically for Indian Sign Language (ISL) continuous syntax.
    """

    # Comprehensive ISL Gloss -> English Sentence Grammatical Dictionary
    IDIOMATIC_SENTENCES: Dict[Tuple[str, ...], str] = {
        # Single Sign Expansions
        ("THANK_YOU",): "Thank you very much.",
        ("INDIAN",): "I am proud to be Indian.",
        ("NAME",): "My name is being signed.",
        ("I",): "I am present and communicating.",
        ("HEARING",): "I am hard of hearing / deaf.",
        ("MORNING",): "Good morning to you!",
        ("BYE",): "Goodbye, have a great day!",
        ("NAMASTE",): "Namaste, warm greetings to you.",
        ("HELP",): "I need assistance, please help me.",
        ("WATER",): "Please give me some drinking water.",
        ("FOOD",): "I would like to have food.",
        ("YES",): "Yes, I agree.",
        ("NO",): "No, I disagree.",
        ("PLEASE",): "Please do this for me.",
        ("SORRY",): "I am sincerely sorry.",

        # 2-Sign Sequences (ISL SOV & Topic-Comment constructions)
        ("I", "INDIAN"): "I am an Indian citizen.",
        ("INDIAN", "I"): "I am proud of my Indian heritage.",
        ("I", "NAME"): "My name is KINEX User.",
        ("NAME", "I"): "Please call me by my name.",
        ("I", "HEARING"): "I am deaf or hard of hearing.",
        ("HEARING", "I"): "As a deaf individual, I communicate using ISL.",
        ("I", "THANK_YOU"): "I want to express my sincere thanks to you.",
        ("THANK_YOU", "I"): "Thank you from the bottom of my heart.",
        ("I", "BYE"): "I will take your leave now, goodbye!",
        ("BYE", "I"): "Goodbye, I am departing now.",
        ("I", "MORNING"): "Good morning! I am ready to begin.",
        ("MORNING", "I"): "Good morning from my side.",

        ("MORNING", "THANK_YOU"): "Good morning! Thank you very much for your time.",
        ("THANK_YOU", "MORNING"): "Thank you, and have a wonderful morning!",
        ("THANK_YOU", "BYE"): "Thank you so much, goodbye and take care!",
        ("BYE", "THANK_YOU"): "Goodbye! Thank you once again.",
        ("MORNING", "BYE"): "Good morning and goodbye, have a pleasant day!",
        ("MORNING", "INDIAN"): "Good morning to all fellow Indians!",
        ("INDIAN", "MORNING"): "A very good morning from India!",
        ("INDIAN", "THANK_YOU"): "Thank you on behalf of the Indian Sign Language community.",
        ("THANK_YOU", "INDIAN"): "Thank you, proud to be Indian!",
        ("NAME", "INDIAN"): "My identity and roots are in India.",
        ("INDIAN", "NAME"): "My Indian name is registered.",
        ("HEARING", "THANK_YOU"): "Thank you for understanding my hearing condition.",
        ("THANK_YOU", "HEARING"): "Thank you for supporting accessibility for the deaf.",
        ("NAME", "HEARING"): "My name belongs to the deaf community.",
        ("HEARING", "NAME"): "I communicate my name using deaf sign language.",
        ("NAME", "THANK_YOU"): "Thank you for asking for my name.",
        ("NAME", "BYE"): "Goodbye, remember my name!",

        # 3-Sign Sequences
        ("MORNING", "I", "INDIAN"): "Good morning! I am a proud Indian.",
        ("I", "INDIAN", "THANK_YOU"): "I am an Indian, and I thank you with gratitude.",
        ("I", "HEARING", "THANK_YOU"): "I am hard of hearing; thank you for communicating with me.",
        ("MORNING", "THANK_YOU", "BYE"): "Good morning! Thank you very much, and goodbye!",
        ("I", "NAME", "INDIAN"): "My name is registered and I belong to India.",
        ("I", "THANK_YOU", "BYE"): "I thank you warmly, goodbye and take care!",
        ("MORNING", "I", "THANK_YOU"): "Good morning! I sincerely thank you today.",
        ("I", "HEARING", "INDIAN"): "I am a deaf Indian communicating through sign language.",
        ("INDIAN", "HEARING", "THANK_YOU"): "Thank you for empowering Indian deaf communication.",
        ("NAME", "THANK_YOU", "BYE"): "Thank you for taking my name, goodbye!",
    }

    # Semantic Categorization of Vocabulary for Rule-Based Fallback
    SUBJECTS = {"I", "ME", "YOU", "WE", "THEY", "HE", "SHE"}
    GREETINGS = {"MORNING", "HELLO", "NAMASTE", "HI", "WELCOME"}
    FAREWELLS = {"BYE", "GOODBYE", "LATER", "FAREWELL"}
    POLITENESS = {"THANK_YOU", "THANKS", "PLEASE", "SORRY"}
    IDENTITY = {"INDIAN", "NAME", "HEARING", "STUDENT", "TEACHER", "FRIEND"}
    REQUESTS = {"HELP", "WATER", "FOOD", "MEDICINE", "DOCTOR", "RESTROOM"}

    def __init__(self):
        pass

    def synthesize(
        self,
        tokens: List[Any],
        mode: str = "vision",
    ) -> Dict[str, Any]:
        """
        Synthesizes a list of sign gloss tokens into a complete English sentence.

        Args:
            tokens: List of strings or dictionaries containing {'gesture': str, 'confidence': float}
            mode: Operating mode ('vision', 'glove', 'fusion')

        Returns:
            Dictionary containing:
                - sentence: Formatted English sentence
                - tokens: List of cleaned uppercase gloss names
                - confidence: Aggregated confidence score
                - grammar_rule: Applied grammatical rule/template identifier
                - timestamp_ms: Current timestamp
        """
        if not tokens:
            return {
                "sentence": "Waiting for sign actions...",
                "tokens": [],
                "confidence": 0.0,
                "grammar_rule": "EMPTY_BUFFER",
                "timestamp_ms": time.time() * 1000.0,
            }

        cleaned_tokens: List[str] = []
        confidences: List[float] = []

        for item in tokens:
            if isinstance(item, dict):
                g = str(item.get("gesture", "")).upper().strip()
                c = float(item.get("confidence", 0.90))
            else:
                g = str(item).upper().strip()
                c = 0.92

            # Filter out non-gestures
            if g and g not in ["UNKNOWN", "NONE", "IDLE"]:
                cleaned_tokens.append(g)
                confidences.append(c)

        if not cleaned_tokens:
            return {
                "sentence": "No confirmed sign actions detected.",
                "tokens": [],
                "confidence": 0.0,
                "grammar_rule": "NO_VALID_TOKENS",
                "timestamp_ms": time.time() * 1000.0,
            }

        # Calculate average confidence with slight bonus for multi-token agreement
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.85
        if len(cleaned_tokens) > 1:
            avg_conf = min(0.99, avg_conf + 0.02)

        # 1. Check exact dictionary match
        token_tuple = tuple(cleaned_tokens)
        if token_tuple in self.IDIOMATIC_SENTENCES:
            return {
                "sentence": self.IDIOMATIC_SENTENCES[token_tuple],
                "tokens": cleaned_tokens,
                "confidence": round(avg_conf, 3),
                "grammar_rule": f"EXACT_IDIOMATIC_L{len(cleaned_tokens)}",
                "timestamp_ms": time.time() * 1000.0,
            }

        # 2. De-duplicate consecutive identical gestures (e.g., ['I', 'I', 'INDIAN'] -> ['I', 'INDIAN'])
        deduped: List[str] = []
        for g in cleaned_tokens:
            if not deduped or deduped[-1] != g:
                deduped.append(g)

        deduped_tuple = tuple(deduped)
        if deduped_tuple in self.IDIOMATIC_SENTENCES:
            return {
                "sentence": self.IDIOMATIC_SENTENCES[deduped_tuple],
                "tokens": cleaned_tokens,
                "confidence": round(avg_conf, 3),
                "grammar_rule": f"DEDUPED_IDIOMATIC_L{len(deduped)}",
                "timestamp_ms": time.time() * 1000.0,
            }

        # 3. Rule-based Grammatical Synthesizer (for custom signs & extended phrases)
        sentence, rule = self._assemble_rule_based_sentence(deduped)

        return {
            "sentence": sentence,
            "tokens": cleaned_tokens,
            "confidence": round(avg_conf, 3),
            "grammar_rule": rule,
            "timestamp_ms": time.time() * 1000.0,
        }

    def _assemble_rule_based_sentence(self, tokens: List[str]) -> Tuple[str, str]:
        """Synthesizes English sentence using ISL syntactic decomposition rules."""
        token_set = set(tokens)

        # Case A: Greeting + Farewell
        if token_set & self.GREETINGS and token_set & self.FAREWELLS:
            return "Greetings to you, and goodbye!", "RULE_GREET_FAREWELL"

        # Case B: Greeting + Subject / Identity
        if token_set & self.GREETINGS:
            non_greetings = [t for t in tokens if t not in self.GREETINGS]
            if not non_greetings:
                return "Good morning! Wishing you a productive day.", "RULE_GREETING_STANDALONE"
            phrase = " ".join(t.capitalize() for t in non_greetings)
            return f"Good morning! Communicating {phrase}.", "RULE_GREETING_EXTENDED"

        # Case C: Requests / Emergency needs (HELP, WATER, FOOD)
        if token_set & self.REQUESTS:
            reqs = [t.lower() for t in tokens if t in self.REQUESTS]
            if "help" in reqs:
                return "Please provide urgent assistance.", "RULE_REQUEST_HELP"
            if "water" in reqs:
                return "Could you please give me water to drink?", "RULE_REQUEST_WATER"
            return f"I am requesting assistance regarding: {', '.join(reqs)}.", "RULE_REQUEST_GENERIC"

        # Case D: Subject + Politeness / Gratitude
        if token_set & self.SUBJECTS and token_set & self.POLITENESS:
            return "I sincerely thank you and express my appreciation.", "RULE_SUBJECT_POLITENESS"

        # Case E: Generic multi-token phrase smoothing
        parts: List[str] = []
        for t in tokens:
            formatted = t.replace("_", " ").title()
            parts.append(formatted)

        natural_chain = " ".join(parts)
        return f"{natural_chain} — expressed in Indian Sign Language.", "RULE_NATURAL_SYNTHESIS"


# Global singleton instance
synthesizer = ISLSentenceSynthesizer()
