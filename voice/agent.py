import re
import logging
from typing import Dict, Any, List, Optional
from tools.search.server import ProductionPropertyPipeline, _serialize_document

logger = logging.getLogger("voice_agent")


def _format_spoken_price(amount: float | int) -> str:
    """Format price in natural conversational Indian English for TTS."""
    amt = int(round(amount))
    if amt >= 100000:
        lakhs = amt / 100000
        if lakhs.is_integer():
            return f"{int(lakhs)} lakh rupees"
        return f"{lakhs:.1f} lakh rupees"
    return f"{amt:,} rupees"


def _format_rounded_delta(amount: float | int) -> str:
    """Round large price deltas for smooth, natural speech."""
    amt = abs(int(amount))
    if amt >= 10000:
        rounded = round(amt, -3)
        return f"about {rounded:,} rupees"
    if amt >= 1000:
        rounded = round(amt, -2)
        return f"about {rounded:,} rupees"
    return f"{amt:,} rupees"


class VoiceRentalAgent:
    """
    Conversational Voice Rental Advisor for HabitatOS.
    Executes the 4-stage production retrieval pipeline and shapes
    concise, voice-optimized verbal responses with multi-turn dialogue context.
    """

    SUPPORTED_CITIES = {"delhi", "mumbai", "bangalore", "kolkata", "hyderabad", "chennai"}

    def __init__(self):
        logger.info("Initializing VoiceRentalAgent...")
        self.pipeline = ProductionPropertyPipeline()
        self.history: List[Dict[str, str]] = []
        self.last_listings: List[Dict[str, Any]] = []
        self.current_index: int = 0

    def reset_conversation(self):
        self.history.clear()
        self.last_listings.clear()
        self.current_index = 0

    def _is_affirmative_or_detail_request(self, lower: str) -> bool:
        """Detect if the user is asking for more details about the current listing."""
        detail_cues = [
            "yes", "yeah", "yep", "sure", "tell me more", "more details",
            "what about amenities", "amenities", "details", "tell me about it",
            "sounds good", "elaborate", "interested", "how many bathrooms",
            "furnishing", "parking"
        ]
        return any(cue in lower for cue in detail_cues)

    def _is_next_listing_request(self, lower: str) -> bool:
        """Detect if the user wants to see the next/alternative listing."""
        next_cues = [
            "next", "another", "other", "cheaper", "second", "third",
            "show me another", "what else", "anything else", "next one"
        ]
        return any(cue in lower for cue in next_cues)

    def _handle_detail_query(self, lower: str) -> Optional[Dict[str, Any]]:
        """Answer follow-up detail questions using cached search results."""
        if not self.last_listings:
            return None

        top = self.last_listings[self.current_index]
        title = top.get("title", "Property")
        bhk = top.get("bedrooms", 2)
        bathrooms = top.get("bathrooms", 1)
        furnishing = top.get("furnishing", "semi-furnished")
        price = top.get("price", 0)
        locality = top.get("locality") or top.get("city") or "the area"
        spoken_price = _format_spoken_price(price)
        sqft = top.get("area_sqft", "")

        val = top.get("rent_valuation") or {}
        drivers = val.get("explanation_drivers", [])
        driver_sentence = ""
        if drivers:
            raw_driver = drivers[0]
            factor_name = raw_driver.split("(")[0].split(":")[0].strip()
            if factor_name and "model unavailable" not in factor_name.lower():
                driver_sentence = f" A key price factor is the {factor_name.lower()}."

        bath_str = "1 bathroom" if bathrooms == 1 else f"{bathrooms} bathrooms"

        speech = (
            f"This {bhk} BHK in {locality} features {bath_str} and is {furnishing}."
            f"{f' It spans {sqft} square feet.' if sqft else ''}"
            f" The rent is {spoken_price} per month.{driver_sentence}"
            f" Would you like to check the next property or search another area?"
        )
        display = (
            f"### Details: {title}\n"
            f"- **Rent**: ₹{price:,}/month\n"
            f"- **Bedrooms**: {bhk} BHK | **Bathrooms**: {bathrooms}\n"
            f"- **Furnishing**: {furnishing.title()}\n"
            f"- **Area**: {sqft} sq.ft\n"
        )
        self.history.append({"role": "assistant", "content": speech})
        return {
            "speech_text": speech,
            "display_text": display,
            "listings": [top],
        }

    def _handle_next_listing(self) -> Optional[Dict[str, Any]]:
        """Advance to the next candidate property from the previous search."""
        if not self.last_listings or self.current_index + 1 >= len(self.last_listings):
            return None

        self.current_index += 1
        top = self.last_listings[self.current_index]
        return self._compose_search_response([top], prefix="The next best match is ")

    def _compose_search_response(self, listings: List[Dict[str, Any]], prefix: str = "") -> Dict[str, Any]:
        """Compose voice-optimized response and markdown display cards."""
        top = listings[0]
        top_loc = top.get("locality") or top.get("city") or "Delhi"
        top_price = top.get("price", 0)
        top_bhk = top.get("bedrooms", 1)

        val = top.get("rent_valuation") or {}
        verdict = val.get("valuation_verdict", "Fair Market Price")
        p50 = val.get("fair_market_rent_p50", top_price)
        savings = val.get("savings_or_premium", int(p50 - top_price))

        val_voice = ""
        if savings > 1000:
            val_voice = f" It is rated {verdict}, priced {_format_rounded_delta(savings)} below the locality fair median."
        elif savings < -1000:
            val_voice = f" It is priced {_format_rounded_delta(savings)} above average market rent for this area."
        else:
            val_voice = " It is priced right at fair market rent."

        count_text = (
            f"I found 1 matching property."
            if len(listings) == 1
            else f"I found {len(listings)} matching properties."
        )

        spoken_price = _format_spoken_price(top_price)
        intro = prefix if prefix else f"{count_text} Top match is "

        speech = (
            f"{intro}a {top_bhk} BHK in {top_loc} for {spoken_price} per month.{val_voice} "
            f"Would you like more details on this home?"
        )

        display = f"### Found {len(listings)} Properties\n\n"
        for i, item in enumerate(listings, 1):
            item_val = item.get("rent_valuation") or {}
            item_verdict = item_val.get("valuation_verdict", "Fair Market")
            item_p50 = item_val.get("fair_market_rent_p50", item.get("price"))
            drivers = item_val.get("explanation_drivers", [])
            driver_str = f" ({drivers[0]})" if drivers else ""
            display += (
                f"**{i}. {item.get('title')}**\n"
                f"- **Rent**: ₹{item.get('price'):,}/mo | **Area**: {item.get('locality', '')}, {item.get('city', '')}\n"
                f"- **Valuation**: {item_verdict} (Fair P50: ₹{item_p50:,}){driver_str}\n\n"
            )

        self.history.append({"role": "assistant", "content": speech})
        return {
            "speech_text": speech,
            "display_text": display,
            "listings": listings,
        }

    async def process_utterance(self, user_text: str) -> Dict[str, Any]:
        """
        Process user speech transcription:
        1. Handle empty / noise inputs.
        2. Classify intent (greeting vs follow-up vs new search).
        3. Maintain conversational context for multi-turn dialogues.
        4. Run 4-stage retrieval + valuation pipeline for property search.
        5. Compose voice-optimized responses with natural spoken phrasing.
        """
        user_clean = user_text.strip()
        self.history.append({"role": "user", "content": user_clean})
        lower = user_clean.lower()

        # 1. Guard against empty / noise inputs
        if len(user_clean) < 2 or not any(c.isalnum() for c in user_clean):
            speech = "I didn't quite catch that. Could you tell me what city, neighborhood, or budget you are looking for?"
            self.history.append({"role": "assistant", "content": speech})
            return {"speech_text": speech, "display_text": speech, "listings": []}

        # 2. Handle pure greetings (ensure search criteria like 'bhk', 'rent', 'flat', numbers aren't swallowed)
        has_search_clue = any(cue in lower for cue in ["bhk", "bedroom", "flat", "rent", "k", "lakh", "under", "budget"]) or any(char.isdigit() for char in lower)
        greeting_words = ["hello", "hi there", "hey", "good morning", "good evening", "hi"]
        is_greeting = any(g in lower for g in greeting_words) and len(lower.split()) <= 3

        if is_greeting and not has_search_clue:
            speech = "Hello! I am HabitatOS, your voice rental advisor. Tell me what kind of home, budget, and neighborhood you are looking for."
            self.history.append({"role": "assistant", "content": speech})
            return {"speech_text": speech, "display_text": speech, "listings": []}

        # 3. Check for multi-turn dialogue follow-ups (if user previously had search results)
        if self.last_listings:
            if self._is_affirmative_or_detail_request(lower):
                detail_resp = self._handle_detail_query(lower)
                if detail_resp:
                    return detail_resp
            elif self._is_next_listing_request(lower):
                next_resp = self._handle_next_listing()
                if next_resp:
                    return next_resp

        # 4. Check for unsupported city inquiries (proactive guidance)
        detected_noida_or_gurgaon = any(c in lower for c in ["noida", "gurgaon", "gurugram"])
        if detected_noida_or_gurgaon and not any(loc in lower for loc in ["sector 62", "sector 137", "cyber city", "golf course"]):
            speech = (
                "Our current real-time verified coverage includes Delhi, Mumbai, Bangalore, Hyderabad, Chennai, and Kolkata. "
                "Would you like to explore listings in nearby South Delhi or another covered city?"
            )
            self.history.append({"role": "assistant", "content": speech})
            return {"speech_text": speech, "display_text": speech, "listings": []}

        # 5. Execute 4-Stage Production Retrieval Pipeline
        try:
            results = self.pipeline.search(user_clean, top_k=3)
            serialized = [_serialize_document(r) for r in results]
        except Exception as e:
            logger.error(f"Error executing pipeline search: {e}")
            speech = "I had trouble checking the database. Could you please specify your preferred area and budget again?"
            return {"speech_text": speech, "display_text": speech, "listings": []}

        if not serialized:
            speech = "I could not find any available listings matching that criteria. Try expanding your budget buffer or searching in another locality."
            self.history.append({"role": "assistant", "content": speech})
            return {"speech_text": speech, "display_text": speech, "listings": []}

        # Cache active listings and reset focal index for subsequent follow-ups
        self.last_listings = serialized
        self.current_index = 0

        # 6. Compose conversational spoken answer from top match
        return self._compose_search_response(serialized)
