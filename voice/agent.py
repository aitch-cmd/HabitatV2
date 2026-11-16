import re
import logging
from typing import Dict, Any, List
from tools.search.server import ProductionPropertyPipeline, _serialize_document

logger = logging.getLogger("voice_agent")


class VoiceRentalAgent:
    """
    Conversational Voice Rental Advisor for HabitatOS.
    Executes the 4-stage production retrieval pipeline and shapes
    concise, voice-optimized verbal responses.
    """

    def __init__(self):
        logger.info("Initializing VoiceRentalAgent...")
        self.pipeline = ProductionPropertyPipeline()
        self.history: List[Dict[str, str]] = []

    def reset_conversation(self):
        self.history.clear()

    async def process_utterance(self, user_text: str) -> Dict[str, Any]:
        """
        Process user speech transcription:
        1. Classify intent (greeting vs rental search vs alert/status).
        2. Run 4-stage retrieval + valuation pipeline if search.
        3. Compose conversational spoken response and visual card payloads.
        """
        user_clean = user_text.strip()
        self.history.append({"role": "user", "content": user_clean})
        lower = user_clean.lower()

        # Handle simple greetings
        if any(g in lower for g in ["hello", "hi there", "hey", "good morning", "good evening"]) and len(lower.split()) <= 3:
            speech = "Hello! I am HabitatOS, your voice rental advisor. Tell me what kind of home, budget, and neighborhood you are looking for."
            display = speech
            self.history.append({"role": "assistant", "content": speech})
            return {"speech_text": speech, "display_text": display, "listings": []}

        # Check for listing search intent
        try:
            results = self.pipeline.search(user_clean, top_k=3)
            serialized = [_serialize_document(r) for r in results]
        except Exception as e:
            logger.error(f"Error executing pipeline search: {e}")
            speech = "I had trouble checking the database. Could you please specify your preferred area and budget again?"
            return {"speech_text": speech, "display_text": speech, "listings": []}

        if not serialized:
            speech = f"I could not find any available listings matching that criteria. Try expanding your budget buffer or searching in another locality."
            display = speech
            self.history.append({"role": "assistant", "content": speech})
            return {"speech_text": speech, "display_text": display, "listings": []}

        # Compose spoken answer from top deal
        top = serialized[0]
        top_title = top.get("title") or f"{top.get('bedrooms', 2)} BHK"
        top_loc = top.get("locality") or top.get("city") or "Delhi"
        top_price = top.get("price", 0)
        top_bhk = top.get("bedrooms", 1)

        val = top.get("rent_valuation") or {}
        verdict = val.get("valuation_verdict", "Fair Market Price")
        p50 = val.get("fair_market_rent_p50", top_price)
        savings = val.get("savings_or_premium", int(p50 - top_price))

        val_voice = ""
        if savings > 1000:
            val_voice = f" It is rated {verdict}, priced ₹{abs(int(savings)):,} below the locality fair median."
        elif savings < -1000:
            val_voice = f" It is priced ₹{abs(int(savings)):,} above average market rent for this area."
        else:
            val_voice = " It is priced right at fair market rent."

        speech = (
            f"I found {len(serialized)} matching properties. "
            f"Top match is a {top_bhk} BHK in {top_loc} for ₹{top_price:,} per month.{val_voice} "
            f"Would you like more details on this home?"
        )

        display = f"### Found {len(serialized)} Properties\n\n"
        for i, item in enumerate(serialized, 1):
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
            "listings": serialized,
        }
