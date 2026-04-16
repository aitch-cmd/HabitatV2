from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core.llm import llm


class PropertySearchCriteria(BaseModel):
    """Schema for structured property search extraction."""
    location: Optional[str] = Field(
        default=None,
        description="The city, area, or neighborhood mentioned (e.g., 'Bangalore', 'Koramangala')"
    )
    price: Optional[int] = Field(
        default=None,
        description="Budget or max price as a number (e.g., 20000). Extract only the numeric value."
    )
    rag_content: Optional[str] = Field(
        default=None,
        description=(
            "Additional search criteria: number of bedrooms (e.g., '2BHK', '3BHK'), "
            "furnishing (furnished/semi-furnished/unfurnished), property type (flat/apartment/villa), "
            "amenities (parking, gym, pool), and any other requirements"
        )
    )


class UserMessageParser:
    """
    Parses user search queries into structured fields for property retrieval.

    Example:
        Input: "Show me 2BHK flats in Bangalore under 20k"
        Output: {
            "location": "Bangalore",
            "price": 20000,
            "rag_content": "2BHK flats"
        }
    """

    def __init__(self):
        self.output_parser = JsonOutputParser(pydantic_object=PropertySearchCriteria)

        self.prompt = ChatPromptTemplate.from_template("""You are an expert at extracting property search criteria from natural language.

Extract the following from the user's message:
1. location: City, area, or neighborhood (if mentioned)
2. price: Maximum budget as a number only (convert "20k" to 20000, "1.5L" to 150000, "1.5 lakh" to 150000)
3. rag_content: All other requirements (bedrooms, furnishing, amenities, preferences)

Important:
- If a field is not mentioned, set it to null
- For price, extract only the numeric value (no currency symbols or text)
- Be thorough in capturing all requirements in rag_content

User message: {user_message}
{format_instructions}""")

        self.chain = self.prompt | llm | self.output_parser

    def extract(self, user_message: str) -> dict:
        """Extract structured property search criteria from user message."""
        try:
            result = self.chain.invoke({
                "user_message": user_message,
                "format_instructions": self.output_parser.get_format_instructions()
            })
            return result
        except Exception as e:
            print(f"Error parsing message: {e}")
            return {
                "location": None,
                "price": None,
                "rag_content": user_message  # Fallback: use entire message
            }


if __name__ == "__main__":
    extractor = UserMessageParser()

    test_messages = [
        "Show me 2BHK flats in Bangalore under 20k",
        "3BHK furnished apartment in Koramangala with parking, budget 50000",
        "Find me a villa in Whitefield",
        "Studio apartment near MG Road under 15k",
        "I need a 1BHK in HSR Layout for 18000"
    ]

    print("Testing UserMessageParser:\n")
    for msg in test_messages:
        print(f"Input: {msg}")
        result = extractor.extract(msg)
        print(f"Output: {result}\n")