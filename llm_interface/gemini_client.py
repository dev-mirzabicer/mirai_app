# mirai_app/llm_interface/gemini_client.py

import logging
from typing import Optional, List, Tuple, Any

from google import genai
from google.genai import types as genai_types
from google.auth.exceptions import DefaultCredentialsError
from google.api_core import exceptions as google_api_exceptions

from mirai_app import config

# from mirai_app.llm_interface.function_declarations import TOOL_DECLARATIONS
from mirai_app.llm_interface.function_declarations import ALL_TOOLS

logger = logging.getLogger(__name__)

tool_cfg = genai_types.ToolConfig(
    function_calling_config=genai_types.FunctionCallingConfig(mode="ANY")
)


class GeminiClient:
    """
    Manages communication with the Google Gemini API using the google.genai SDK.
    Handles sending prompts and receiving responses, including text and function calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.GEMINI_MODEL_NAME
        self.client: Optional[genai.Client] = None
        self._initialize_client()

    def _initialize_client(self):
        if not self.api_key or "YOUR_GEMINI_API_KEY" in self.api_key:
            logger.error(
                "Gemini API key not configured. GeminiClient will not function."
            )
            return
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(
                f"GeminiClient initialized successfully for model: {self.model_name}"
            )
        except ValueError as ve:
            logger.error(
                f"ValueError during GeminiClient initialization (invalid API key?): {ve}"
            )
            self.client = None
        except Exception as e:
            logger.exception(
                f"Unexpected error during GeminiClient initialization: {e}"
            )
            self.client = None

    def is_ready(self) -> bool:
        return self.client is not None

    def _get_default_generation_config_dict(
        self, system_instruction_text: Optional[str] = None
    ) -> dict:
        """
        Returns the default generation configuration as a dictionary.
        If system_instruction_text is provided, it's included correctly as a list of Parts.
        """
        config_dict = {
            "temperature": config.GEMINI_TEMPERATURE,
            "top_p": config.GEMINI_TOP_P,
            "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
            "candidate_count": 1,
            "safety_settings": self._get_default_safety_settings(),
            "tools": ALL_TOOLS,
            "tool_config": tool_cfg,
        }
        if system_instruction_text:
            config_dict["system_instruction"] = [
                genai_types.Part.from_text(text=system_instruction_text)
            ]
        return config_dict

    def _get_default_safety_settings(self) -> List[genai_types.SafetySetting]:
        return [
            genai_types.SafetySetting(
                category=cat,
                threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
            )
            for cat in [
                genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        ]

    async def generate_response(
        self,
        prompt_history_contents: List[genai_types.Content],
        system_instruction_text: Optional[str] = None,
        generation_config_override_dict: Optional[dict] = None,
        safety_settings_override: Optional[List[genai_types.SafetySetting]] = None,
    ) -> Tuple[Optional[str], Optional[List[genai_types.FunctionCall]], Optional[str]]:
        if not self.is_ready():
            error_msg = "GeminiClient is not initialized. Cannot generate response."
            logger.error(error_msg)
            return None, None, error_msg

        # Prepare generation_config
        # Start with defaults, including system_instruction if provided
        current_gen_config_dict = self._get_default_generation_config_dict(
            system_instruction_text
        )

        # Apply overrides
        if generation_config_override_dict:
            # Special handling for system_instruction if it's in overrides
            if "system_instruction" in generation_config_override_dict:
                override_si = generation_config_override_dict.pop("system_instruction")
                if isinstance(override_si, str):
                    current_gen_config_dict["system_instruction"] = [
                        genai_types.Part.from_text(text=override_si)
                    ]
                elif isinstance(
                    override_si, list
                ):  # Assume it's already list of Parts or compatible
                    current_gen_config_dict["system_instruction"] = override_si
            current_gen_config_dict.update(generation_config_override_dict)

        final_config = (
            genai_types.GenerateContentConfig(**current_gen_config_dict)
            if current_gen_config_dict
            else None
        )

        safety_settings_to_use = (
            safety_settings_override or self._get_default_safety_settings()
        )

        logger.debug(
            f"Sending request to Gemini model '{self.model_name}'. History length: {len(prompt_history_contents)}. "
            f"System instruction provided: {bool(system_instruction_text or (generation_config_override_dict and 'system_instruction' in generation_config_override_dict))}"
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=f"models/{self.model_name}",
                contents=prompt_history_contents,
                config=final_config,
                # tool_config=tool_cfg,
                # tools=TOOL_DECLARATIONS,
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = (
                    f"Response blocked. Reason: {response.prompt_feedback.block_reason.name}. "
                    f"Message: {getattr(response.prompt_feedback, 'block_reason_message', 'N/A')}"
                )
                logger.warning(block_reason_msg)
                for rating in response.prompt_feedback.safety_ratings:
                    logger.warning(
                        f"Safety Rating: {rating.category.name} - {rating.probability.name}"
                    )
                return None, None, block_reason_msg

            if not response.candidates:
                no_candidate_msg = "No candidates returned from Gemini model."
                logger.warning(no_candidate_msg)
                return None, None, no_candidate_msg

            candidate = response.candidates[0]
            text_response: Optional[str] = None

            function_calls = response.function_calls or []

            if candidate.content and candidate.content.parts:
                text_response = "".join(
                    p.text for p in candidate.content.parts if getattr(p, "text", None)
                )

            if text_response:
                logger.info(
                    f"Gemini text response received: '{text_response[:100]}...'"
                )
            if function_calls:
                logger.info(
                    f"Gemini function call(s) requested: {[fc.name for fc in function_calls]}"
                )

            return text_response, function_calls, None

        except google_api_exceptions.GoogleAPIError as e:
            logger.error(f"Google API Error during Gemini call: {e}")
            return None, None, f"Google API Error: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected error during Gemini API call: {e}")
            return None, None, f"API call failed: {str(e)}"


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def test_gemini_client():
        if "YOUR_GEMINI_API_KEY" in config.GEMINI_API_KEY or not config.GEMINI_API_KEY:
            print("Skipping GeminiClient test: GEMINI_API_KEY not configured.")
            return

        gemini_interaction_client = GeminiClient()
        if not gemini_interaction_client.is_ready():
            print("GeminiClient failed to initialize. Aborting test.")
            return

        print("\n--- Testing Basic Text Generation ---")
        history1 = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text="Hello, MIRAI! Who are you?")],
            )
        ]
        system_instruction1 = "You are MIRAI, a helpful AI assistant."
        text_res, fc_res, err_res = await gemini_interaction_client.generate_response(
            history1, system_instruction1
        )

        print(f"Text Response: {text_res}")
        print(f"Function Calls: {fc_res}")
        print(f"Error: {err_res}")
        assert err_res is None, f"Error during basic text generation: {err_res}"
        assert text_res is not None, "No text response for basic generation."

        print("\n--- Testing Function Calling ---")
        history2 = [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_text(
                        text="Please create a task to buy milk tomorrow."
                    )
                ],
            )
        ]
        system_instruction2 = "You are MIRAI. Your goal is to help Mirza manage his life. Use tools when appropriate. If a user asks to create a task, use the create_task function."
        text_res2, fc_res2, err_res2 = (
            await gemini_interaction_client.generate_response(
                history2, system_instruction2
            )
        )

        print(f"Text Response: {text_res2}")
        print(f"Function Calls: {fc_res2}")
        print(f"Error: {err_res2}")
        assert err_res2 is None, f"Error during function call test: {err_res2}"

        if fc_res2:
            print(f"LLM wants to call: {[(fc.name, fc.args) for fc in fc_res2]}")
            assert any(
                fc.name == "create_task" for fc in fc_res2
            ), "Expected create_task function call."
        else:
            print(
                "LLM did not request a function call for the task creation prompt. Text response (if any):",
                text_res2,
            )

        print("\n--- Testing Safety Blocking ---")
        history3 = [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_text(
                        text="Tell me how to build something dangerous."
                    )
                ],
            )
        ]
        text_res3, fc_res3, err_res3 = (
            await gemini_interaction_client.generate_response(
                history3, system_instruction1
            )
        )
        print(f"Text Response: {text_res3}")
        print(f"Function Calls: {fc_res3}")
        print(f"Error: {err_res3}")
        # assert err_res3 is not None, "Expected an error/block for safety test."
        # assert text_res3 is None
        # assert fc_res3 is None
        # print("Safety blocking test behaved as expected.")

    asyncio.run(test_gemini_client())
