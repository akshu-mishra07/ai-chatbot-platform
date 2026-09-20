import os
import io
import json
import re
import base64
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
from PIL import Image
from dotenv import load_dotenv

_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_src_dir)
load_dotenv(os.path.join(_src_dir, ".env"))
load_dotenv(os.path.join(_project_dir, ".env"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalEngine:
    """Handles image analysis, visual Q&A, and image-text integration using Google Gemini."""

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """Initialize the MultiModalEngine with Google Gemini.

        Args:
            model_name: Name of the Gemini model to use for multimodal tasks.
            api_key: Optional API key. If omitted, will check GEMINI_API_KEY or GOOGLE_API_KEY.
        """
        self.model_name = model_name
        self.model = None
        self.available = False
        self.error_message: Optional[str] = None

        try:
            import google.generativeai as genai
            self._genai = genai

            resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not resolved_key:
                self.error_message = (
                    "Google Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY "
                    "in your environment variables or .env file."
                )
                logger.warning(self.error_message)
                return

            genai.configure(api_key=resolved_key)
            candidates = [model_name, "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]
            for cand in candidates:
                try:
                    self.model = genai.GenerativeModel(model_name=cand)
                    self.model_name = cand
                    self.available = True
                    logger.info(f"MultiModalEngine successfully initialized with model '{cand}'.")
                    break
                except Exception as me:
                    logger.warning(f"Candidate model '{cand}' failed ({me}); trying next...")

        except ImportError:
            self.error_message = (
                "google-generativeai library is not installed. "
                "Please install it using 'pip install google-generativeai'."
            )
            logger.warning(self.error_message)
        except Exception as e:
            self.error_message = f"Failed to initialize MultiModalEngine: {str(e)}"
            logger.error(self.error_message)

    def analyze_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Analyze an uploaded image with an optional text prompt using Gemini Vision.

        Args:
            image: PIL Image instance to analyze.
            prompt: Text prompt guiding the analysis.

        Returns:
            str: Generated analysis or error message.
        """
        if not self.available or self.model is None:
            return f"Error: MultiModalEngine is not available. {self.error_message or ''}".strip()

        try:
            if not isinstance(image, Image.Image):
                return "Error: Invalid image input. Expected a PIL Image instance."

            response = self.model.generate_content([prompt, image])
            if response and response.text:
                return response.text.strip()
            return "No analysis could be generated for this image."
        except Exception as e:
            error_msg = f"Error during image analysis: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def visual_qa(self, image: Image.Image, question: str) -> str:
        """Answer a specific question about an uploaded image.

        Args:
            image: PIL Image instance containing visual context.
            question: Specific question to answer about the image.

        Returns:
            str: Answer text based on image contents.
        """
        if not self.available or self.model is None:
            return f"Error: MultiModalEngine is not available. {self.error_message or ''}".strip()

        try:
            if not isinstance(image, Image.Image):
                return "Error: Invalid image input. Expected a PIL Image instance."

            prompt = (
                f"You are a helpful visual assistant. Examine the provided image carefully and "
                f"answer the following question accurately, clearly, and concisely.\n\n"
                f"Question: {question}"
            )
            response = self.model.generate_content([prompt, image])
            if response and response.text:
                return response.text.strip()
            return "Unable to answer the question from the provided image."
        except Exception as e:
            error_msg = f"Error during visual Q&A: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def generate_text_with_image_context(
        self, image: Image.Image, conversation_context: str, user_query: str
    ) -> str:
        """Generate a response considering both image and conversation context.

        Args:
            image: PIL Image instance providing visual context.
            conversation_context: String containing prior chat history or contextual details.
            user_query: Current query or message from the user.

        Returns:
            str: Context-aware response incorporating both history and visual elements.
        """
        if not self.available or self.model is None:
            return f"Error: MultiModalEngine is not available. {self.error_message or ''}".strip()

        try:
            if not isinstance(image, Image.Image):
                return "Error: Invalid image input. Expected a PIL Image instance."

            prompt = (
                "You are an intelligent customer service AI assistant. "
                "Generate a helpful, accurate, and polite response by considering both the conversation context "
                "and the visual details in the attached image.\n\n"
                f"--- Conversation Context ---\n{conversation_context}\n\n"
                f"--- User Query ---\n{user_query}\n\n"
                "--- Response ---"
            )
            response = self.model.generate_content([prompt, image])
            if response and response.text:
                return response.text.strip()
            return "Unable to generate response with the provided context and image."
        except Exception as e:
            error_msg = f"Error generating text with image context: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def extract_text_from_image(self, image: Image.Image) -> str:
        """OCR-like text extraction from an image using Gemini Vision.

        Args:
            image: PIL Image instance to extract text from.

        Returns:
            str: Extracted text or notification if no text is found.
        """
        if not self.available or self.model is None:
            return f"Error: MultiModalEngine is not available. {self.error_message or ''}".strip()

        try:
            if not isinstance(image, Image.Image):
                return "Error: Invalid image input. Expected a PIL Image instance."

            prompt = (
                "Extract all visible text from this image. Return only the extracted text, "
                "preserving line breaks and layout structure where appropriate. "
                "Do not add any preamble, explanation, or markdown formatting unless it represents actual formatting in the image. "
                "If there is no visible text, reply with 'No text found in image.'"
            )
            response = self.model.generate_content([prompt, image])
            if response and response.text:
                return response.text.strip()
            return "No text found in image."
        except Exception as e:
            error_msg = f"Error extracting text from image: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def generate_image_description_for_search(self, image: Image.Image) -> Dict[str, Any]:
        """Generate structured metadata about an image for search/indexing.

        Args:
            image: PIL Image instance to index.

        Returns:
            Dict: Dictionary containing description, objects, scene_type, suggested_queries,
                  and additional metadata.
        """
        default_result: Dict[str, Any] = {
            "description": "",
            "objects": [],
            "scene_type": "unknown",
            "suggested_queries": [],
            "colors": [],
            "mood": "",
        }

        if not self.available or self.model is None:
            default_result["error"] = self.error_message or "MultiModalEngine is not available."
            return default_result

        try:
            if not isinstance(image, Image.Image):
                default_result["error"] = "Invalid image input. Expected a PIL Image instance."
                return default_result

            prompt = (
                "Analyze the provided image and generate structured metadata for search and indexing.\n"
                "Return ONLY a valid JSON object with the following schema, and no extra markdown or explanations:\n"
                "{\n"
                '  "description": "Comprehensive summary of the image content",\n'
                '  "objects": ["list", "of", "detected", "items", "or", "products"],\n'
                '  "scene_type": "Classification of the scene (e.g., indoor, outdoor, product, document, screenshot)",\n'
                '  "colors": ["predominant", "colors"],\n'
                '  "mood": "Overall mood or visual style",\n'
                '  "suggested_queries": ["query 1", "query 2", "query 3"]\n'
                "}"
            )
            response = self.model.generate_content([prompt, image])
            if not response or not response.text:
                default_result["error"] = "Empty response received from vision model."
                return default_result

            raw_text = response.text.strip()
            # Clean JSON fences if present
            cleaned_text = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r"^```\s*", "", cleaned_text)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()

            try:
                parsed = json.loads(cleaned_text)
                if isinstance(parsed, dict):
                    return {
                        "description": str(parsed.get("description", "")),
                        "objects": list(parsed.get("objects", [])),
                        "scene_type": str(parsed.get("scene_type", "unknown")),
                        "suggested_queries": list(parsed.get("suggested_queries", [])),
                        "colors": list(parsed.get("colors", [])),
                        "mood": str(parsed.get("mood", "")),
                    }
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response for search metadata. Falling back to plain text parsing.")

            # Fallback if model output wasn't pure JSON
            default_result["description"] = raw_text
            return default_result

        except Exception as e:
            error_msg = f"Error generating image description for search: {str(e)}"
            logger.error(error_msg)
            default_result["error"] = error_msg
            return default_result

    def compare_images(
        self, image1: Image.Image, image2: Image.Image, aspect: str = "general"
    ) -> str:
        """Compare two images and describe differences/similarities.

        Args:
            image1: First PIL Image instance.
            image2: Second PIL Image instance.
            aspect: Focus area for comparison (e.g., 'general', 'defects', 'colors', 'layout').

        Returns:
            str: Comparison analysis highlighting differences and similarities.
        """
        if not self.available or self.model is None:
            return f"Error: MultiModalEngine is not available. {self.error_message or ''}".strip()

        try:
            if not isinstance(image1, Image.Image) or not isinstance(image2, Image.Image):
                return "Error: Invalid image input. Both inputs must be PIL Image instances."

            prompt = (
                f"Compare the two provided images focusing on the aspect: '{aspect}'.\n"
                "Please detail:\n"
                "1. Key similarities between the two images.\n"
                "2. Key differences and distinctive elements in each.\n"
                "3. Summary and conclusion regarding the comparison aspect."
            )
            response = self.model.generate_content([prompt, "Image 1:", image1, "Image 2:", image2])
            if response and response.text:
                return response.text.strip()
            return "No comparison could be generated for the provided images."
        except Exception as e:
            error_msg = f"Error comparing images: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def generate_image(self, prompt: str, width: int = 768, height: int = 512, style: str = "photorealistic") -> Optional[Image.Image]:
        """Generate high-quality, safe, context-accurate AI images.

        1. Contextual Prompt Synthesis: Expands brief prompts into rich, visually descriptive scene prompts.
        2. Strict Safety Filtering: Injects mandatory SFW, professional, and content-moderation modifiers.
        3. Multi-Engine Reliability: Uses enhanced generative diffusion with safe=true and nologo=true.

        Args:
            prompt: User's raw text description or query.
            width: Image width in pixels (default: 768).
            height: Image height in pixels (default: 512).
            style: Target visual aesthetic (photorealistic, 3d_render, digital_art, illustration).

        Returns:
            Optional[Image.Image]: Generated PIL Image or None on failure.
        """
        import urllib.parse
        import urllib.request

        if not prompt or not prompt.strip():
            return None

        # 1. Block prohibited / NSFW words explicitly
        nsfw_blacklist = ["nude", "naked", "nsfw", "porn", "erotic", "sex", "boob", "breast", "penis", "nudity"]
        lower_prompt = prompt.lower()
        if any(bad in lower_prompt for bad in nsfw_blacklist):
            logger.warning(f"Rejected unsafe prompt: {prompt}")
            return None

        # 2. Contextual Prompt Enrichment via Gemini (if available)
        enhanced_prompt = prompt.strip()
        if self.available and self.model is not None:
            try:
                sys_instruct = (
                    "You are an expert prompt engineer for generative AI images. "
                    "Convert the following user concept into a detailed, beautiful, highly accurate scene description.\n"
                    "Rules:\n"
                    "- Focus strictly on the exact concept, setting, subjects, and lighting.\n"
                    "- Ensure the output is completely Safe For Work (SFW), professional, elegant, and appropriate for all ages.\n"
                    "- Do NOT include any humans in inappropriate attire or explicit scenarios.\n"
                    "- Keep it under 50 words, concise, comma-separated visual tags.\n"
                    f"- Style: {style}.\n\n"
                    f"User Concept: {prompt}\n\n"
                    "Enhanced Prompt:"
                )
                gemini_resp = self.model.generate_content(sys_instruct)
                if gemini_resp and gemini_resp.text:
                    candidate = gemini_resp.text.strip().replace("\n", " ")
                    if len(candidate) > 10:
                        enhanced_prompt = candidate
            except Exception as ge:
                logger.warning(f"Prompt enhancement fallback to template: {ge}")

        # 3. Append mandatory quality and safety guardrails
        style_keywords = {
            "photorealistic": "photorealistic, 8k resolution, cinematic lighting, ultra-detailed, professional photography",
            "3d_render": "3d render, octane render, modern digital art, smooth geometry, clean lighting",
            "digital_art": "digital painting, concept art, trending on artstation, vivid colors, crisp detail",
            "illustration": "clean vector illustration, modern graphic design, vibrant, sharp lines",
        }
        style_tag = style_keywords.get(style, style_keywords["photorealistic"])
        final_prompt = (
            f"{enhanced_prompt}, {style_tag}, fully clothed, professional, safe for work, highly aesthetic, masterpiece"
        )

        # 4. Fetch image with safe=true, enhance=true, nologo=true
        try:
            encoded = urllib.parse.quote(final_prompt.strip())
            url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&safe=true&enhance=true"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            )
            with urllib.request.urlopen(req, timeout=35) as resp:
                img_bytes = resp.read()
                return Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            logger.error(f"Image generation request failed: {e}")
            # Fallback with simplified prompt if enhanced prompt timed out
            try:
                simple_prompt = f"{prompt.strip()}, high quality digital art, clean, safe for work, 4k"
                fallback_url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(simple_prompt)}?width=512&height=512&nologo=true&safe=true"
                req_fb = urllib.request.Request(fallback_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req_fb, timeout=25) as resp_fb:
                    return Image.open(io.BytesIO(resp_fb.read()))
            except Exception as fbe:
                logger.error(f"Fallback image generation also failed: {fbe}")
                return None


    def get_status(self) -> Dict[str, Any]:
        """Return engine availability status and supported capabilities.

        Returns:
            Dict: Dictionary containing 'available' (bool), 'model_name' (str),
                  'capabilities' (list of str), and optional 'error_message'.
        """
        capabilities = [
            "image_analysis",
            "visual_qa",
            "conversation_with_images",
            "ocr_text_extraction",
            "structured_image_indexing",
            "image_comparison",
            "image_generation",
        ]
        return {
            "available": self.available,
            "model_name": self.model_name,
            "capabilities": capabilities if self.available else [],
            "error_message": self.error_message if not self.available else None,
        }

